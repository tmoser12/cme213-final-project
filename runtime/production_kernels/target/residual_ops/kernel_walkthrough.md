# Residual Operations Kernel Walkthrough

This document breaks down exactly how the custom `residual_add_kernel` and the `lm_head_forward` launcher work under the hood.

This directory contains two operations that connect the larger model components. Residual addition folds the attention and MLP outputs back into the hidden state. The LM head turns the final normalized hidden state into one score for every vocabulary token.

The mathematical formulas are:
1. **Residual Addition:** $Y_i = A_i + B_i$
2. **Vocabulary Projection:** $\text{logits} = XW_{\text{lm}}^T$

For Qwen2.5-7B, $X$ has hidden size 3584 and $W_{\text{lm}}$ has shape `[152064, 3584]`. The residual addition is a custom element-wise CUDA kernel; the vocabulary projection is a cuBLAS matrix multiplication.

---

## 1. Grid and Block Mapping
The residual kernel treats the input tensors as flat buffers:
*   **1 Thread = 8 FP16 Elements per Iteration:** Each thread adds one 128-bit vector from `a` to one from `b`.
*   **256 Threads per Block:** A full block handles 2048 elements on its first iteration.
*   **Grid-Stride Loop:** The launcher requests `ceil((numel / 8) / 256)` blocks, capped at 65535. Threads loop over further vectors if the cap is reached.

For a single 3584-element token, we have 448 vectors and launch **2 blocks**. The first block handles 256 vectors and the second handles the remaining 192. Blocks are assigned flat chunks of elements rather than whole tokens.

The inputs must have identical shapes, be contiguous CUDA FP16 tensors on the same device, and contain a total number of elements divisible by 8. There is no broadcasting or scalar tail path.

---

## 2. Phase 1: The 128-bit Vectorized Reads
```cpp
float4 av = a4[i];
float4 bv = b4[i];
float4 ov;

half2* ah = reinterpret_cast<half2*>(&av);
half2* bh = reinterpret_cast<half2*>(&bv);
half2* oh = reinterpret_cast<half2*>(&ov);
```

The input pointers were cast from `__half*` to `float4*`, so each load brings in eight FP16 values. As in RMSNorm, `float4` carries the bits; it does not change their numerical type.

Each thread then views its two input vectors as four `half2` pairs. Neighboring threads read neighboring vectors, keeping the global-memory accesses contiguous across the warp.

Residual addition does very little computation for the amount of data it moves: two input reads and one output write per element. Vectorization reduces the number of memory instructions needed for this simple operation.

---

## 3. Phase 2: Native FP16 Addition
```cpp
#pragma unroll
for (int j = 0; j < 4; ++j) {
    oh[j] = __hadd2(ah[j], bh[j]);
}
out4[i] = ov;
```

The intrinsic `__hadd2` adds two FP16 pairs, with each lane rounded to FP16. Four iterations therefore produce all eight output elements.

Unlike RMSNorm's sum of squares or SwiGLU's nonlinear activation, this operation does not explicitly convert the values to FP32. There is also no reduction: every output depends only on the two input values at the same index.

The thread packs the results into `ov` and writes the complete 128-bit vector back to global memory. No shared memory, atomics, or block-wide synchronization is needed.

---

## 4. Phase 3: Returning the Residual Stream
```cpp
auto out = torch::empty_like(a);
launch_residual_add(a.data_ptr<at::Half>(), b.data_ptr<at::Half>(),
                    out.data_ptr<at::Half>(), a.numel());
return out;
```

The public `residual_add_forward` allocates a fresh tensor, preserving both inputs. Each decoder layer uses the operation twice: once to add the attention output to the incoming residual, and once to add the MLP output to the updated residual.

RMSNorm remains a separate operation. The unnormalized sum continues along the residual stream, while a normalized version feeds the next sub-block. This launcher only materializes the sum; it does not normalize it.

---

## 5. Phase 4: The LM Head Vocabulary Projection
The second entry point, `lm_head_forward`, takes `[M, H]` hidden states and returns `[M, vocab]` logits. It calls `cublasGemmEx` with FP16 inputs and output and FP32 accumulation.

cuBLAS uses column-major matrix views. The launcher swaps the output dimensions and uses `CUBLAS_OP_T` for the weight view, effectively computing `logits.T = weight @ hidden.T` into the row-major output buffer. No explicit transpose copy is required.

```cpp
auto logits = at::empty({M, N}, hidden.options());
const float alpha = 1.0f;
const float beta  = 0.0f;
```

Here, `N` is the vocabulary size. There is no bias, and `beta = 0` means the matrix multiplication overwrites the uninitialized output. The target's LM-head weights are distinct from its embedding weights.

The Python wrapper can flatten `[B, S, H]` into `[B * S, H]` and restore `[B, S, vocab]` afterward. Every supplied row is projected; selecting which token positions to pass happens outside this operation. The result is raw FP16 logits, with no softmax, argmax, or sampling inside this launcher.
