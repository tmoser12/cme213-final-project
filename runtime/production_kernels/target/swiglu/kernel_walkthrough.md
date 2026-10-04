# SwiGLU Kernel Walkthrough

This document breaks down exactly how `swiglu_forward` and the custom `swiglu_act_kernel` work under the hood.

SwiGLU is the feed-forward part of each Qwen decoder layer. Two learned projections expand the hidden state, an activation gates one projection with the other, and a final projection brings the result back to the hidden size.

The mathematical formula is:
1. **Gate Projection:** $G = XW_{\text{gate}}^T$
2. **Up Projection:** $U = XW_{\text{up}}^T$
3. **Gated Activation:** $A = \operatorname{SiLU}(G) \odot U$, where $\operatorname{SiLU}(g) = g / (1 + e^{-g})$
4. **Down Projection:** $Y = AW_{\text{down}}^T$

Where $X$ has shape `[M, H]`, $M$ is the number of tokens, and $H$ is 3584 for Qwen2.5-7B. The expanded intermediate size $I$ is 18944, so both $G$ and $U$ have shape `[M, 18944]`.

---

## 1. Phase 1: The cuBLAS Projections
```cpp
auto gate = linear_no_bias(x, W_gate);
auto up   = linear_no_bias(x, W_up);
```

The large matrix multiplications are handled by `linear_no_bias`, which calls **cuBLAS**. The custom CUDA kernel handles the element-wise activation between them. The full MLP is therefore several GPU operations, even though Python exposes it through one `swiglu_forward` call.

The helper computes `y = x @ W.T` with FP16 inputs and output, using FP32 accumulation. Because cuBLAS interprets matrices as column-major, the call swaps the output dimensions and uses `CUBLAS_OP_T` for the weight view. This computes the transpose of the desired result into the same row-major output storage, without explicitly transposing the tensors.

All three projections are bias-free. Setting `beta = 0` tells cuBLAS to overwrite the newly allocated output instead of adding its previous contents.

---

## 2. Grid and Block Mapping
The activation treats the entire `[M, I]` buffer as a flat list:
*   **1 Thread = 8 FP16 Elements per Iteration:** One vector contains 128 bits, or eight halves.
*   **256 Threads per Block:** A full block processes 2048 elements on its first iteration.
*   **Grid-Stride Loop:** The launcher requests `ceil((numel / 8) / 256)` blocks, capped at 65535. Each thread advances by the total number of threads in the grid if more vectors remain.

Unlike RMSNorm, a block does not own exactly one token. For one token with 18944 intermediate elements, there are 2368 vectors and **10 blocks**; the final block has only 64 active threads.

The vector count must be an integer. The full MLP checks that `I` is a multiple of 8, and the launch helper checks that the total element count is a multiple of 8.

---

## 3. Phase 2: The 128-bit Vectorized Reads
```cpp
const float4* gate4 = reinterpret_cast<const float4*>(gate);
const float4* up4   = reinterpret_cast<const float4*>(up);
float4*       out4  = reinterpret_cast<float4*>(out);
```

Each thread loads one `float4` from the gate buffer and one from the up buffer. Here, `float4` is a **128-bit Storage Container**: the cast does not numerically convert eight halves into four floats.

This is the same vectorized memory pattern used by RMSNorm. Moving eight values together reduces the number of load/store instructions, while neighboring threads access neighboring vectors.

---

## 4. Phase 3: SiLU and Multiplication in FP32
```cpp
float2 gf = __half22float2(gh[j]);
float2 uf = __half22float2(uh[j]);
float2 of;
of.x = (gf.x / (1.0f + __expf(-gf.x))) * uf.x;
of.y = (gf.y / (1.0f + __expf(-gf.y))) * uf.y;
oh[j] = __float22half2_rn(of);
```

Inside each thread, the loaded bits are viewed as four `half2` pairs. The loop converts a pair from each input to FP32, evaluates SiLU with `__expf`, and multiplies by the corresponding up-projection values.

Finally, `__float22half2_rn` rounds the two results back to FP16. The nonlinear arithmetic happens in FP32; `half2` groups the input and output values in pairs.

Every element is independent, so this kernel needs no shared memory or parallel reduction. Fusing SiLU with multiplication also avoids writing a separate SiLU result to global memory and reading it back for the multiply.

---

## 5. Phase 4: Write-Back and Down Projection
```cpp
launch_swiglu_act(gate.data_ptr<at::Half>(), up.data_ptr<at::Half>(),
                  gate.data_ptr<at::Half>(), gate.numel());
auto out = linear_no_bias(gate, W_down);
```

The full MLP passes the gate buffer as the activation output. Each thread reads its gate and up vector before writing the result to that same gate location, reusing the `[M, I]` allocation. The kernel writes the eight activated values together through `out4[i] = o4`.

The down projection then multiplies this activated buffer by `W_down.T`, producing `[M, H]`. The Python wrapper also accepts `[B, S, H]`, flattening the token dimensions before the call and restoring them afterward.

The returned tensor is the MLP contribution. The residual addition is a separate operation in `residual_ops`. For isolated activation testing, the extension also exposes `swiglu_act_forward`, which allocates a fresh output instead of reusing the gate buffer.
