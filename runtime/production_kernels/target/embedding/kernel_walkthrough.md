# Embedding Kernel Walkthrough

This document breaks down exactly how the custom `embedding_kernel` works under the hood.

The embedding is the first step in the model: it turns each integer token ID into a learned vector of hidden-dimension values. There is no matrix multiplication here. We simply look up a row in the embedding table and copy it into the output.

The mathematical formula is:
1. **Look Up the Token:** $t = \text{input\_ids}[n]$
2. **Copy Its Vector:** $\text{output}[n, i] = \text{weight}[t, i]$

Where $n$ is the flattened token index, $i$ is the hidden-dimension index, and `weight` has shape `[V, H]`. For the Qwen2.5-7B target, $V$ is 152064 and $H$ is 3584.

---

## 1. Grid and Block Mapping
The very first thing the kernel does is assign each output token to a block:
*   **1 Block = 1 Token:** `blockIdx.x` selects one of the `N = input_ids.numel()` tokens.
*   **128 Threads = Workers Across the Hidden Dimension:** `threadIdx.x` selects the first element each thread copies. The thread then advances by 128 elements at a time.

If we have a batch size of 2 and a sequence length of 128, we launch **256 blocks**, each with **128 threads**. With a hidden size of 3584, every thread copies exactly **28 FP16 elements** (3584 / 128 = 28).

Unlike RMSNorm, the launcher does not derive the thread count from the hidden size. It always uses 128 threads, and the loop handles the row width.

---

## 2. Phase 1: Finding the Source Row
```cpp
int64_t n = blockIdx.x;
if (n >= N) return;

int64_t id = input_ids[n];
const __half* src = weight + id * H;
__half*       dst = output + n  * H;
```

The block first reads the integer token ID. Because the embedding table is contiguous, multiplying that ID by `H` takes us directly to the beginning of its learned vector.

Notice that the source row uses `id`, while the destination row uses `n`. For example, if token position 3 contains ID 42, we copy **weight row 42 into output row 3**. Repeated IDs can read the same source row, but every block still writes its own output row.

The caller must supply IDs in `[0, V)`: this kernel does not check each token ID before using it as an address.

---

## 3. Phase 2: Coalesced Reads and Writes
```cpp
for (int i = threadIdx.x; i < H; i += blockDim.x) {
    dst[i] = src[i];
}
```

Threads on an NVIDIA GPU execute in groups of 32 called **Warps**. On the first loop iteration, threads 0 through 31 copy elements 0 through 31; on the next iteration, they copy elements 128 through 159.

In both cases, neighboring threads access neighboring FP16 values. These are **Coalesced Memory Accesses**: the GPU can serve the warp's adjacent addresses with a small number of memory transactions.

The token IDs themselves may point to unrelated vocabulary rows, but the access *within* each row is contiguous. This is where the kernel gets its useful memory-access pattern. The code uses scalar `__half` copies rather than the explicit `float4` vectorization used by RMSNorm.

---

## 4. Phase 3: Independent Copying, No Reduction
Each output element has exactly one writer. No thread needs another thread's result, so there is no shared-memory buffer, warp reduction, or `__syncthreads()` inside the kernel.

This operation is fundamentally a **Memory Copy with Indexed Source Rows**. Each token copies 3584 halves, or 7168 bytes, into its output row. There is no arithmetic on those values and no FP16-to-FP32 conversion, so the copied bit patterns are preserved exactly.

For hidden sizes that are not multiples of 128, the `i < H` condition makes the final loop iteration safe: only threads whose element is still inside the row perform a copy.

---

## 5. Phase 4: Returning the Original Token Layout
```cpp
auto out_shape = input_ids.sizes().vec();
out_shape.push_back(H);
auto output = torch::empty(out_shape, weight.options());
```

The host launcher accepts contiguous CUDA `int64` IDs and a contiguous CUDA FP16 weight matrix on the same device. It appends the hidden dimension to the input shape: `[B, S]` IDs become `[B, S, H]` embeddings, even though the kernel treats the token positions as one flat list.

For an empty input, it returns the correctly shaped empty tensor without launching a kernel. Otherwise, it launches the copy on PyTorch's current CUDA stream and returns the embeddings ready for the first decoder layer.
