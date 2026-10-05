# Attention Kernel Walkthrough

This document breaks down exactly how the custom attention path works under the hood, from QKV projection through the KV cache, fused attention, and output projection.

Attention lets each token combine information from earlier tokens. This directory has two implementations of the attention calculation: `flash_attention_kernel` processes prefill with Tensor Core tiles, while the decode kernels handle one token or a small group of speculative verification tokens.

The mathematical formula is:
1. **Project:** $[Q, K, V] = XW_{\text{qkv}}^T + b_{\text{qkv}}$
2. **Rotate:** Apply RoPE (Rotary Position Embeddings) to $Q$ and $K$.
3. **Score:** $S_{r,t} = Q_r K_t^T / \sqrt{D}$, with future positions masked to $-\infty$.
4. **Attend:** $O = \operatorname{softmax}(S)V$
5. **Output Projection:** $Y = \operatorname{concat\_heads}(O)W_o^T$

For Qwen2.5-7B, the hidden size is 3584, there are **28 query heads**, **4 KV heads**, and **128 elements per head**. Seven query heads share each KV head; this is **Grouped Query Attention (GQA)**.

---

## 1. Phase 1: The Fused QKV Projection
`qkv_proj_forward` takes `[M, 3584]` input, where `M = batch * seq_len`, and performs one cuBLAS matrix multiplication for all three projections.

The combined weight has 4608 rows:
*   **Q:** 28 heads × 128 elements = 3584 columns of output.
*   **K:** 4 heads × 128 elements = 512 columns.
*   **V:** Another 512 columns.

```cpp
auto y = at::empty({M, N}, b_qkv.options());
y.copy_(b_qkv);
```

The launcher first broadcasts the QKV bias into the output. Its `cublasGemmEx` call uses `alpha = 1` and `beta = 1`, so the matrix product is added to that bias. Inputs and output are FP16, with FP32 accumulation.

cuBLAS uses column-major views, so the call swaps the output dimensions and transposes the weight view to write the desired row-major result without a separate transpose operation.

The executor splits the output into contiguous `Q:[B, 28, S, 128]` and `K/V:[B, 4, S, 128]` tensors. Here, `S` is the number of new tokens in this call.

---

## 2. Phase 2: RoPE and the KV Cache Write
`rope_kv_write_kernel` launches **one block per `(batch, kv_head, 16-row sequence tile)`**, with **256 threads per block**. The final tile masks rows past the input length. At batch one and 1,742 input rows, the target launches 436 blocks instead of four; single-token decode still uses four blocks.

**Step 2a: Copy V**
V is not rotated. Its pointers are viewed as `int4*`, so each copy moves 128 bits, or eight FP16 values. These are raw bits; the cast does not numerically convert V into integers.

**Step 2b: Rotate K While Writing It**
RoPE pairs element `d` with element `d + D/2`, rather than pairing adjacent elements. Each thread converts a pair to FP32, rotates it, and rounds the results back to FP16 directly in the cache:

```cpp
cache_k[doff + d]        = __float2half(x0 * c - x1 * sn);
cache_k[doff + d + half] = __float2half(x1 * c + x0 * sn);
```

The cache tensors are contiguous `[B, H_kv, max_seq, D]` buffers. This kernel writes rows `[write_pos, write_pos + S)` using direct offsets; there is no page-table lookup in this implementation.

The cosine and sine tensors have shape `[B, S, D]`. Their values already encode the new tokens' absolute positions, while the kernel indexes their rows locally from zero. `write_pos` shifts the cache destination, not the cosine/sine lookup. Fusing rotation with the write avoids storing and re-reading a separate rotated K tensor.

---

## 3. Prefill Grid and Block Mapping
`fused_attn_forward` launches `flash_attention_kernel` with this fixed geometry:
*   **1 Block = 64 Query Rows for One Head:** `blockIdx.x` selects a group of 64 query tokens, `blockIdx.y` selects the query head, and `blockIdx.z` selects the batch item.
*   **128 Threads = 4 Warps:** Each warp owns 16 query rows.
*   **32 KV Rows per Iteration:** The block streams the cache in tiles of 32 keys and values.
*   **Head Dimension = 128:** This target kernel is compiled for `D = 128`.

For `B = 1` and `S = 128`, we launch `2 * 28 = 56` blocks. Each block determines its shared KV head with:

```cpp
const int kv_head = head / (h_q / h_kv);
```

Heads 0 through 6 read KV head 0, heads 7 through 13 read KV head 1, and so on. This avoids expanding the cache into 28 physically repeated KV heads, although separate query-head blocks still load their own tiles.

---

## 4. Phase 3: Shared-Memory Tiles and RoPE on Q
The block loads its `[64, 128]` Q tile using `int4` copies and keeps it in shared memory across the KV loop. Rows beyond `seq_len` in a partial final tile are zero-filled and never written to the output.

If cosine and sine tensors are supplied, the block applies the same `rotate_half` operation to Q **inside shared memory**. This avoids an extra global-memory pass over Q. If they are omitted, the kernel uses Q as supplied.

The main shared-memory buffers are:
*   **Q Tile:** `[64, 128]` halves = 16 KiB.
*   **V Tile:** `[32, 128]` halves = 8 KiB.
*   **K / Probability Storage:** A union that reserves 8 KiB for K, then reuses the space for probabilities after K has been consumed.

Each KV iteration loads a new K tile and V tile. A block-wide `__syncthreads()` makes those values visible before the warps use them.

---

## 5. Phase 4: Tensor Core Scores and Causal Masking
Each warp computes a `[16, 32]` score tile using **WMMA**, CUDA's warp-level matrix multiplication interface. It holds two `[16, 16]` score fragments, accumulates across eight 16-element chunks of the head dimension, and keeps the sums in FP32.

```cpp
wmma::mma_sync(S_frag[j], Q_frag, K_frag, S_frag[j]);
```

K is loaded through a column-major fragment view, giving the transpose needed for `Q @ K.T`. After multiplication, the code extracts scores from the fragments and multiplies them by `softmax_scale` (the executor supplies $1/\sqrt{128}$).

For local query row `r`, its absolute position is `cur_len - seq_len + r`. The kernel masks any key after that position, plus any key beyond the valid cache length:

```cpp
if (k_pos > q_pos_a || k_pos >= cur_len) s_a[c] = -INFINITY;
```

For example, three new queries with `cur_len = 103` sit at positions 100, 101, and 102. The first query can see keys 0 through 100, including itself, but cannot see keys 101 or 102.

The code explicitly maps accumulator entries through `S_frag[j].x[...]`. Each four-lane group handles a pair of rows eight positions apart, with each lane holding eight of the 32 scores for each row. Shuffle operations with XOR offsets 1 and 2 combine the four lanes' maxima and sums. This fragment mapping is an implementation-specific detail to revisit when porting the kernel to a different GPU architecture.

---

## 6. Phase 5: Online Softmax
We need a softmax over the entire valid cache, but only 32 keys fit in the current tile. Writing the full score matrix to global memory would defeat the purpose of tiling.

Instead, each query row keeps three running quantities:
*   **`m`:** The largest score seen so far.
*   **`l`:** The sum of exponentials measured relative to `m`.
*   **`O_frag`:** The unnormalized, exponentially weighted sum of V.

When a new tile arrives, the update is:
1. **New Maximum:** $m' = \max(m, \max_j s_j)$
2. **Rescale Old Work:** $\alpha = e^{m-m'}$
3. **Tile Weights:** $p_j = e^{s_j-m'}$
4. **New Denominator:** $l' = \alpha l + \sum_j p_j$
5. **New Weighted Sum:** $A' = \alpha A + \sum_j p_j V_j$

The rescaling keeps earlier tiles and the current tile on the same numerical scale. Subtracting the running maximum also keeps the exponentials from growing with large positive scores. Initially, `m = -INFINITY`, `l = 0`, and the output accumulator is zero; the code handles that first update explicitly.

Masked scores contribute zero weight. The full attention matrix is never materialized in global memory.

---

## 7. Phase 6: Probability × Value and Write-Back
After all warps finish reading K, its shared-memory storage is reused for the current tile's unnormalized probabilities. They are rounded to FP16 and placed in a `[64, 40]` buffer: 32 useful columns plus 8 padding columns, with `P_STRIDE = 40` used by the WMMA loads.

Each warp then computes `P @ V` using two 16-element partitions of the KV tile and eight 16-column output fragments. The results accumulate into the rescaled FP32 `O_frag` values. A block barrier prevents the next KV iteration from overwriting shared buffers too soon.

After the final tile, the kernel divides each output row by its accumulated denominator `l`. It converts the normalized FP32 values to FP16 and writes them through `__half2` stores, skipping padded query rows.

The result has shape `[B, 28, S, 128]`. Scores, maxima, denominators, and output accumulation use FP32, while the Tensor Core probability input is FP16; this is not an entirely FP32 attention calculation.

---

## 8. Decode and Verification Grid Mapping
A 64-query prefill tile is a poor fit when we only have one new token. The decode path instead uses `decode_attn_kernel`, or its split-KV version, with ordinary FP32 arithmetic and **128 threads per block**.

`decode_attn_forward` accepts exactly **one query token**. `small_q_attn_forward` accepts **1 through 9 query tokens**, with a separate compile-time specialization for each length. Both use the same launch helper and can select either decode implementation.

In the single-block version, **one block owns one `(batch, query_head)`** and streams 32 KV rows at a time. Q is loaded once into shared memory and optionally rotated there. The block also keeps shared FP32 scores, output accumulators, and one running softmax state per query row. The scalar kernels pad their K tile to `[32][D + 2]`: neighboring threads reading different keys at the same channel now access successive shared-memory banks rather than the same bank. The WMMA prefill layout is unchanged.

The new K/V rows must already be in the cache. Query row `r` represents position `cur_len - Q_TOKENS + r`, so the same causal rule applies during speculative verification: a candidate can see earlier candidates and itself, but not later candidates.

---

## 9. Phase 7: Decode Scores, Softmax, and Values
The decode kernel changes thread ownership between stages:

**Step 9a: One Thread per Query/Key Pair**
Each worker computes a complete 128-element dot product in FP32 and writes one scaled, masked score into shared memory. No cross-thread dot-product reduction is needed.

**Step 9b: One Thread per Query Row**
The first `Q_TOKENS` threads each scan their row's 32 scores, update the running maximum and denominator, and overwrite the score buffer with unnormalized probabilities. These probabilities remain FP32.

**Step 9c: One Thread per Output Channel**
With 128 threads and `D = 128`, thread `d` owns output channel `d`. It rescales the previous accumulator, loops over the 32 values for that channel, and updates every query row's weighted sum.

Block barriers separate the stages because their shared-memory results are read by different threads. Once all KV tiles have been processed, the kernel divides by the row denominator and writes FP16 output.

---

## 10. Phase 8: Splitting the KV Cache Across Blocks
At batch size 1, the single-block decode path only launches 28 blocks. `decode_attn_split_kernel` creates more parallel work by dividing the cache range across `grid.x`:
*   **`grid.x`:** KV split.
*   **`grid.y`:** Query head.
*   **`grid.z`:** Batch item.

`choose_num_splits` targets 144 blocks, limits splitting using `ceil(cur_len / 256)`, and caps the count at 32. For example, `B = 1`, 28 heads, and `cur_len = 2048` select **6 splits**, giving 168 blocks. When the chosen count is 1, the launcher uses the single-block kernel directly.

The split kernel rounds `ceil(cur_len / num_splits)` up to a multiple of 32. Each block runs the same online-softmax algorithm over only its assigned range, then writes **unnormalized FP32 partials** `(m, l, acc)` to scratch memory.

`decode_attn_combine_kernel` launches one 128-thread block per `(head, batch)` and merges the partials:
1. **Shared Maximum:** $m = \max_s m_s$
2. **Combined Denominator:** $l = \sum_s e^{m_s-m}l_s$
3. **Final Output:** $O = \left(\sum_s e^{m_s-m}A_s\right)/l$

We cannot simply average the partial outputs: each split has its own maximum and denominator. Rescaling makes them comparable before summation. Empty splits write `m = -INFINITY`, `l = 0`, and zero accumulators, and the combine kernel skips them. Each block owns its scratch slot, so no atomics are needed.

---

## 11. Phase 9: Device Scalars for CUDA Graph Replay
The `_dev` entry points keep changing positions in one-element CUDA `int64` tensors:
*   `rope_kv_write_forward_dev` reads `write_pos` on the GPU.
*   `decode_attn_forward_dev` and `small_q_attn_forward_dev` read `cur_len` on the GPU.

```cpp
if (cur_len_ptr) cur_len = static_cast<int>(*cur_len_ptr);
```

A host integer would be fixed at CUDA graph capture time. Reading a device scalar allows replay to use an updated sequence length without rebuilding the graph. The executor also refreshes the cosine/sine buffers for the current positions.

For this path, the launcher chooses the split count from the fixed cache capacity `max_seq`. The split kernel derives the actual range lengths from the current device value of `cur_len`, keeping the grid and scratch shapes fixed across replays.

The caller must maintain valid position bounds: `write_pos + S <= max_seq` and `S <= cur_len <= max_seq`, with a nonnegative write position. The device-scalar launchers do not copy these values back to the CPU to check them.

---

## 12. Phase 10: Output Projection
The executor transposes and reshapes the attention result from `[B, 28, S, 128]` to `[B * S, 3584]`, concatenating the query heads for each token.

`o_proj_forward` then calls cuBLAS for `x @ W_o.T`. As with QKV projection, it uses FP16 inputs/output and FP32 accumulation, but the output projection has **no bias** and sets `beta = 0`.

The returned `[B, S, 3584]` tensor is the attention contribution to the decoder layer. Adding it to the incoming residual stream happens afterward in `residual_add_forward`.
