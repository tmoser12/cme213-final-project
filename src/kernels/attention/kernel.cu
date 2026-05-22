// src/kernels/attention/kernel.cu
// Custom CUDA attention sub-ops for Qwen2.5.
//
// Five host launchers (scaffold — __global__ bodies and cuBLAS calls not yet
// written). See src/kernels/attention/wrapper.py for orchestration.
//   1. qkv_proj_forward   — fused QKV projection (cuBLAS GEMM, fp16)
//   2. rope_forward       — in-place rotary on Q and K (custom kernel)
//   3. kv_write_forward   — scatter new K/V into paged cache (custom kernel)
//   4. fused_attn_forward — causal SDPA, GQA-aware (custom kernel)
//   5. o_proj_forward     — output projection (cuBLAS GEMM, fp16)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>     // at::cuda::getCurrentCUDABlasHandle
#include <c10/cuda/CUDAException.h>    // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cuda_fp16.h>  // fa2
#include <mma.h>        // fa2

using namespace nvcuda; // fa2


// Wrap every cuBLAS call. cuBLAS returns a status enum and never throws;
// without this macro a failure would silently produce garbage output.
// cublasGetStatusString lands in CUDA 11.4.2, available in our 12.1 toolkit.
#define CUBLAS_CHECK(expr)                                              \
    do {                                                                \
        cublasStatus_t _status = (expr);                                \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                   \
                    "cuBLAS error: ", cublasGetStatusString(_status));  \
    } while (0)



// cuBLAS QKV Projection
torch::Tensor qkv_proj_forward(torch::Tensor x,
                               torch::Tensor W_qkv,
                               torch::Tensor b_qkv) {
    TORCH_CHECK(x.is_cuda() && W_qkv.is_cuda() && b_qkv.is_cuda(),
                "qkv_proj: all tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && W_qkv.is_contiguous() && b_qkv.is_contiguous(),
                "qkv_proj: all tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                W_qkv.scalar_type() == torch::kHalf &&
                b_qkv.scalar_type() == torch::kHalf,
                "qkv_proj: all tensors must be float16");
    TORCH_CHECK(x.dim() == 2,     "qkv_proj: x must be 2-D [M, H]");
    TORCH_CHECK(W_qkv.dim() == 2, "qkv_proj: W_qkv must be 2-D [N, H]");
    TORCH_CHECK(b_qkv.dim() == 1, "qkv_proj: b_qkv must be 1-D [N]");
    TORCH_CHECK(W_qkv.size(1) == x.size(1),
                "qkv_proj: W_qkv inner dim must equal x hidden dim");
    TORCH_CHECK(b_qkv.size(0) == W_qkv.size(0),
                "qkv_proj: b_qkv length must equal W_qkv rows");

    const int64_t M = x.size(0);          // rows of X and rows of output Y (batch size * seq len)
    const int64_t K = x.size(1);          // hidden_size (3584 for Qwen2.5-7B)
    const int64_t N = W_qkv.size(0);      // H_q + 2*H_kv (4608 for 7B)

    // Initialize output y as broadcasted bias so the GEMM (beta=1) accumulates
    // into it. 
    auto y = at::empty({M, N}, b_qkv.options());
    y.copy_(b_qkv);  // PyTorch broadcasts [N] -> [M, N]

    // Borrow PyTorch's per-process cuBLAS handle. 
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Alpha/beta = 1, match compute dtype 
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                                    // op(A): transpose W_qkv (col-major view)
        CUBLAS_OP_N,                                    // op(B): leave X as-is
        static_cast<int>(N),                            // M and N are swapped since cuBLAS is column-major
        static_cast<int>(M),                            
        static_cast<int>(K),                         
        &alpha,
        W_qkv.data_ptr<at::Half>(), CUDA_R_16F,         // A pointer + dtype
        static_cast<int>(K),                            // lda: leading dim of A in col-major view
        x.data_ptr<at::Half>(),     CUDA_R_16F,         // B pointer + dtype
        static_cast<int>(K),                            // ldb: leading dim of B in col-major view
        &beta,
        y.data_ptr<at::Half>(),     CUDA_R_16F,         // C pointer + dtype
        static_cast<int>(N),                            // ldc: leading dim of C in col-major view
        CUBLAS_COMPUTE_32F,                             // fp16 mul, fp32 accumulate
        CUBLAS_GEMM_DEFAULT));                          // let cuBLAS pick the Tensor Core path

    return y;
}


// __global__ RoPE kernel (body TBD). Call this once for Q (with H = num_heads)
// and once for K (with H = num_kv_heads); the cos/sin tables are shared.
//
// Suggested launch geometry:
//   grid  = dim3(S, H, B)            // one block per (b, h, s) row
//   block = dim3(D / 2)              // each thread handles one (i, i + D/2) pair
//
// Math (matches HF apply_rotary_pos_emb at src/models/modeling_qwen2.py:237):
//   For i in [0, D/2):
//     x_new[b, h, s, i]       = x[b, h, s, i]       * cos[b, s, i] - x[b, h, s, i + D/2] * sin[b, s, i]
//     x_new[b, h, s, i + D/2] = x[b, h, s, i + D/2] * cos[b, s, i] + x[b, h, s, i]       * sin[b, s, i]
//   HF concatenates the same D/2 angles twice along the last dim, so
//   cos[b, s, i] == cos[b, s, i + D/2] (and likewise for sin). The kernel can
//   read just the first half.
__global__ void rope_kernel(
    __half*       __restrict__ x,     // [B, H, S, D] row-major; mutated in place
    const __half* __restrict__ cos,   // [B, S, D]   (last D/2 entries duplicate the first D/2)
    const __half* __restrict__ sin,   // [B, S, D]
    int H,                            // num_heads (Q) or num_kv_heads (K)
    int S,                            // seq_len
    int D                             // head_dim (must be even)
) {
    // TODO: implement. See math above.
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z * blockDim.y + threadIdx.y;
    int d = threadIdx.x;
    if (s >= S) return;

    int x_offset = ((b * H + h) * S + s) * D;

    float x_0 = __half2float(x[x_offset + d]);
    float x_1 = __half2float(x[x_offset + d + D/2]);

    int cos_offset = (b * S + s) * D;
    float cos_val = __half2float(cos[cos_offset + d]);
    float sin_val = __half2float(sin[cos_offset + d]);

    x[x_offset + d] = __float2half(x_0 * cos_val - x_1 * sin_val);
    x[x_offset + d + D/2] = __float2half(x_1 * cos_val + x_0 * sin_val);

}


// 2. In-place RoPE on Q and K.
//    q:[B, num_heads, S, D], k:[B, num_kv_heads, S, D], cos/sin:[..., D].
void rope_forward(torch::Tensor q,
                  torch::Tensor k,
                  torch::Tensor cos,
                  torch::Tensor sin) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "rope: all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "rope: q and k must be contiguous (mutated in place)");
    TORCH_CHECK(q.scalar_type() == torch::kHalf &&
                k.scalar_type() == torch::kHalf &&
                cos.scalar_type() == torch::kHalf &&
                sin.scalar_type() == torch::kHalf,
                "rope: all tensors must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4,
                "rope: q and k must be 4-D [B, H_*, S, D]");
    TORCH_CHECK(q.size(0) == k.size(0) &&
                q.size(2) == k.size(2) &&
                q.size(3) == k.size(3),
                "rope: q and k must share batch / seq / head_dim");
    TORCH_CHECK(cos.size(-1) == q.size(3) && sin.size(-1) == q.size(3),
                "rope: cos/sin last dim must equal head_dim");

    // RoPE kernel: rotates q and k in place.
    const int B = q.size(0);
    const int HQ = q.size(1);
    const int HKV = k.size(1);
    const int S = q.size(2);
    const int D = q.size(3);
    constexpr int TILE_SEQ = 8;

    dim3 grid(B, HQ, (S + TILE_SEQ - 1) / TILE_SEQ);
    dim3 block(D / 2, TILE_SEQ);

    auto* cos_ptr = reinterpret_cast<const __half*>(cos.data_ptr<at::Half>());
    auto* sin_ptr = reinterpret_cast<const __half*>(sin.data_ptr<at::Half>());

    // Q Rope first, 28 query heads
    rope_kernel<<<grid, block>>>(reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
                                 cos_ptr, sin_ptr, HQ, S, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // K Rope next, 4 key value heads
    dim3 grid_k(B, HKV, (S + TILE_SEQ - 1) / TILE_SEQ);
    dim3 block_k(D / 2, TILE_SEQ);
    rope_kernel<<<grid_k, block_k>>>(reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
                                     cos_ptr, sin_ptr, HKV, S, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// 3a. KV-cache scatter kernel.
//
// What this does: copies one contiguous slab of newly-projected K/V activations
// into the model's persistent KV cache at the current write offset. Each layer
// of Qwen2.5 owns one such cache; speculative decoding writes `gamma` new
// tokens per draft step, then either accepts them or rolls back the per-layer
// write pointer.
//
// Shapes (all fp16, row-major contiguous):
//   new_k,   new_v   : [B, H_kv, S,        D]   (the just-projected K, V)
//   cache_k, cache_v : [B, H_kv, max_seq,  D]   (the persistent cache)
//   target slice     : cache[:, :, write_pos : write_pos + S, :]
//
// Memory-layout trick. Within one (b, h_kv) "slab":
//   * src elements [s = 0..S, d = 0..D]                     -> S*D contiguous fp16
//   * dst elements [s = write_pos..write_pos+S, d = 0..D]   -> S*D contiguous fp16
// The cache's stride along the seq axis is exactly D, so the destination
// slice we want to write into is itself a contiguous chunk of memory --
// not a scatter across rows. So per slab the operation is a plain
// contiguous-to-contiguous memcpy. Only the *base* of each src/dst slab
// differs:
//     src_base[bh] = bh * (S       * D)
//     dst_base[bh] = bh * (max_seq * D) + write_pos * D
// where bh = b * H_kv + h_kv flattens the (b, h_kv) pair to a single index.
//
// Vectorization. D = 128 fp16 = 256 bytes is a multiple of 16 B, and
// PyTorch's caching allocator gives us at least 512 B alignment on every
// tensor base pointer. So we cast all four pointers to int4* (one int4 =
// 16 bytes = 8 packed fp16) and have each thread issue 16-byte loads and
// stores. Why this matters: a Turing warp issues a single coalesced
// transaction per memory instruction when its 32 threads touch contiguous
// addresses. With 2-byte loads (scalar fp16) the warp moves   64 B/issue;
// with 4-byte loads (__half2)            the warp moves        128 B/issue;
// with 16-byte loads (int4)              the warp moves        512 B/issue.
// More bytes in flight per memory instruction is what saturates the
// 672 GB/s HBM on a copy-only op like this one.
//
// Grid / block geometry.
//   gridDim.x  = B * H_kv   (one block per slab)
//   blockDim.x = 256        (256 threads grid-stride over the slab)
// Block-per-slab keeps the indexing trivial: blockIdx.x identifies which
// slab we own, and there is no inter-block coordination. Each thread
// independently copies one int4 of K and one int4 of V per loop iteration,
// then advances by blockDim.x int4s.
//
// Why fuse K and V into a single kernel?
//   1) For the tiny configs (e.g. B=1, S=1 -- 256 B of work total), kernel
//      launch latency (~3-5 us) dominates the runtime. Halving the number
//      of launches halves the dominant cost.
//   2) For the bandwidth-bound large configs, interleaving the two
//      independent load/store streams in the same thread doubles the
//      in-flight memory transactions per thread, improving memory-level
//      parallelism and helping saturate HBM.
__global__ void kv_write_kernel(
    const int4* __restrict__ new_k,        // src K, viewed as int4 (one int4 = 8 fp16)
    const int4* __restrict__ new_v,        // src V, same layout as new_k
    int4*       __restrict__ cache_k,      // dst K base ptr (slab + write offsets added below)
    int4*       __restrict__ cache_v,      // dst V base ptr
    int slab_vecs_in,                      // int4s per src slab    = S       * D / 8
    int slab_vecs_out,                     // int4s per dst slab    = max_seq * D / 8
    int write_off_vecs                     // int4 offset into dst slab = write_pos * D / 8
) {
    // Which head does this block own?
    const int bh = blockIdx.x;

    const int4* sk = new_k   + bh * slab_vecs_in; // source key, points to the head that this block is responsible for
    const int4* sv = new_v   + bh * slab_vecs_in; // source value, points to the head that this block is responsible for

    int4* dk = cache_k + bh * slab_vecs_out + write_off_vecs; // destination key, points to the head that this block is responsible for
    int4* dv = cache_v + bh * slab_vecs_out + write_off_vecs; // destination value, points to the head that this block is responsible for

    for (int i = threadIdx.x; i < slab_vecs_in; i += blockDim.x) {
        // 16-byte load + 16-byte store for K, same for V. The compiler
        // will pipeline these so the V-load issues before K's store has
        // to wait on the K-load to return -- giving us memory-level
        // parallelism within a single thread.
        dk[i] = sk[i];
        dv[i] = sv[i];
    }
}

// 3. KV write: scatter new_k/new_v into the cache_k/cache_v at [..., write_pos:write_pos+S, :]. 
// We should fuse this into QKV projection.
void kv_write_forward(torch::Tensor new_k,
                      torch::Tensor new_v,
                      torch::Tensor cache_k,
                      torch::Tensor cache_v,
                      int64_t write_pos) {
    TORCH_CHECK(new_k.is_cuda() && new_v.is_cuda() &&
                cache_k.is_cuda() && cache_v.is_cuda(),
                "kv_write: all tensors must be CUDA");
    TORCH_CHECK(new_k.is_contiguous() && new_v.is_contiguous() &&
                cache_k.is_contiguous() && cache_v.is_contiguous(),
                "kv_write: all tensors must be contiguous");
    TORCH_CHECK(new_k.scalar_type() == torch::kHalf &&
                new_v.scalar_type() == torch::kHalf &&
                cache_k.scalar_type() == torch::kHalf &&
                cache_v.scalar_type() == torch::kHalf,
                "kv_write: all tensors must be float16");
    TORCH_CHECK(new_k.dim() == 4 && cache_k.dim() == 4,
                "kv_write: K/V tensors must be 4-D [B, H_kv, S, D]");
    TORCH_CHECK(new_k.size(0) == cache_k.size(0) &&
                new_k.size(1) == cache_k.size(1) &&
                new_k.size(3) == cache_k.size(3),
                "kv_write: new and cache K must share batch / kv_heads / head_dim");
    TORCH_CHECK(write_pos >= 0,
                "kv_write: write_pos must be non-negative");
    TORCH_CHECK(write_pos + new_k.size(2) <= cache_k.size(2),
                "kv_write: write_pos + S exceeds cache max_seq_len");

    // --- Launch the scatter kernel. -----------------------------------------

    const int B       = static_cast<int>(new_k.size(0)); // batch size
    const int H_kv    = static_cast<int>(new_k.size(1)); // number of key value heads=4
    const int S       = static_cast<int>(new_k.size(2)); // sequence length
    const int D       = static_cast<int>(new_k.size(3)); // head dimension
    const int max_seq = static_cast<int>(cache_k.size(2)); // maximum sequence length

    // Assert head dimension D must be a multiple of 8 for vectorized loads and stores
    TORCH_CHECK(D % 8 == 0,
                "kv_write: head_dim must be a multiple of 8 for int4 vectorization (got ", D, ")");

    const int D_v4           = D / 8;                               // int4s per k/v vector, head dimension (128) divided into int4 chunks of 16 bytes
    const int slab_vecs_in   = S * D_v4;                            // int4s per src slab, number of tokens being written in * int4 per token
    const int slab_vecs_out  = max_seq * D_v4;                      // int4s per dst slab, maximum sequence length * int4 per token
    const int write_off_vecs = static_cast<int>(write_pos) * D_v4;  // offset where we're writing

    // Reinterpret the fp16 storage as int4 (16 B = 8 fp16) for vectorized loads and stores
    const int4* new_k_ptr = reinterpret_cast<const int4*>(new_k.data_ptr<at::Half>());
    const int4* new_v_ptr = reinterpret_cast<const int4*>(new_v.data_ptr<at::Half>());
    int4*       cache_k_ptr = reinterpret_cast<int4*>(cache_k.data_ptr<at::Half>());
    int4*       cache_v_ptr = reinterpret_cast<int4*>(cache_v.data_ptr<at::Half>());

    // One block per head; 256 threads perform the loads 
    const int threads = 256;
    const int blocks  = B * H_kv; // one thread block per head per batch 

    kv_write_kernel<<<blocks, threads>>>(
        new_k_ptr, new_v_ptr, cache_k_ptr, cache_v_ptr,
        slab_vecs_in, slab_vecs_out, write_off_vecs);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================
// flash_attention_kernel — fused causal SDPA, GQA-aware, FP16 in/out, FP32
// accumulate. FlashAttention-1 style: block over Q rows and KV rows, with an
// online softmax that never materializes the [seq_len x cur_len] attention
// matrix. Tuned for Turing (sm_75; Quadro RTX 6000).
//
// One grid block per (batch, head, q_block).  Inside the block, 4 warps each
// own ROWS_PER_WARP = WMMA_M = 16 rows of the Q tile and produce the matching
// 16 rows of the O output.
//
// Geometry (template params, currently instantiated at Q=64, KV=32, D=128,
// 128 threads = 4 warps, WMMA 16x16x16):
//
//   q_tile     [Q_BLOCK   x D]    fp16   16 KB
//   v_tile     [KV_BLOCK  x D]    fp16    8 KB
//   union { k_tile [KV_BLOCK x D] fp16    8 KB   // live during QK^T
//           p_tile [Q_BLOCK x KV_BLOCK] fp16  4 KB } // live during PV
//                                          ───────
//                                          32 KB / block
//
// Static shmem is 32 KB exactly, which on Turing's 64 KB/SM lets the
// scheduler resident 2 blocks per SM (was 1 in the previous version at
// 45.8 KB, which was the dominant occupancy limiter).
//
// Per-warp register state held ACROSS the KV loop:
//   O_frag[D/WMMA_N = 8]    fp32 acc fragments (64 fp32 = 64 regs per lane)
//   m_a, m_b, l_a, l_b      online softmax state for the 2 rows each lane owns
// Per-warp transient state per KV iteration (released before P write):
//   S_frag[KV_BLOCK/WMMA_N = 2]   fp32 acc fragments
//
// FP32 accumulator fragment layout on sm_75 (m16n16k16) — used heavily below:
//   group        = lane / 4   ∈ 0..7    ← within the 16-row tile, two rows
//   group_thread = lane % 4   ∈ 0..3      (group and group + 8) per lane
//   frag.x[0] -> (row = group,     col = 2*gt    )
//   frag.x[1] -> (row = group,     col = 2*gt + 1)
//   frag.x[2] -> (row = group + 8, col = 2*gt    )
//   frag.x[3] -> (row = group + 8, col = 2*gt + 1)
//   frag.x[4] -> (row = group,     col = 2*gt + 8)
//   frag.x[5] -> (row = group,     col = 2*gt + 9)
//   frag.x[6] -> (row = group + 8, col = 2*gt + 8)
//   frag.x[7] -> (row = group + 8, col = 2*gt + 9)
// So the 4 lanes {4g..4g+3} together hold all 16 cols of rows g and g+8.
// For KV_BLOCK = 32 (2 fragments) the same 4 lanes collectively hold the
// full row, so a row-wide reduce is a __shfl_xor across 2 levels (mask 1,
// mask 2). This was the layout the previous version verified at runtime by
// passing torch.allclose; we rely on the same correspondence here.
//
// Per-iteration loop body:
//   1. Vector-load K and V tiles via int4 (8 fp16 / load).
//   2. __syncthreads — k/v_tile visible to all warps.
//   3. QK^T into S_frag (WMMA), stays in registers.
//   4. __syncthreads — required before we overwrite k_tile with p_tile, since
//      other warps may still be reading k_tile in their last K_frag load.
//   5. Per-lane scale + causal/range mask, then warp-shuffle row reductions
//      (max, sum) across the 4 lanes that share a row. Online softmax update
//      (m, l, alpha) in registers.
//   6. Rescale O_frag elementwise by per-row alpha.
//   7. Store P (fp16) into p_tile (= same shmem as the dead k_tile).
//   8. __syncwarp — own warp's p_tile slice ready for PV load.
//   9. PV via WMMA: O_frag += P @ V.
//  10. __syncthreads — before next iter overwrites k_tile / v_tile.
//
// Bottlenecks addressed vs the previous version (measured at B=1, S=2048):
//   * Eliminated S_smem (8 KB fp32) + m_smem + l_smem + alpha_smem (0.75 KB).
//     Those values live in registers now. Frees 8.75 KB of shmem, which
//     combined with the k/p union below moves us from 1 → 2 blocks/SM.
//   * Aliased k_tile and p_tile through a shmem union (-4 KB).
//   * Vectorized Q/K/V global loads from scalar fp16 (2 B/load) to int4
//     (16 B/load). 8× fewer LSU ops → directly attacks the 19.6 %
//     long_scoreboard and 4.4 % lg_throttle stall reasons.
//   * Softmax row reductions use ALL 32 lanes per warp via __shfl_xor_sync
//     instead of 16 lanes doing a serial 32-wide scan. Removes the 30.9 %
//     mio_throttle dependency chain and doubles effective lane utilization.
//   * O is written directly from O_frag registers to global memory as half2
//     stores. No fp32 → shmem → scalar fp16 → global stage, no per-column
//     __syncthreads. Was the largest single chunk of acknowledged-inefficient
//     code in the previous version.
// ============================================================================
template<int Q_BLOCK, int KV_BLOCK, int D, int NUM_THREADS,
         int WMMA_M, int WMMA_K, int WMMA_N>
__launch_bounds__(NUM_THREADS, 2)
__global__ void flash_attention_kernel(
    const __half* __restrict__ q,       // [B, h_q,  seq_len, D]
    const __half* __restrict__ k,       // [B, h_kv, max_seq, D]
    const __half* __restrict__ v,       // [B, h_kv, max_seq, D]
    __half*       __restrict__ o,       // [B, h_q,  seq_len, D]
    float softmax_scale,
    int B, int h_q, int h_kv,
    int seq_len, int max_seq, int cur_len)
{
    constexpr int NUM_WARPS     = NUM_THREADS / 32;       // 4
    constexpr int ROWS_PER_WARP = WMMA_M;                 // 16
    constexpr int N_BLOCKS      = KV_BLOCK / WMMA_N;      // 2 (S col-fragments / warp)
    constexpr int K_BLOCKS      = D / WMMA_K;             // 8 (QK^T sum partitions)
    constexpr int O_BLOCKS      = D / WMMA_N;             // 8 (O col-fragments / warp)
    constexpr int P_BLOCKS      = KV_BLOCK / WMMA_K;      // 2 (PV K-partitions)
    constexpr int VECS_PER_ROW  = D / 8;                  // 16 (int4 per D-row of fp16)
    static_assert(Q_BLOCK == NUM_WARPS * ROWS_PER_WARP,
                  "Q_BLOCK must equal NUM_WARPS * WMMA_M");

    // --- Shared memory (32 KB total — fits 2 blocks/SM on Turing) -----------
    __shared__ __half q_tile[Q_BLOCK][D];                 // 16 KB
    __shared__ __half v_tile[KV_BLOCK][D];                //  8 KB
    
    // k_tile (only live during QK^T) and p_tile (written after softmax,
    // read by PV) alias in shmem. One __syncthreads between them is enough.
    union KPUnion {
        __half k_tile[KV_BLOCK][D];                       //  8 KB
        __half p_tile[Q_BLOCK][KV_BLOCK];                 //  4 KB
    };
    __shared__ KPUnion kp;

    // --- Block / warp / lane indexing --------------------------------------
    const int q_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_head = head / (h_q / h_kv);              // GQA: 28/4 = 7, head/7
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;                          // 0..3
    const int lane    = tid & 31;                          // 0..31
    const int warp_row_off = warp_id * ROWS_PER_WARP;      // 0, 16, 32, 48
    const int group   = lane >> 2;                         // 0..7   (row index pair)
    const int gt      = lane & 3;                          // 0..3   (col cluster)

    // The 8 KV columns this lane owns, IDENTICAL for both row A (=group) and
    // row B (=group+8) of its warp slice — see fragment-layout block at the
    // top. The 4 lanes {4g..4g+3} cover all 32 cols of these two rows.
    const int my_cols[8] = {
        2*gt + 0,  2*gt + 1,  2*gt + 8,  2*gt + 9,
        2*gt + 16, 2*gt + 17, 2*gt + 24, 2*gt + 25,
    };

    // --- O accumulator: fp32, REGISTER-RESIDENT across the entire KV loop.
    //     8 fragments × 8 fp32/lane = 64 fp32 = 64 regs per lane just for O.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frag[O_BLOCKS];
    #pragma unroll
    for (int j = 0; j < O_BLOCKS; ++j) wmma::fill_fragment(O_frag[j], 0.0f);

    // --- Per-lane online softmax state (replaces m_smem/l_smem entirely).
    //     Each lane owns the (m, l) for ITS two rows (row A = warp_row_off +
    //     group, row B = ... + group + 8). The 4 lanes in the same 4-lane
    //     group keep redundant copies — they update in lock-step via shuffle
    //     reductions, so they always agree.
    float m_a = -INFINITY, l_a = 0.0f;
    float m_b = -INFINITY, l_b = 0.0f;

    // --- Load Q tile via int4 (16 B / thread / iter) ------------------------
    // Q_BLOCK*D = 8192 fp16 = 1024 int4. 128 threads × 8 iters = 1024.
    // PyTorch's caching allocator gives ≥512 B base alignment so the int4*
    // cast is safe; D = 128 fp16 = 256 B per row is also int4-aligned.
    const int q_row_start = q_block * Q_BLOCK;
    const int q_slab_base = ((batch * h_q + head) * seq_len) * D;
    {
        const int4  zero_v = {0, 0, 0, 0};
        const int4* q_vec_base = reinterpret_cast<const int4*>(q + q_slab_base);
        int4*       q_tile_vec = reinterpret_cast<int4*>(&q_tile[0][0]);
        const int   total_vecs = Q_BLOCK * VECS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < total_vecs; i += NUM_THREADS) {
            const int row     = i / VECS_PER_ROW;
            const int col_vec = i % VECS_PER_ROW;
            const int q_row_global = q_row_start + row;
            // Mask trailing rows when seq_len is not a multiple of Q_BLOCK.
            q_tile_vec[i] = (q_row_global < seq_len)
                ? q_vec_base[q_row_global * VECS_PER_ROW + col_vec]
                : zero_v;
        }
    }
    __syncthreads();

    // --- KV iteration bounds -----------------------------------------------
    // Pure prefill: cur_len == seq_len → q_pos_offset = 0.
    // Speculative decode rollback: cur_len > seq_len → we're attending to
    // history starting at q_pos_offset.
    const int q_pos_offset  = cur_len - seq_len;
    const int q_pos_first   = q_pos_offset + q_row_start;
    const int kv_limit      = min(q_pos_first + Q_BLOCK, cur_len);
    const int num_kv_blocks = (kv_limit + KV_BLOCK - 1) / KV_BLOCK;
    const int kv_slab_base  = ((batch * h_kv) + kv_head) * max_seq * D;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_row_start_global = kv_block * KV_BLOCK;

        // === 1. Load K and V tiles via int4 ================================
        // KV_BLOCK*D = 4096 fp16 = 512 int4 per tile. 128 threads × 4 iter
        // = 512. The two loads in the same loop body let the compiler
        // pipeline K and V issues for memory-level parallelism per thread.
        {
            const int4  zero_v     = {0, 0, 0, 0};
            const int4* k_vec_base = reinterpret_cast<const int4*>(k + kv_slab_base);
            const int4* v_vec_base = reinterpret_cast<const int4*>(v + kv_slab_base);
            int4*       k_tile_vec = reinterpret_cast<int4*>(&kp.k_tile[0][0]);
            int4*       v_tile_vec = reinterpret_cast<int4*>(&v_tile[0][0]);
            const int   total_vecs = KV_BLOCK * VECS_PER_ROW;
            #pragma unroll
            for (int i = tid; i < total_vecs; i += NUM_THREADS) {
                const int row     = i / VECS_PER_ROW;
                const int col_vec = i % VECS_PER_ROW;
                const int kv_row_global = kv_row_start_global + row;
                const bool valid = (kv_row_global < cur_len);
                k_tile_vec[i] = valid ? k_vec_base[kv_row_global * VECS_PER_ROW + col_vec]
                                      : zero_v;
                v_tile_vec[i] = valid ? v_vec_base[kv_row_global * VECS_PER_ROW + col_vec]
                                      : zero_v;
            }
        }
        __syncthreads();  // (2) k/v_tile published to all warps.

        // === 2. QK^T into S_frag (registers; never spilled to shmem) =======
        // Then pack into per-row arrays s_a / s_b for the softmax pass. The
        // S_frag local goes out of scope so its 16 regs are freed before we
        // allocate p_a / p_b.
        float s_a[8], s_b[8];
        {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag[N_BLOCKS];
            #pragma unroll
            for (int j = 0; j < N_BLOCKS; ++j) wmma::fill_fragment(S_frag[j], 0.0f);

            #pragma unroll
            for (int d_block = 0; d_block < K_BLOCKS; ++d_block) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> Q_frag;
                wmma::load_matrix_sync(Q_frag,
                                       &q_tile[warp_row_off][d_block * WMMA_K], D);
                #pragma unroll
                for (int j = 0; j < N_BLOCKS; ++j) {
                    // K stored row-major [KV_BLOCK, D]; reinterpreting the
                    // same memory with col_major layout gives K^T = [D, KV_BLOCK]
                    // which is the B operand of S = Q @ K^T. No memory motion.
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   __half, wmma::col_major> K_frag;
                    wmma::load_matrix_sync(K_frag,
                                           &kp.k_tile[j * WMMA_N][d_block * WMMA_K], D);
                    wmma::mma_sync(S_frag[j], Q_frag, K_frag, S_frag[j]);
                }
            }

            // Pack into per-row arrays + apply softmax_scale here (saves a
            // pass over s_a/s_b later). Indices follow the fp32 acc layout
            // documented at the top of the file. Each lane holds 4 cols per
            // row per fragment; with N_BLOCKS = 2 fragments that's 8 cols.
            #pragma unroll
            for (int j = 0; j < N_BLOCKS; ++j) {
                s_a[4*j + 0] = S_frag[j].x[0] * softmax_scale;
                s_a[4*j + 1] = S_frag[j].x[1] * softmax_scale;
                s_a[4*j + 2] = S_frag[j].x[4] * softmax_scale;
                s_a[4*j + 3] = S_frag[j].x[5] * softmax_scale;
                s_b[4*j + 0] = S_frag[j].x[2] * softmax_scale;
                s_b[4*j + 1] = S_frag[j].x[3] * softmax_scale;
                s_b[4*j + 2] = S_frag[j].x[6] * softmax_scale;
                s_b[4*j + 3] = S_frag[j].x[7] * softmax_scale;
            }
        }
        // S_frag dead here.

        // === 3. Sync before reusing k_tile shmem as p_tile =================
        // All warps must be past their last wmma::load_matrix_sync(K_frag)
        // before ANY warp writes p_tile (which aliases k_tile).
        __syncthreads();

        // === 4. Causal + range mask ========================================
        // q_pos = the absolute position of the query row. For causal SDPA we
        // mask any k_pos > q_pos. We also mask k_pos beyond cur_len (the
        // tail of the last KV block when cur_len isn't a multiple of KV_BLOCK).
        const int q_pos_a = q_pos_first + warp_row_off + group;
        const int q_pos_b = q_pos_first + warp_row_off + group + 8;
        const int kv_base = kv_block * KV_BLOCK;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int k_pos = kv_base + my_cols[c];
            if (k_pos > q_pos_a || k_pos >= cur_len) s_a[c] = -INFINITY;
            if (k_pos > q_pos_b || k_pos >= cur_len) s_b[c] = -INFINITY;
        }

        // === 5. Row-max via local-then-shuffle reduction ===================
        // Each lane has 8 cols of its row. After the local fmaxf-fold, two
        // __shfl_xor (mask 1 and 2) propagate the max across the 4-lane group
        // sharing the row. After both shuffles all 4 lanes hold the row max.
        // Cost: O(log4) shuffles. Was O(32) serial elements in the previous
        // version (with 16 lanes idle, no less).
        float row_max_a = -INFINITY, row_max_b = -INFINITY;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            row_max_a = fmaxf(row_max_a, s_a[c]);
            row_max_b = fmaxf(row_max_b, s_b[c]);
        }
        row_max_a = fmaxf(row_max_a, __shfl_xor_sync(0xFFFFFFFF, row_max_a, 1));
        row_max_a = fmaxf(row_max_a, __shfl_xor_sync(0xFFFFFFFF, row_max_a, 2));
        row_max_b = fmaxf(row_max_b, __shfl_xor_sync(0xFFFFFFFF, row_max_b, 1));
        row_max_b = fmaxf(row_max_b, __shfl_xor_sync(0xFFFFFFFF, row_max_b, 2));

        // === 6. Online softmax update + alpha rescale ======================
        // alpha = exp(m_old - m_new) is the shrink factor applied to the
        // *previously accumulated* O contributions to renormalize them under
        // the new row-max. Two edge cases collapse to alpha = 0:
        //   (a) m_old == -INF (this is the first time the row sees a finite
        //       value; previous O contribution is zero anyway).
        //   (b) m_new == -INF (the entire row is masked through this iter;
        //       guards against expf(-inf - -inf) = NaN).
        const float m_new_a = fmaxf(m_a, row_max_a);
        const float m_new_b = fmaxf(m_b, row_max_b);
        const float alpha_a = (m_a == -INFINITY || m_new_a == -INFINITY)
                                  ? 0.0f : __expf(m_a - m_new_a);
        const float alpha_b = (m_b == -INFINITY || m_new_b == -INFINITY)
                                  ? 0.0f : __expf(m_b - m_new_b);

        // p = exp(s - m_new); per-row local sum then shuffle reduction.
        float p_a[8], p_b[8];
        float row_sum_a = 0.0f, row_sum_b = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            p_a[c] = (m_new_a == -INFINITY) ? 0.0f : __expf(s_a[c] - m_new_a);
            p_b[c] = (m_new_b == -INFINITY) ? 0.0f : __expf(s_b[c] - m_new_b);
            row_sum_a += p_a[c];
            row_sum_b += p_b[c];
        }
        row_sum_a += __shfl_xor_sync(0xFFFFFFFF, row_sum_a, 1);
        row_sum_a += __shfl_xor_sync(0xFFFFFFFF, row_sum_a, 2);
        row_sum_b += __shfl_xor_sync(0xFFFFFFFF, row_sum_b, 1);
        row_sum_b += __shfl_xor_sync(0xFFFFFFFF, row_sum_b, 2);

        l_a = alpha_a * l_a + row_sum_a;
        l_b = alpha_b * l_b + row_sum_b;
        m_a = m_new_a;
        m_b = m_new_b;

        // === 7. Rescale O_frag by per-row alpha ============================
        // Same row pattern as the QK^T extraction above:
        //   O_frag[j].x[0,1,4,5] -> row A (= row `group`)
        //   O_frag[j].x[2,3,6,7] -> row B (= row `group + 8`)
        #pragma unroll
        for (int j = 0; j < O_BLOCKS; ++j) {
            O_frag[j].x[0] *= alpha_a; O_frag[j].x[1] *= alpha_a;
            O_frag[j].x[4] *= alpha_a; O_frag[j].x[5] *= alpha_a;
            O_frag[j].x[2] *= alpha_b; O_frag[j].x[3] *= alpha_b;
            O_frag[j].x[6] *= alpha_b; O_frag[j].x[7] *= alpha_b;
        }

        // === 8. Write P (fp16) into p_tile =================================
        // We can't directly construct an fp16 matrix_a WMMA fragment from an
        // fp32 acc fragment on Turing — the two fragments have different
        // per-lane layouts in the C++ WMMA API. So we round-trip through
        // 4 KB of shmem (the aliased k_tile slot). The 8 element pairs per
        // row are 2-element-adjacent in column space, so we pack each pair
        // into one __half2 store → 4 half2 stores per row per lane = 8 per
        // lane total (vs the 16 scalar fp16 stores otherwise).
        __half2* p_row_a = reinterpret_cast<__half2*>(&kp.p_tile[warp_row_off + group    ][0]);
        __half2* p_row_b = reinterpret_cast<__half2*>(&kp.p_tile[warp_row_off + group + 8][0]);
        // half2 index within a row:
        //   gt      -> cols (2*gt    , 2*gt + 1)     ← from p_a[0], p_a[1]
        //   gt + 4  -> cols (2*gt + 8, 2*gt + 9)     ← from p_a[2], p_a[3]
        //   gt + 8  -> cols (2*gt+16, 2*gt + 17)     ← from p_a[4], p_a[5]
        //   gt + 12 -> cols (2*gt+24, 2*gt + 25)     ← from p_a[6], p_a[7]
        p_row_a[gt +  0] = __floats2half2_rn(p_a[0], p_a[1]);
        p_row_a[gt +  4] = __floats2half2_rn(p_a[2], p_a[3]);
        p_row_a[gt +  8] = __floats2half2_rn(p_a[4], p_a[5]);
        p_row_a[gt + 12] = __floats2half2_rn(p_a[6], p_a[7]);
        p_row_b[gt +  0] = __floats2half2_rn(p_b[0], p_b[1]);
        p_row_b[gt +  4] = __floats2half2_rn(p_b[2], p_b[3]);
        p_row_b[gt +  8] = __floats2half2_rn(p_b[4], p_b[5]);
        p_row_b[gt + 12] = __floats2half2_rn(p_b[6], p_b[7]);
        __syncwarp();  // p_tile is per-warp-disjoint, so __syncwarp suffices.

        // === 9. PV via WMMA: O_frag += P @ V ===============================
        // P is 16x32 per warp; V is 32x128. Sum over P_BLOCKS = 2 K-blocks
        // per O-col fragment. All 4 warps read the same V tile (different
        // P slices, since each warp owns disjoint Q rows).
        #pragma unroll
        for (int p_block = 0; p_block < P_BLOCKS; ++p_block) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> P_frag;
            wmma::load_matrix_sync(P_frag,
                                   &kp.p_tile[warp_row_off][p_block * WMMA_K],
                                   KV_BLOCK);
            #pragma unroll
            for (int d_col = 0; d_col < O_BLOCKS; ++d_col) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> V_frag;
                wmma::load_matrix_sync(V_frag,
                                       &v_tile[p_block * WMMA_K][d_col * WMMA_N],
                                       D);
                wmma::mma_sync(O_frag[d_col], P_frag, V_frag, O_frag[d_col]);
            }
        }

        __syncthreads();  // (10) before next iter overwrites k_tile / v_tile.
    }

    // === 11. Final normalize: O_frag /= l per row ==========================
    // (l == 0 means the entire row was masked; the output is undefined per
    // the standard SDPA definition. We return 0 rather than NaN.)
    {
        const float inv_l_a = (l_a == 0.0f) ? 0.0f : 1.0f / l_a;
        const float inv_l_b = (l_b == 0.0f) ? 0.0f : 1.0f / l_b;
        #pragma unroll
        for (int j = 0; j < O_BLOCKS; ++j) {
            O_frag[j].x[0] *= inv_l_a; O_frag[j].x[1] *= inv_l_a;
            O_frag[j].x[4] *= inv_l_a; O_frag[j].x[5] *= inv_l_a;
            O_frag[j].x[2] *= inv_l_b; O_frag[j].x[3] *= inv_l_b;
            O_frag[j].x[6] *= inv_l_b; O_frag[j].x[7] *= inv_l_b;
        }
    }

    // === 12. Write O to global memory ======================================
    // Direct register → global, no shmem staging.
    //
    // For each fragment j (covering D-cols [j*16 .. j*16 + 16)):
    //   Row A (warp_row_off + group):
    //     cols (2*gt + 0, 2*gt + 1) ← x[0], x[1]
    //     cols (2*gt + 8, 2*gt + 9) ← x[4], x[5]
    //   Row B (warp_row_off + group + 8):
    //     cols (2*gt + 0, 2*gt + 1) ← x[2], x[3]
    //     cols (2*gt + 8, 2*gt + 9) ← x[6], x[7]
    //
    // Each (lane, row, fragment) is 2 half2 stores = 4 fp16 written. Total
    // per lane: 2 rows × 8 fragments × 2 half2 = 32 half2 stores = 128 B.
    // Lanes in the same 4-lane group write to adjacent cols, so the 4
    // stores per "wave" coalesce into a single L2 transaction per row.
    //
    // No __syncthreads here: O rows are disjoint across warps and writes go
    // straight to global. (The previous version's 8 __syncthreads inside the
    // O-write loop were a major contributor to its `barrier` and `wait`
    // stall categories.)
    {
        const int row_a        = warp_row_off + group;
        const int row_b        = warp_row_off + group + 8;
        const int row_a_global = q_row_start + row_a;
        const int row_b_global = q_row_start + row_b;
        const int o_slab_base  = ((batch * h_q + head) * seq_len) * D;
        const bool row_a_in    = row_a_global < seq_len;
        const bool row_b_in    = row_b_global < seq_len;

        #pragma unroll
        for (int j = 0; j < O_BLOCKS; ++j) {
            const int col_off = j * WMMA_N;
            if (row_a_in) {
                __half2* dst = reinterpret_cast<__half2*>(
                    o + o_slab_base + row_a_global * D + col_off);
                // half2 index within the 16-col fragment:
                //   gt     → cols (2*gt + 0, 2*gt + 1)
                //   gt + 4 → cols (2*gt + 8, 2*gt + 9)
                dst[gt    ] = __floats2half2_rn(O_frag[j].x[0], O_frag[j].x[1]);
                dst[gt + 4] = __floats2half2_rn(O_frag[j].x[4], O_frag[j].x[5]);
            }
            if (row_b_in) {
                __half2* dst = reinterpret_cast<__half2*>(
                    o + o_slab_base + row_b_global * D + col_off);
                dst[gt    ] = __floats2half2_rn(O_frag[j].x[2], O_frag[j].x[3]);
                dst[gt + 4] = __floats2half2_rn(O_frag[j].x[6], O_frag[j].x[7]);
            }
        }
    }
}

torch::Tensor fused_attn_forward(torch::Tensor q,
                                 torch::Tensor cache_k,
                                 torch::Tensor cache_v,
                                 int64_t cur_len,
                                 double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && cache_k.is_cuda() && cache_v.is_cuda(),
                "fused_attn: all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && cache_k.is_contiguous() && cache_v.is_contiguous(),
                "fused_attn: all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kHalf &&
                cache_k.scalar_type() == torch::kHalf &&
                cache_v.scalar_type() == torch::kHalf,
                "fused_attn: all tensors must be float16");
    TORCH_CHECK(q.dim() == 4 && cache_k.dim() == 4 && cache_v.dim() == 4,
                "fused_attn: q / cache_k / cache_v must be 4-D");
    TORCH_CHECK(q.size(0) == cache_k.size(0) && q.size(0) == cache_v.size(0),
                "fused_attn: batch sizes must match");
    TORCH_CHECK(q.size(3) == cache_k.size(3) && q.size(3) == cache_v.size(3),
                "fused_attn: head_dim must match across q / cache_k / cache_v");
    TORCH_CHECK(q.size(1) % cache_k.size(1) == 0,
                "fused_attn: num_heads must be divisible by num_kv_heads (GQA)");
    TORCH_CHECK(cur_len > 0 && cur_len <= cache_k.size(2),
                "fused_attn: cur_len must be in (0, max_seq_len]");

    const int B       = q.size(0);
    const int h_q     = q.size(1);
    const int h_kv    = cache_k.size(1);
    const int seq_len = q.size(2);
    const int max_seq = cache_k.size(2);

    auto o = torch::empty_like(q);

    // Template params — see the geometry block at the top of the kernel.
    constexpr int Q_block_size  = 64;
    constexpr int KV_block_size = 32;
    constexpr int D             = 128;
    constexpr int NUM_THREADS   = 128;        // = 4 warps
    constexpr int WMMA_M        = 16;
    constexpr int WMMA_N        = 16;
    constexpr int WMMA_K        = 16;

    dim3 grid((seq_len + Q_block_size - 1) / Q_block_size, h_q, B);
    dim3 block(NUM_THREADS);

    flash_attention_kernel<Q_block_size, KV_block_size, D, NUM_THREADS, WMMA_M, WMMA_K, WMMA_N>
        <<<grid, block>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(cache_k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(cache_v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
            static_cast<float>(softmax_scale),
            B, h_q, h_kv,
            seq_len, max_seq, static_cast<int>(cur_len)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}



// 5. Output projection: y = x @ W_o^T  (no bias; matches Qwen2 o_proj).
//    x:[M, H_q], W_o:[H, H_q] -> y:[M, H].
torch::Tensor o_proj_forward(torch::Tensor x, torch::Tensor W_o) {
    TORCH_CHECK(x.is_cuda() && W_o.is_cuda(),
                "o_proj: x and W_o must be CUDA");
    TORCH_CHECK(x.is_contiguous() && W_o.is_contiguous(),
                "o_proj: x and W_o must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                W_o.scalar_type() == torch::kHalf,
                "o_proj: x and W_o must be float16");
    TORCH_CHECK(x.dim() == 2,    "o_proj: x must be 2-D [M, H_q]");
    TORCH_CHECK(W_o.dim() == 2,  "o_proj: W_o must be 2-D [H, H_q]");
    TORCH_CHECK(W_o.size(1) == x.size(1),
                "o_proj: W_o inner dim must equal x hidden dim");

    const int64_t M = x.size(0);
    const int64_t H = W_o.size(0);
    // TODO: cuBLAS GEMM (cublasGemmEx, CUDA_R_16F), no bias.
    return torch::empty({M, H}, x.options());
}
