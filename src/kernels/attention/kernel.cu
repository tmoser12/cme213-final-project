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


template<int Q_block_size, int KV_block_size, int D, int NUM_THREADS, int WMMA_M, int WMMA_K, int WMMA_N>
__global__ void flash_attention_kernel(
    const __half* __restrict__ q,       // [B, h_q,  seq_len, D]
    const __half* __restrict__ k,       // [B, h_kv, max_seq, D]
    const __half* __restrict__ v,       // [B, h_kv, max_seq, D]
    __half*       __restrict__ o,       // [B, h_q,  seq_len, D]
    float softmax_scale,
    int B, int h_q, int h_kv,
    int seq_len, int max_seq, int cur_len)
{
    // --- Shared memory layout (~45 KB total) ---------------------------------
    __shared__ __half  q_tile[Q_block_size][D];                         // 16 KB
    __shared__ __half  k_tile[KV_block_size][D];                        //  8 KB
    __shared__ __half  v_tile[KV_block_size][D];                        //  8 KB
    __shared__ float   S_smem[Q_block_size][KV_block_size];             //  8 KB
    __shared__ __half  P_smem[Q_block_size][KV_block_size];             //  4 KB
    __shared__ float   m_smem[Q_block_size];
    __shared__ float   l_smem[Q_block_size];
    __shared__ float   alpha_smem[Q_block_size];

    // --- Thread / warp / batch indexing --------------------------------------
    const int q_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_head = head / (h_q / h_kv);          // GQA: 28/4=7, head/7

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;                     // 0..3
    const int lane_id = tid % 32;                     // 0..31
    const int warp_row_off = warp_id * WMMA_M;        // 0, 16, 32, 48

    // --- Initialize O fragments (live in registers across KV iterations) -----
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        O_frag[D / WMMA_N];
    #pragma unroll
    for (int i = 0; i < D / WMMA_N; ++i) {
        wmma::fill_fragment(O_frag[i], 0.0f);
    }

    // --- Initialize softmax state --------------------------------------------
    if (tid < Q_block_size) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    // --- Load Q tile ---------------------------------------------------------
    const int q_start = q_block * Q_block_size;
    const int q_base  = (((batch * h_q) + head) * seq_len + q_start) * D;

    #pragma unroll
    for (int i = tid; i < Q_block_size * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int q_row_global = q_start + row;
        q_tile[row][col] = (q_row_global < seq_len)
            ? q[q_base + row * D + col]
            : __float2half(0.0f);
    }
    __syncthreads();

    // --- KV iteration bounds -------------------------------------------------
    // q_pos_offset = 0 for pure prefill (cur_len == seq_len).
    const int q_pos_offset  = cur_len - seq_len;
    const int q_pos_first   = q_pos_offset + q_start;
    const int kv_limit      = min(q_pos_first + Q_block_size, cur_len);
    const int num_kv_blocks = (kv_limit + KV_block_size - 1) / KV_block_size;
    const int kv_slab_base  = ((batch * h_kv) + kv_head) * max_seq * D;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {

        // --- Load K and V tiles ----------------------------------------------
        const int kv_offset = kv_slab_base + kv_block * KV_block_size * D;

        #pragma unroll
        for (int i = tid; i < KV_block_size * D; i += NUM_THREADS) {
            int row = i / D;
            int col = i % D;
            int kv_row_global = kv_block * KV_block_size + row;
            bool valid = (kv_row_global < cur_len);
            k_tile[row][col] = valid ? k[kv_offset + row * D + col]
                                     : __float2half(0.0f);
            v_tile[row][col] = valid ? v[kv_offset + row * D + col]
                                     : __float2half(0.0f);
        }
        __syncthreads();

        // --- QK^T via WMMA ---------------------------------------------------
        // Each warp computes its 16x32 S slice = 2 col-fragments,
        // summing over 8 D-chunks of width WMMA_K=16.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
            S_frag[KV_block_size / WMMA_N];

        #pragma unroll
        for (int j = 0; j < KV_block_size / WMMA_N; ++j) {
            wmma::fill_fragment(S_frag[j], 0.0f);
        }

        #pragma unroll
        for (int d_block = 0; d_block < D / WMMA_K; ++d_block) {

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> Q_frag;
            wmma::load_matrix_sync(Q_frag,
                                   &q_tile[warp_row_off][d_block * WMMA_K], D);

            #pragma unroll
            for (int j = 0; j < KV_block_size / WMMA_N; ++j) {
                // K is row-major [B_c, D]. Loading the same memory with
                // col_major layout reinterprets it as K^T = [D, B_c],
                // which is exactly the B operand for S = Q @ K^T.
                // No memory movement.
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::col_major> K_frag;
                wmma::load_matrix_sync(
                    K_frag, &k_tile[j * WMMA_N][d_block * WMMA_K], D);

                wmma::mma_sync(S_frag[j], Q_frag, K_frag, S_frag[j]);
            }
        }

        // --- Stage S to shared memory (per warp owns rows warp_row_off..+16) -
        #pragma unroll
        for (int j = 0; j < KV_block_size / WMMA_N; ++j) {
            wmma::store_matrix_sync(&S_smem[warp_row_off][j * WMMA_N],
                                    S_frag[j], KV_block_size,
                                    wmma::mem_row_major);
        }
        __syncwarp();    // own-warp store -> read ordering

        // --- Online softmax: 16 lanes per warp handle 16 rows ----------------
        if (lane_id < WMMA_M) {
            int my_row = warp_row_off + lane_id;
            int q_pos  = q_pos_first + my_row;

            // Scale + causal/range mask + row max
            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < KV_block_size; ++j) {
                int k_pos = kv_block * KV_block_size + j;
                float s = S_smem[my_row][j] * softmax_scale;
                if (k_pos > q_pos || k_pos >= cur_len) s = -INFINITY;
                S_smem[my_row][j] = s;
                row_max = fmaxf(row_max, s);
            }

            float m_old = m_smem[my_row];
            float m_new = fmaxf(m_old, row_max);

            if (m_new == -INFINITY) {
                // Entire row masked - shouldn't happen with our bounds,
                // but safe to guard against -inf - -inf = NaN.
                alpha_smem[my_row] = 0.0f;
                #pragma unroll
                for (int j = 0; j < KV_block_size; ++j) {
                    P_smem[my_row][j] = __float2half(0.0f);
                }
            } else {
                float alpha = (m_old == -INFINITY) ? 0.0f
                                                   : __expf(m_old - m_new);
                alpha_smem[my_row] = alpha;

                float row_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < KV_block_size; ++j) {
                    float p = __expf(S_smem[my_row][j] - m_new);
                    row_sum += p;
                    P_smem[my_row][j] = __float2half(p);
                }

                l_smem[my_row] = alpha * l_smem[my_row] + row_sum;
                m_smem[my_row] = m_new;
            }
        }
        __syncwarp();

        // --- Rescale O fragments by per-row alpha ----------------------------
        // Fragment layout for mma.m16n16k16 fp32 accumulator on Turing/Ampere:
        //   group = lane_id / 4    (0..7)
        //   group_thread = lane_id % 4    (0..3)
        // Each lane holds 8 elements in two rows: `group` and `group + 8`.
        //   frag.x[0,1,4,5] -> row `group`
        //   frag.x[2,3,6,7] -> row `group + 8`
        // *** VERIFY THIS LAYOUT WITH A UNIT TEST BEFORE USING. ***
        {
            const int group = lane_id / 4;
            float alpha_a = alpha_smem[warp_row_off + group];
            float alpha_b = alpha_smem[warp_row_off + group + 8];

            #pragma unroll
            for (int j = 0; j < D / WMMA_N; ++j) {
                O_frag[j].x[0] *= alpha_a;
                O_frag[j].x[1] *= alpha_a;
                O_frag[j].x[2] *= alpha_b;
                O_frag[j].x[3] *= alpha_b;
                O_frag[j].x[4] *= alpha_a;
                O_frag[j].x[5] *= alpha_a;
                O_frag[j].x[6] *= alpha_b;
                O_frag[j].x[7] *= alpha_b;
            }
        }

        // --- PV via WMMA: O_frag += P @ V ------------------------------------
        // P is 16x32 (one warp's slice), V is 32x128, O is 16x128.
        // Sum over 2 P-col-blocks (= V-row-blocks) per O column.
        #pragma unroll
        for (int p_block = 0; p_block < KV_block_size / WMMA_K; ++p_block) {

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> P_frag;
            wmma::load_matrix_sync(
                P_frag, &P_smem[warp_row_off][p_block * WMMA_K], KV_block_size);

            #pragma unroll
            for (int d_col = 0; d_col < D / WMMA_N; ++d_col) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __half, wmma::row_major> V_frag;
                wmma::load_matrix_sync(
                    V_frag, &v_tile[p_block * WMMA_K][d_col * WMMA_N], D);

                wmma::mma_sync(O_frag[d_col], P_frag, V_frag, O_frag[d_col]);
            }
        }

        __syncthreads();   // before next iteration overwrites k_tile/v_tile
    }

    // --- Final normalize: O_frag /= l_smem[its_row] --------------------------
    // Same row-pattern as the alpha rescale.
    {
        const int group = lane_id / 4;
        float inv_l_a = 1.0f / l_smem[warp_row_off + group];
        float inv_l_b = 1.0f / l_smem[warp_row_off + group + 8];
        if (!isfinite(inv_l_a)) inv_l_a = 0.0f;
        if (!isfinite(inv_l_b)) inv_l_b = 0.0f;

        #pragma unroll
        for (int j = 0; j < D / WMMA_N; ++j) {
            O_frag[j].x[0] *= inv_l_a;
            O_frag[j].x[1] *= inv_l_a;
            O_frag[j].x[2] *= inv_l_b;
            O_frag[j].x[3] *= inv_l_b;
            O_frag[j].x[4] *= inv_l_a;
            O_frag[j].x[5] *= inv_l_a;
            O_frag[j].x[6] *= inv_l_b;
            O_frag[j].x[7] *= inv_l_b;
        }
    }

    // --- Write O to global memory --------------------------------------------
    // Staged one D-column-block at a time through S_smem (reused as fp32 buf).
    // wmma::store_matrix_sync only outputs in the fragment dtype (fp32 here);
    // Turing has no built-in fp32->fp16 store path, so we cast manually.
    //
    // *** This path is correct but inefficient. Worth replacing with a
    //     single coalesced pass once correctness is established. ***
    #pragma unroll
    for (int j = 0; j < D / WMMA_N; ++j) {
        float (*stage)[WMMA_N] =
            reinterpret_cast<float(*)[WMMA_N]>(&S_smem[0][0]);
        wmma::store_matrix_sync(&stage[warp_row_off][0], O_frag[j],
                                WMMA_N, wmma::mem_row_major);
        __syncwarp();

        if (lane_id < WMMA_M) {
            int row = warp_row_off + lane_id;
            int q_row_global = q_start + row;
            if (q_row_global < seq_len) {
                int row_off = (((batch * h_q) + head) * seq_len + q_row_global)
                              * D + j * WMMA_N;
                #pragma unroll
                for (int c = 0; c < WMMA_N; ++c) {
                    o[row_off + c] = __float2half(stage[row][c]);
                }
            }
        }
        __syncthreads();   // before reusing S_smem for the next column
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
    (void)softmax_scale;
    

    const int B       = q.size(0);
    const int h_q     = q.size(1);
    const int h_kv    = cache_k.size(1);
    const int seq_len = q.size(2);
    const int max_seq = cache_k.size(2);

    auto o = torch::empty_like(q);

    constexpr int Q_block_size = 64; constexpr int KV_block_size = 32; constexpr int D = 128; 
    constexpr int NUM_THREADS = 128; constexpr int NUM_WARPS = 32;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 grid((seq_len + Q_block_size - 1) / Q_block_size, h_q, B);
    dim3 block(NUM_THREADS);                    // 128, not 256

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
