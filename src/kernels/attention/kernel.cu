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


// Wrap every cuBLAS call. cuBLAS returns a status enum and never throws without this
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

// RoPE kernel
__global__ void rope_kernel(
    __half*       __restrict__ x,     // [B, H, S, D] row-major; mutated in place
    const __half* __restrict__ cos,   // [B, S, D]   (last D/2 entries duplicate the first D/2)
    const __half* __restrict__ sin,   // [B, S, D]
    int H,                            // num_heads (Q) or num_kv_heads (K)
    int S,                            // seq_len
    int D                             // head_dim (must be even)
) {
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

// KV write kernel
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

    constexpr int P_STRIDE      = KV_BLOCK + 8;           // 40
    static_assert(P_STRIDE % 8 == 0, "P_STRIDE must be a multiple of 8 for WMMA ldm");
    static_assert(Q_BLOCK == NUM_WARPS * ROWS_PER_WARP,
                  "Q_BLOCK must equal NUM_WARPS * WMMA_M");

    // --- Shared memory ------------
    __shared__ __half q_tile[Q_BLOCK][D];                 // 16 KB
    __shared__ __half v_tile[KV_BLOCK][D];                //  8 KB

    union KPUnion {
        __half k_tile[KV_BLOCK][D];                       //  8 KB
        __half p_tile[Q_BLOCK][P_STRIDE];                 //  5 KB (padded; see P_STRIDE)
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

    const int my_cols[8] = {
        2*gt + 0,  2*gt + 1,  2*gt + 8,  2*gt + 9,
        2*gt + 16, 2*gt + 17, 2*gt + 24, 2*gt + 25,
    };

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frag[O_BLOCKS];
    #pragma unroll
    for (int j = 0; j < O_BLOCKS; ++j) wmma::fill_fragment(O_frag[j], 0.0f);

    float m_a = -INFINITY, l_a = 0.0f;
    float m_b = -INFINITY, l_b = 0.0f;

    // --- Load Q tile via int4 ------------------------
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
    const int q_pos_offset  = cur_len - seq_len;
    const int q_pos_first   = q_pos_offset + q_row_start;
    const int kv_limit      = min(q_pos_first + Q_BLOCK, cur_len);
    const int num_kv_blocks = (kv_limit + KV_BLOCK - 1) / KV_BLOCK;
    const int kv_slab_base  = ((batch * h_kv) + kv_head) * max_seq * D;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_row_start_global = kv_block * KV_BLOCK;

        // === 1. Load K and V tiles via int4 ================================
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
        __syncthreads();  

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
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   __half, wmma::col_major> K_frag;
                    wmma::load_matrix_sync(K_frag,
                                           &kp.k_tile[j * WMMA_N][d_block * WMMA_K], D);
                    wmma::mma_sync(S_frag[j], Q_frag, K_frag, S_frag[j]);
                }
            }

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

      
        __syncthreads();

        const int q_pos_a = q_pos_first + warp_row_off + group;
        const int q_pos_b = q_pos_first + warp_row_off + group + 8;
        const int kv_base = kv_block * KV_BLOCK;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int k_pos = kv_base + my_cols[c];
            if (k_pos > q_pos_a || k_pos >= cur_len) s_a[c] = -INFINITY;
            if (k_pos > q_pos_b || k_pos >= cur_len) s_b[c] = -INFINITY;
        }

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

        #pragma unroll
        for (int j = 0; j < O_BLOCKS; ++j) {
            O_frag[j].x[0] *= alpha_a; O_frag[j].x[1] *= alpha_a;
            O_frag[j].x[4] *= alpha_a; O_frag[j].x[5] *= alpha_a;
            O_frag[j].x[2] *= alpha_b; O_frag[j].x[3] *= alpha_b;
            O_frag[j].x[6] *= alpha_b; O_frag[j].x[7] *= alpha_b;
        }

        __half2* p_row_a = reinterpret_cast<__half2*>(&kp.p_tile[warp_row_off + group    ][0]);
        __half2* p_row_b = reinterpret_cast<__half2*>(&kp.p_tile[warp_row_off + group + 8][0]);

        p_row_a[gt +  0] = __floats2half2_rn(p_a[0], p_a[1]);
        p_row_a[gt +  4] = __floats2half2_rn(p_a[2], p_a[3]);
        p_row_a[gt +  8] = __floats2half2_rn(p_a[4], p_a[5]);
        p_row_a[gt + 12] = __floats2half2_rn(p_a[6], p_a[7]);
        p_row_b[gt +  0] = __floats2half2_rn(p_b[0], p_b[1]);
        p_row_b[gt +  4] = __floats2half2_rn(p_b[2], p_b[3]);
        p_row_b[gt +  8] = __floats2half2_rn(p_b[4], p_b[5]);
        p_row_b[gt + 12] = __floats2half2_rn(p_b[6], p_b[7]);
        __syncwarp();  

    
        #pragma unroll
        for (int p_block = 0; p_block < P_BLOCKS; ++p_block) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __half, wmma::row_major> P_frag;
            wmma::load_matrix_sync(P_frag,
                                   &kp.p_tile[warp_row_off][p_block * WMMA_K],
                                   P_STRIDE);
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

    const int64_t M = x.size(0);          // rows of X and Y (batch * seq_len)
    const int64_t K = x.size(1);          // H_q (3584 for Qwen2.5-7B)
    const int64_t N = W_o.size(0);        // H   (3584 for Qwen2.5-7B)

    // No bias on o_proj (Qwen2 sets bias=False), so beta=0 and y is left
    // uninitialized — the GEMM overwrites it.
    auto y = at::empty({M, N}, x.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                                    // op(A): transpose W_o (col-major view)
        CUBLAS_OP_N,                                    // op(B): leave X as-is
        static_cast<int>(N),                            // M and N are swapped since cuBLAS is column-major
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        W_o.data_ptr<at::Half>(), CUDA_R_16F,           // A pointer + dtype
        static_cast<int>(K),                            // lda
        x.data_ptr<at::Half>(),   CUDA_R_16F,           // B pointer + dtype
        static_cast<int>(K),                            // ldb
        &beta,
        y.data_ptr<at::Half>(),   CUDA_R_16F,           // C pointer + dtype
        static_cast<int>(N),                            // ldc
        CUBLAS_COMPUTE_32F,                             // fp16 mul, fp32 accumulate
        CUBLAS_GEMM_DEFAULT));                          // let cuBLAS pick the Tensor Core path

    return y;
}
