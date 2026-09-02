// kernel_dev/target/kernels/residual_ops/kernel.cu
// Two host-side ops the decoder loop needs once attention/swiglu/rmsnorm are in
// place but before the final logits:
//
//   * residual_add_forward(a, b)        -> out = a + b   (elementwise fp16)
//       The two "+ residual" writes per decoder layer. swiglu_forward and the
//       attention o_proj return the residual-ready sub-block output; this kernel
//       is the add that folds it back into the residual stream. A fused
//       add+rmsnorm is the eventual optimization, but the sum still has to be
//       materialized (it flows forward as the *unnormalized* residual while the
//       norm consumes a normalized copy), so the standalone add is the oracle
//       and the safe default. Bit-exact vs torch's fp16 add: __hadd2 is the same
//       round-to-nearest fp16 add, so allclose passes at atol=0.
//
//   * lm_head_forward(hidden, weight)   -> logits = hidden @ weight^T
//       The final vocab projection. weight is [vocab=151936, H=896] (Qwen2.5-0.5B
//       TIES embeddings, so the runtime passes embed_tokens.weight here -- this
//       kernel is weight-source-agnostic and just GEMMs whatever it's given).
//       A plain cuBLAS GEMM, identical to swiglu's gate/up/down and attention's
//       qkv/o projections -- cuBLAS owns the 151936x896 matmul, no hand-rolled
//       kernel would beat it. Returns fp16 logits [M, vocab] for whatever rows
//       are passed in; the host pre-slices to the position(s) it needs (last
//       token in decode, the gamma+1 candidates in spec-verify). Argmax/sampling
//       stays out of here -- that belongs to the fused speculative sampler.
//
// The add mirrors rmsnorm/swiglu's float4 / half2 vectorization: cast half* to
// float4* for 128-bit loads/stores, do the work on half2 lanes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>     // at::cuda::getCurrentCUDABlasHandle
#include <c10/cuda/CUDAException.h>    // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

// Wrap every cuBLAS call. cuBLAS returns a status enum and never throws.
#define CUBLAS_CHECK(expr)                                              \
    do {                                                                \
        cublasStatus_t _status = (expr);                                \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                   \
                    "cuBLAS error: ", cublasGetStatusString(_status));  \
    } while (0)


// ----------------------------------------------------------------------------
// residual_add kernel:  out = a + b.  Flat elementwise over the whole buffer.
// 1 thread -> 8 halves (one float4 = 128-bit load/store), grid-stride so any
// element count is covered. In-place safe (out may alias a or b): each lane
// reads index i and writes index i.
// ----------------------------------------------------------------------------
__global__ void residual_add_kernel(
    const __half* __restrict__ a,      // [numel]
    const __half* __restrict__ b,      // [numel]
    __half*       __restrict__ out,    // [numel] (may alias a or b)
    int64_t num_vec)                   // numel / 8
{
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4*       out4 = reinterpret_cast<float4*>(out);

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_vec;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float4 av = a4[i];
        float4 bv = b4[i];
        float4 ov;

        // Treat the 128 bits as four 32-bit half2 lanes; native fp16 add.
        half2* ah = reinterpret_cast<half2*>(&av);
        half2* bh = reinterpret_cast<half2*>(&bv);
        half2* oh = reinterpret_cast<half2*>(&ov);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            oh[j] = __hadd2(ah[j], bh[j]);
        }
        out4[i] = ov;
    }
}

static void launch_residual_add(const at::Half* a, const at::Half* b,
                                at::Half* out, int64_t numel) {
    TORCH_CHECK(numel % 8 == 0,
                "residual_add: element count must be a multiple of 8 for the vectorized kernel");
    const int64_t num_vec = numel / 8;

    const int threads = 256;
    int64_t want_blocks = (num_vec + threads - 1) / threads;
    const int64_t max_blocks = 65535;   // grid-stride loop covers any remainder
    int blocks = static_cast<int>(want_blocks < max_blocks ? want_blocks : max_blocks);
    if (blocks < 1) blocks = 1;

    residual_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(out),
        num_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ----------------------------------------------------------------------------
// residual_add_forward:  out = a + b   (elementwise fp16, out-of-place).
// a, b: identical shapes, any rank. Returns a fresh tensor shaped like a.
// ----------------------------------------------------------------------------
torch::Tensor residual_add_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "residual_add: a and b must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device(),
                "residual_add: a and b must be on the same device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "residual_add: a and b must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf,
                "residual_add: a and b must be float16");
    TORCH_CHECK(a.sizes() == b.sizes(),
                "residual_add: a and b must have identical shapes");

    auto out = torch::empty_like(a);
    launch_residual_add(a.data_ptr<at::Half>(), b.data_ptr<at::Half>(),
                        out.data_ptr<at::Half>(), a.numel());
    return out;
}


// ----------------------------------------------------------------------------
// lm_head_forward:  logits = hidden @ weight^T.
//   hidden: [M, H]        (M = positions to project, H = hidden_size = 896)
//   weight: [vocab, H]    (lm_head.weight; vocab = 151936 for Qwen2.5-0.5B)
// Returns logits: [M, vocab], fp16. Bias-free (Qwen2 lm_head has no bias).
//
// fp16 multiply, fp32 accumulate -- the exact GEMM config used by swiglu's
// linear_no_bias and attention's qkv/o projections.
// ----------------------------------------------------------------------------
torch::Tensor lm_head_forward(torch::Tensor hidden, torch::Tensor weight) {
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(),
                "lm_head: hidden and weight must be CUDA tensors");
    TORCH_CHECK(hidden.device() == weight.device(),
                "lm_head: hidden and weight must be on the same device");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(),
                "lm_head: hidden and weight must be contiguous");
    TORCH_CHECK(hidden.scalar_type() == torch::kHalf && weight.scalar_type() == torch::kHalf,
                "lm_head: hidden and weight must be float16");
    TORCH_CHECK(hidden.dim() == 2, "lm_head: hidden must be 2-D [M, H]");
    TORCH_CHECK(weight.dim() == 2, "lm_head: weight must be 2-D [vocab, H]");
    TORCH_CHECK(hidden.size(1) == weight.size(1),
                "lm_head: hidden inner dim must equal weight inner dim (hidden_size)");

    const int64_t M = hidden.size(0);   // positions
    const int64_t K = hidden.size(1);   // hidden_size, shared inner dim
    const int64_t N = weight.size(0);   // vocab

    // beta = 0, so logits is left uninitialized and the GEMM overwrites it.
    auto logits = at::empty({M, N}, hidden.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major; compute logits^T = weight * hidden^T by swapping
    // M/N and transposing weight (op(A) = T), leaving hidden as-is (op(B) = N).
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                               // op(A): transpose weight
        CUBLAS_OP_N,                               // op(B): leave hidden as-is
        static_cast<int>(N),                       // M and N swapped (column-major)
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        weight.data_ptr<at::Half>(), CUDA_R_16F,   // A pointer + dtype
        static_cast<int>(K),                       // lda
        hidden.data_ptr<at::Half>(), CUDA_R_16F,   // B pointer + dtype
        static_cast<int>(K),                       // ldb
        &beta,
        logits.data_ptr<at::Half>(), CUDA_R_16F,   // C pointer + dtype
        static_cast<int>(N),                       // ldc
        CUBLAS_COMPUTE_32F,                         // fp16 mul, fp32 accumulate
        CUBLAS_GEMM_DEFAULT));                      // let cuBLAS pick the Tensor Core path

    return logits;
}
