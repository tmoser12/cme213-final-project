// src/kernels/swiglu/kernel.cu
// Custom CUDA SwiGLU MLP for Qwen2.5.
//
// Target op (src/models/modeling_qwen2.py Qwen2MLP.forward):
//     down_proj( silu(gate_proj(x)) * up_proj(x) )
//
// Split of labor:
//   * gate_proj / up_proj / down_proj  -> cuBLAS GEMMs, driven in the host
//     launcher below (all three are bias-free; Qwen2 sets bias=False), exactly
//     like attention's qkv_proj_forward / o_proj_forward.
//   * swiglu_act_kernel                -> the one custom kernel: the elementwise
//     fusion  hidden = silu(gate) * up,  where  silu(x) = x * sigmoid(x).
//
// swiglu_forward returns the down_proj output [M, H] -- i.e. the MLP result that
// the decoder layer adds straight into the residual stream (the residual add
// itself is a separate kernel; see the roadmap in CLAUDE.md). The standalone
// swiglu_act_forward(gate, up) is also exposed for granular kernel testing.
//
// The activation is memory-bound (read gate, read up, write hidden), so it
// mirrors rmsnorm's float4 / half2 vectorized pattern: cast the half* buffers
// to float4* for 128-bit loads/stores and do the silu*mul in fp32 on half2
// lanes. See src/kernels/rmsnorm/kernel.cu and kernel_walkthrough.md.

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
// Bias-free linear:  y = x @ W^T.   x:[M, K], W:[N, K] -> y:[M, N].
// fp16 multiply, fp32 accumulate. Matches attention's o_proj_forward GEMM.
// ----------------------------------------------------------------------------
static torch::Tensor linear_no_bias(torch::Tensor x, torch::Tensor W) {
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);   // shared inner dim
    const int64_t N = W.size(0);   // output features

    // beta = 0, so y is left uninitialized and the GEMM overwrites it.
    auto y = at::empty({M, N}, x.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major; we compute y^T = W * x^T by swapping M/N and
    // transposing W (op(A) = T), leaving x as-is (op(B) = N).
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                          // op(A): transpose W (col-major view)
        CUBLAS_OP_N,                          // op(B): leave x as-is
        static_cast<int>(N),                  // M and N swapped (column-major)
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        W.data_ptr<at::Half>(), CUDA_R_16F,   // A pointer + dtype
        static_cast<int>(K),                  // lda
        x.data_ptr<at::Half>(), CUDA_R_16F,   // B pointer + dtype
        static_cast<int>(K),                  // ldb
        &beta,
        y.data_ptr<at::Half>(), CUDA_R_16F,   // C pointer + dtype
        static_cast<int>(N),                  // ldc
        CUBLAS_COMPUTE_32F,                   // fp16 mul, fp32 accumulate
        CUBLAS_GEMM_DEFAULT));                // let cuBLAS pick the Tensor Core path

    return y;
}


// ----------------------------------------------------------------------------
// Custom kernel:  out = silu(gate) * up,  silu(x) = x * sigmoid(x).
// Flat elementwise over the whole [M, I] buffer. 1 thread -> 8 halves (one
// float4 = 128-bit load/store), grid-stride so any element count is covered.
// In-place safe (out may alias gate): each lane reads index i and writes i.
// ----------------------------------------------------------------------------
__global__ void swiglu_act_kernel(
    const __half* __restrict__ gate,   // [numel]
    const __half* __restrict__ up,     // [numel]
    __half*       __restrict__ out,    // [numel] (may alias gate)
    int64_t num_vec)                   // numel / 8
{
    const float4* gate4 = reinterpret_cast<const float4*>(gate);
    const float4* up4   = reinterpret_cast<const float4*>(up);
    float4*       out4  = reinterpret_cast<float4*>(out);

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_vec;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float4 g4 = gate4[i];
        float4 u4 = up4[i];
        float4 o4;

        // Treat the 128 bits as four 32-bit half2 lanes.
        half2* gh = reinterpret_cast<half2*>(&g4);
        half2* uh = reinterpret_cast<half2*>(&u4);
        half2* oh = reinterpret_cast<half2*>(&o4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 gf = __half22float2(gh[j]);
            float2 uf = __half22float2(uh[j]);
            float2 of;
            // silu(g) = g * sigmoid(g) = g / (1 + e^{-g}); then * up.
            of.x = (gf.x / (1.0f + __expf(-gf.x))) * uf.x;
            of.y = (gf.y / (1.0f + __expf(-gf.y))) * uf.y;
            oh[j] = __float22half2_rn(of);
        }
        out4[i] = o4;
    }
}

// Launch helper shared by swiglu_act_forward and swiglu_forward.
static void launch_swiglu_act(const at::Half* gate, const at::Half* up,
                              at::Half* out, int64_t numel) {
    TORCH_CHECK(numel % 8 == 0,
                "swiglu_act: element count must be a multiple of 8 for the vectorized kernel");
    const int64_t num_vec = numel / 8;

    const int threads = 256;
    int64_t want_blocks = (num_vec + threads - 1) / threads;
    // Cap the grid; the grid-stride loop covers any remainder.
    const int64_t max_blocks = 65535;
    int blocks = static_cast<int>(want_blocks < max_blocks ? want_blocks : max_blocks);
    if (blocks < 1) blocks = 1;

    swiglu_act_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(gate),
        reinterpret_cast<const __half*>(up),
        reinterpret_cast<__half*>(out),
        num_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ----------------------------------------------------------------------------
// Standalone activation:  out = silu(gate) * up   (elementwise, fp16).
// gate, up: [M, I]. Exposed for granular kernel testing.
// ----------------------------------------------------------------------------
torch::Tensor swiglu_act_forward(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(),
                "swiglu_act: gate and up must be CUDA tensors");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(),
                "swiglu_act: gate and up must be contiguous");
    TORCH_CHECK(gate.scalar_type() == torch::kHalf && up.scalar_type() == torch::kHalf,
                "swiglu_act: gate and up must be float16");
    TORCH_CHECK(gate.sizes() == up.sizes(),
                "swiglu_act: gate and up must have identical shapes");
    TORCH_CHECK(gate.size(-1) % 8 == 0,
                "swiglu_act: last dim must be a multiple of 8 for the vectorized kernel");

    auto out = torch::empty_like(gate);
    launch_swiglu_act(gate.data_ptr<at::Half>(), up.data_ptr<at::Half>(),
                      out.data_ptr<at::Half>(), gate.numel());
    return out;
}


// ----------------------------------------------------------------------------
// Full fused SwiGLU MLP:  out = down_proj( silu(gate_proj(x)) * up_proj(x) ).
//   x:       [M, H]   (M = batch*seq, H = hidden_size = 3584 for 7B)
//   W_gate:  [I, H]   (gate_proj.weight; I = intermediate_size = 18944 for 7B)
//   W_up:    [I, H]   (up_proj.weight)
//   W_down:  [H, I]   (down_proj.weight)
// Returns out: [M, H] -- the MLP output, ready to add to the residual stream.
// ----------------------------------------------------------------------------
torch::Tensor swiglu_forward(torch::Tensor x,
                             torch::Tensor W_gate,
                             torch::Tensor W_up,
                             torch::Tensor W_down) {
    TORCH_CHECK(x.is_cuda() && W_gate.is_cuda() && W_up.is_cuda() && W_down.is_cuda(),
                "swiglu: all tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && W_gate.is_contiguous() &&
                W_up.is_contiguous() && W_down.is_contiguous(),
                "swiglu: all tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf &&
                W_gate.scalar_type() == torch::kHalf &&
                W_up.scalar_type() == torch::kHalf &&
                W_down.scalar_type() == torch::kHalf,
                "swiglu: all tensors must be float16");
    TORCH_CHECK(x.dim() == 2,      "swiglu: x must be 2-D [M, H]");
    TORCH_CHECK(W_gate.dim() == 2 && W_up.dim() == 2 && W_down.dim() == 2,
                "swiglu: projection weights must be 2-D");

    const int64_t H = x.size(1);          // hidden_size
    const int64_t I = W_gate.size(0);     // intermediate_size

    TORCH_CHECK(W_gate.size(1) == H, "swiglu: W_gate inner dim must equal hidden_size");
    TORCH_CHECK(W_up.size(0) == I && W_up.size(1) == H,
                "swiglu: W_up must match W_gate shape [I, H]");
    TORCH_CHECK(W_down.size(0) == H && W_down.size(1) == I,
                "swiglu: W_down must be [H, I]");
    TORCH_CHECK(I % 8 == 0,
                "swiglu: intermediate_size must be a multiple of 8 for the vectorized kernel");

    // 1) gate = x @ W_gate^T   -> [M, I]
    auto gate = linear_no_bias(x, W_gate);
    // 2) up   = x @ W_up^T     -> [M, I]
    auto up   = linear_no_bias(x, W_up);
    // 3) gate = silu(gate) * up   (in place into the gate buffer) -> [M, I]
    launch_swiglu_act(gate.data_ptr<at::Half>(), up.data_ptr<at::Half>(),
                      gate.data_ptr<at::Half>(), gate.numel());
    // 4) out  = gate @ W_down^T   -> [M, H]
    auto out  = linear_no_bias(gate, W_down);

    return out;
}
