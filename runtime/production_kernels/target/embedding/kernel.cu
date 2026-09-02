// kernel_dev/target/kernels/embedding/kernel.cu
// Custom CUDA embedding (indexed row gather) for Qwen2.5.
//
// Semantics: out[..., :] = weight[input_ids[...], :]
// Matches torch.nn.functional.embedding for the no-padding_idx, no-scale,
// no-renorm case -- which is all Qwen2's embed_tokens does at inference.
//
// Pure gather: output is bit-exact equal to the PyTorch reference.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>    // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cuda_fp16.h>
#include <cstdint>

namespace {

// One block per output token. Threads in the block stride across the hidden
// dimension to copy one row of `weight` into one row of `output`. With H=3584
// and blockDim.x=128 each thread touches 28 halves; per-warp loads are
// contiguous so global memory accesses are fully coalesced.
__global__ void embedding_kernel(
    const int64_t* __restrict__ input_ids,  // [N] (seq_len)
    const __half*  __restrict__ weight,     // [V, H] (152K, 3584)
    __half*        __restrict__ output,     // [N, H] (seq_len, 3584)
    int64_t N,
    int64_t H)
{
    int64_t n = blockIdx.x; 
    if (n >= N) return;

    int64_t id = input_ids[n];
    const __half* src = weight + id * H;
    __half*       dst = output + n  * H;

    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace


torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight) {
    TORCH_CHECK(input_ids.is_cuda(),       "input_ids must be CUDA");
    TORCH_CHECK(weight.is_cuda(),          "weight must be CUDA");
    TORCH_CHECK(input_ids.device() == weight.device(),
                "input_ids and weight must be on the same device");
    TORCH_CHECK(weight.dim() == 2,         "weight must be 2-D [V, H]");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf,
                "weight must be float16");
    TORCH_CHECK(input_ids.scalar_type() == torch::kLong,
                "input_ids must be int64");
    TORCH_CHECK(input_ids.is_contiguous(), "input_ids must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),    "weight must be contiguous");

    const int64_t H = weight.size(1); // 3584, hidden_dim
    const int64_t N = input_ids.numel(); // seq_len

    // Output shape: input_ids.shape + (H,)
    auto out_shape = input_ids.sizes().vec(); // [seq_len, H]
    out_shape.push_back(H);
    auto output = torch::empty(out_shape, weight.options());

    if (N == 0) return output;

    const int threads = 128;
    embedding_kernel<<<static_cast<unsigned int>(N), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_ids.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, H);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
