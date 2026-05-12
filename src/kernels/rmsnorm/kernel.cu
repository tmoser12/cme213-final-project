#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// Parallel Reduction Utilities
// ----------------------------------------------------------------------------

__inline__ __device__ float warpReduceSum(float val) {
    // Use warp shuffle to sum values across a 32-thread warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared memory for max 32 warps (1024 threads)
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // 1. Each warp reduces its values
    val = warpReduceSum(val);

    // 2. The first thread of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); // Wait for all warps to finish writing

    // 3. The first warp reads from shared memory and does a final reduction
    // (Only read valid warp results, pad the rest with 0)
    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// ----------------------------------------------------------------------------
// RMSNorm Kernel
// ----------------------------------------------------------------------------

__global__ void rmsnorm_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float eps
) {
    // In this kernel, one CUDA block processes one token (row).
    // This maps perfectly to the hidden_size dimension.
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Offset pointers to the start of the current row
    const half* row_input = input + row_idx * hidden_size;
    half* row_output = output + row_idx * hidden_size;

    // Step 1: Compute local sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        local_sum += val * val;
    }

    // Step 2: Block-wide parallel reduction to get total sum of squares
    float total_sum = blockReduceSum(local_sum);

    // Step 3: Compute the variance and rsqrt (only thread 0 does the math, then shares it)
    __shared__ float s_rsqrt_var;
    if (tid == 0) {
        float variance = total_sum / hidden_size;
        s_rsqrt_var = rsqrtf(variance + eps);
    }
    __syncthreads(); // Ensure all threads see the computed rsqrt
    float rsqrt_var = s_rsqrt_var;

    // Step 4: Normalize, apply the learned weight, and write output
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        
        float normalized = (val * rsqrt_var) * w;
        row_output[i] = __float2half(normalized);
    }
}

// ----------------------------------------------------------------------------
// C++ Host Function (The bridge between PyTorch and CUDA)
// ----------------------------------------------------------------------------

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    // Assert inputs are on CUDA and are contiguous
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    // Dimensions
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    
    // Allocate output tensor
    auto output = torch::empty_like(input);

    // Grid and Block dimensions
    int num_blocks = batch_size * seq_len; // One block per token
    int num_threads = 256;                 // 256 threads per block is a solid default
    
    // Launch kernel
    rmsnorm_forward_kernel<<<num_blocks, num_threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        hidden_size,
        eps
    );

    return output;
}
