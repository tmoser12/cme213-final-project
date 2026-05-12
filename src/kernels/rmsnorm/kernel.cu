#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// Parallel Reduction Utilities
// ----------------------------------------------------------------------------

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared memory for max 32 warps (1024 threads)
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// ----------------------------------------------------------------------------
// Highly Optimized Vectorized RMSNorm Kernel
// ----------------------------------------------------------------------------

__global__ void rmsnorm_forward_kernel_vectorized(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_size,
    float eps
) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 128-bit memory accesses:
    // A float4 is 16 bytes, which exactly equals 8 `half` (FP16) values.
    // Casting our pointers to float4 allows the memory controller to read
    // 128-bits in a single instruction, drastically increasing memory bandwidth utilization.
    const float4* row_input_f4 = reinterpret_cast<const float4*>(input + row_idx * hidden_size);
    const float4* row_weight_f4 = reinterpret_cast<const float4*>(weight);
    float4* row_output_f4 = reinterpret_cast<float4*>(output + row_idx * hidden_size);

    int num_f4 = hidden_size / 8; 

    // Step 1: Compute local sum of squares
    float local_sum = 0.0f;
    
    // We process 8 elements (1 float4) per loop iteration
    for (int i = tid; i < num_f4; i += blockDim.x) {
        float4 val_f4 = row_input_f4[i];
        
        // Treat the 128 bits as an array of four 32-bit half2 vectors
        half2* h2 = reinterpret_cast<half2*>(&val_f4);
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            // __half22float2 converts two FP16s to two FP32s simultaneously
            float2 f2 = __half22float2(h2[j]);
            local_sum += f2.x * f2.x + f2.y * f2.y;
        }
    }

    // Step 2: Block-wide parallel reduction
    float total_sum = blockReduceSum(local_sum);

    // Step 3: Compute the variance and rsqrt
    __shared__ float s_rsqrt_var;
    if (tid == 0) {
        float variance = total_sum / hidden_size;
        s_rsqrt_var = rsqrtf(variance + eps);
    }
    __syncthreads(); // Ensure all threads see the computed rsqrt
    float rsqrt_var = s_rsqrt_var;

    // Step 4: Normalize, apply weights, and write out using vectorized stores
    for (int i = tid; i < num_f4; i += blockDim.x) {
        float4 in_f4 = row_input_f4[i];
        float4 w_f4 = row_weight_f4[i];
        float4 out_f4;

        half2* h2_in = reinterpret_cast<half2*>(&in_f4);
        half2* h2_w = reinterpret_cast<half2*>(&w_f4);
        half2* h2_out = reinterpret_cast<half2*>(&out_f4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f2_in = __half22float2(h2_in[j]);
            float2 f2_w = __half22float2(h2_w[j]);
            float2 f2_out;
            
            f2_out.x = (f2_in.x * rsqrt_var) * f2_w.x;
            f2_out.y = (f2_in.y * rsqrt_var) * f2_w.y;
            
            // Pack back into half2
            h2_out[j] = __float22half2_rn(f2_out);
        }
        
        // Single 128-bit write instruction
        row_output_f4[i] = out_f4;
    }
}

// ----------------------------------------------------------------------------
// C++ Host Function
// ----------------------------------------------------------------------------

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    
    // Validate we can safely use 128-bit aligned reads
    TORCH_CHECK(hidden_size % 8 == 0, "Hidden size must be a multiple of 8 for vectorized kernel");

    auto output = torch::empty_like(input);

    int num_blocks = batch_size * seq_len;
    
    // DYNAMIC OCCUPANCY OPTIMIZATION:
    // We know 1 thread processes 8 elements.
    // If hidden_size is 3584, 3584 / 8 = 448 threads exactly.
    // Since 448 is a multiple of 32 (14 warps) and <= 1024, 
    // we set blockDim.x = 448. This completely eliminates any loop branching!
    int num_threads = 256; 
    int target_threads = hidden_size / 8;
    if (target_threads <= 1024 && target_threads % 32 == 0) {
        num_threads = target_threads;
    } else if (target_threads > 1024) {
        num_threads = 1024;
    }
    
    rmsnorm_forward_kernel_vectorized<<<num_blocks, num_threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        hidden_size,
        eps
    );

    return output;
}
