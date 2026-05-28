# RMSNorm Kernel Walkthrough

This document breaks down exactly how the custom `rmsnorm_forward_kernel_vectorized` works under the hood. 

RMSNorm (Root Mean Square Normalization) is heavily used in modern LLMs like Qwen, LLaMA, and Mistral because it is computationally cheaper than standard LayerNorm (it skips centering the mean) while providing the same stability.

The mathematical formula for RMSNorm is:
1. **Variance:** $\text{Var} = \frac{1}{H} \sum_{i=1}^{H} x_i^2$
2. **Normalize:** $\hat{x}_i = x_i \times \frac{1}{\sqrt{\text{Var} + \epsilon}}$
3. **Scale:** $y_i = \hat{x}_i \times w_i$

Where $H$ is the `hidden_size` (3584 for Qwen 7B), and $w$ is the learned weight.

---

## 1. Grid and Block Mapping
The very first thing a kernel does is map its execution to the hardware. 
In our kernel, we use a 1D grid and a 1D block:
*   **1 Block = 1 Token (Row):** `blockIdx.x` represents the specific token in the sequence. If we have a batch size of 2 and a sequence length of 128, we launch 256 blocks. 
*   **Threads = Hidden Dimension Elements:** `threadIdx.x` represents a worker assigned to process a chunk of the 3584 elements within that token. 

Because the `hidden_size` is exactly 3584, and we process 8 elements per thread, we launch exactly **448 threads per block** (3584 / 8 = 448).

---

## 2. Phase 1: The 128-bit Vectorized Read
Element-wise operations like RMSNorm are **Memory Bandwidth Bound**, meaning the GPU spends 95% of its time waiting for data to arrive from VRAM and only 5% of its time doing math.

```cpp
const float4* row_input_f4 = reinterpret_cast<const float4*>(input + row_idx * hidden_size);
float4 val_f4 = row_input_f4[tid];
```

To optimize this, we cast the standard `half` (16-bit) pointer into a `float4` (128-bit) pointer. 
*   By doing this, a single thread reads 128 bits in a single hardware instruction. 
*   128 bits is exactly 8 `half` values.
*   This perfectly aligns with the Turing architecture's 128-byte cache lines, ensuring we get the absolute maximum possible memory bandwidth (672 GB/s on the RTX 6000).

---

## 3. Phase 2: Hardware FP16 Math
```cpp
half2* h2 = reinterpret_cast<half2*>(&val_f4);
for (int j = 0; j < 4; ++j) {
    float2 f2 = __half22float2(h2[j]);
    local_sum += f2.x * f2.x + f2.y * f2.y;
}
```
Inside the thread, we have 8 `half` values. Instead of converting them to standard FP32 floats one by one, we treat them as four `half2` vectors. We use the intrinsic `__half22float2` to cast a pair of FP16 values to FP32 in a single hardware step, then square and accumulate them into a `local_sum`.

---

## 4. Phase 4: Block-Wide Parallel Reduction
Now, each of our 448 threads has a `local_sum` representing the sum of squares for its specific 8 elements. We need to add all 448 of these numbers together to get the total sum for the entire token. 

If one thread did this sequentially, it would be extremely slow. Instead, we use a **Tree Reduction**:

**Step 4a: Warp Reduction (`__shfl_down_sync`)**
Threads on an NVIDIA GPU execute in groups of 32 called "Warps". Threads in the same warp can read each other's registers instantly without touching memory. We use `__shfl_down_sync` to instantly sum the 32 values within each warp in just 5 hardware steps.

**Step 4b: Shared Memory Reduction**
Our 448 threads make up exactly 14 warps. 
The 14 "lead" threads of each warp take their warp's total sum and write it to **Shared Memory** (the ultra-fast L1 cache onboard the Streaming Multiprocessor).
Finally, the very first warp (Warp 0) reads those 14 values from shared memory and does one final warp reduction to get the absolute `total_sum`.

---

## 5. Phase 5: Normalization and Write-Back
```cpp
if (tid == 0) {
    float variance = total_sum / hidden_size;
    s_rsqrt_var = rsqrtf(variance + eps);
}
__syncthreads();
```
Thread 0 calculates the Inverse Square Root (`rsqrtf`) of the variance, and saves it to shared memory so all 448 threads can see it simultaneously.

Finally, every thread recalculates its 8 elements: it unpacks its original input values, multiplies them by the `rsqrt_var`, multiplies them by the learned `weight`, and packs them back into a `float4` tensor.

```cpp
row_output_f4[tid] = out_f4; // 128-bit write!
```
The thread writes all 8 values back to VRAM in a single 128-bit instruction, completely saturating the memory bus on the way out!
