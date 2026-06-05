# GPU Specification Profile: Quadro RTX 6000

Technical specifications for the Quadro RTX 6000 (Turing Architecture) on the `gpu-turing` partition. Use when optimizing CUDA kernels, tuning memory, or choosing precision.

## Hardware Overview

- **Architecture:** Turing (TU102)
- **Compute Capability:** SM 7.5
- **CUDA Cores:** 4,608
- **Tensor Cores:** 576 (2nd Gen)
- **RT Cores:** 72
- **Process Node:** 12nm FFN (TSMC)

## Memory Specifications

- **Total VRAM:** 24 GB GDDR6
- **Memory Interface:** 384-bit
- **Memory Bandwidth:** 672 GB/s
- **Memory Clock:** 14 Gbps (effective)
- **L2 Cache Size:** 6 MB
- **ECC Support:** Enabled

## Compute Performance

- **FP32 Performance:** ~16.3 TFLOPS
- **Tensor Performance:** Supports FP16, INT8, and INT4 (No TF32/BF16 hardware acceleration)
- **Max Shared Memory:** Configurable up to 96 KB per SM (combined with L1)

## CUDA Resource Limits (SM 7.5)

- **Max Threads per Block:** 1024
- **Max Registers per Block:** 64K (32-bit)
- **Max Thread Blocks per SM:** 16
- **Warp Size:** 32

## Software Context

- **PyTorch Version:** 2.4.0+cu121
- **CUDA Runtime:** 12.1
- **Driver/Bus:** PCIe 3.0 x16
- **NVCC compile flag:** `-arch=sm_75`

## Optimization Directives

- **Precision:** Focus on FP16 mixed-precision using Tensor Cores. Turing lacks hardware-level BF16 support found in Ampere.
- **Memory:** Address shared memory bank conflicts for SM 7.5 architecture.
- **Occupancy:** Calculate occupancy based on 64K registers and a maximum of 1024 threads per block.
