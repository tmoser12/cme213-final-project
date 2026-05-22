# QKV Projection

Qwen uses GQA attention, 28 query heads and 4 key-value heads.

For each 7 query heads, there are 4 key-value heads.

Each QKV matrix is a 128x3584 matrix.

So when stacked together, the fused QKV projection weight is: 28*128 + 2 * 4 * 128 = 4608x3584
With the bias is 28*128 + 2 * 4 * 128 = 4608x1

cuBLAS assumes data is column-major, so we need to transpose the weight matrix and compute x @ W^T
x: (tok x hidden=3584) @ w^T: (hidden x 4608) = (tok x 4608) = y: tok x 4608

# RoPE

