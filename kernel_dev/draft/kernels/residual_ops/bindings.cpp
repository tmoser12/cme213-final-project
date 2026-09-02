// kernel_dev/target/kernels/residual_ops/bindings.cpp
// pybind11 module exposing the residual-add and LM-head launchers to Python.

#include <torch/extension.h>

// Forward declarations of the C++ host functions from kernel.cu.
torch::Tensor residual_add_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor lm_head_forward(torch::Tensor hidden, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residual_add_forward", &residual_add_forward,
          "Elementwise residual add: out = a + b (fp16, vectorized)");
    m.def("lm_head_forward", &lm_head_forward,
          "LM head projection: logits = hidden @ weight^T (cuBLAS, fp16)");
}
