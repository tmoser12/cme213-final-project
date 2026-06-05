// src/kernels/embedding/bindings.cpp
// pybind11 module exposing the CUDA embedding launcher to Python.

#include <torch/extension.h>

// Declared and defined in kernel.cu.
torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward,
          "Qwen embedding gather (CUDA, fp16)");
}
