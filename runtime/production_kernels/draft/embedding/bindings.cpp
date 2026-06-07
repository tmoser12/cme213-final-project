// pybind11 module exposing the CUDA embedding launcher to Python.

#include <torch/extension.h>

torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight);

torch::Tensor embedding_forward_py(const torch::Tensor& input_ids,
                                   const torch::Tensor& weight) {
    py::gil_scoped_release release;
    return embedding_forward(input_ids, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward_py,
          "Qwen embedding gather (CUDA, fp16)");
}
