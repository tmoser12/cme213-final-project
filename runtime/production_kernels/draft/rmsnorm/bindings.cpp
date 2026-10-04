#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps);

torch::Tensor forward_py(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       float eps) {
    py::gil_scoped_release release;
    return rmsnorm_forward(input, weight, eps);
}

// WALKTHROUGH [DRAFT_RMSNORM_BINDING] — see demo/WALKTHROUGH.md.
// Expose C++ host launchers as Python-callable functions. The wrappers release
// the GIL while dispatching; tensor storage remains on the GPU.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_py, "Fused RMSNorm forward pass");
}
