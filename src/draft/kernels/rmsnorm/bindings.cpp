#include <torch/extension.h>

// Forward declaration of the C++ host function from kernel.cu
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps);

// Expose the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "Fused RMSNorm forward pass");
}
