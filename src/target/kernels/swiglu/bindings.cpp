// src/kernels/swiglu/bindings.cpp
// pybind11 module exposing the SwiGLU launchers to Python.

#include <torch/extension.h>

// Forward declarations of the C++ host functions from kernel.cu.
torch::Tensor swiglu_forward(torch::Tensor x,
                             torch::Tensor W_gate,
                             torch::Tensor W_up,
                             torch::Tensor W_down);
torch::Tensor swiglu_act_forward(torch::Tensor gate, torch::Tensor up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward,
          "Fused SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)) (fp16)");
    m.def("swiglu_act_forward", &swiglu_act_forward,
          "Standalone SiLU(gate) * up activation (fp16)");
}
