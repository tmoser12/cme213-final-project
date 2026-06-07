// pybind11 module exposing the SwiGLU launchers to Python.

#include <torch/extension.h>

torch::Tensor swiglu_forward(torch::Tensor x,
                             torch::Tensor W_gate,
                             torch::Tensor W_up,
                             torch::Tensor W_down);
torch::Tensor swiglu_act_forward(torch::Tensor gate, torch::Tensor up);

torch::Tensor swiglu_forward_py(const torch::Tensor& x,
                                const torch::Tensor& W_gate,
                                const torch::Tensor& W_up,
                                const torch::Tensor& W_down) {
    py::gil_scoped_release release;
    return swiglu_forward(x, W_gate, W_up, W_down);
}

torch::Tensor swiglu_act_forward_py(const torch::Tensor& gate,
                                  const torch::Tensor& up) {
    py::gil_scoped_release release;
    return swiglu_act_forward(gate, up);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward_py,
          "Fused SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)) (fp16)");
    m.def("swiglu_act_forward", &swiglu_act_forward_py,
          "Standalone SiLU(gate) * up activation (fp16)");
}
