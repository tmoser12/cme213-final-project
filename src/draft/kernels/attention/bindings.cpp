// src/kernels/attention/bindings.cpp
// pybind11 module exposing the attention sub-op launchers to Python.

#include <torch/extension.h>

torch::Tensor qkv_proj_forward(torch::Tensor x, torch::Tensor W_qkv, torch::Tensor b_qkv);
void          rope_kv_write_forward(torch::Tensor new_k, torch::Tensor new_v,
                                    torch::Tensor cache_k, torch::Tensor cache_v,
                                    int64_t write_pos,
                                    torch::Tensor cos, torch::Tensor sin);
torch::Tensor fused_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                 int64_t cur_len, double softmax_scale,
                                 c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor decode_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                  int64_t cur_len, double softmax_scale,
                                  c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor small_q_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                   int64_t cur_len, double softmax_scale,
                                   c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor o_proj_forward(torch::Tensor x, torch::Tensor W_o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_proj_forward",    &qkv_proj_forward,    "Fused QKV projection (cuBLAS, fp16)");
    m.def("rope_kv_write_forward", &rope_kv_write_forward,
          "RoPE-fused KV write: rotate K + scatter K/V into paged cache (fp16)",
          py::arg("new_k"), py::arg("new_v"), py::arg("cache_k"), py::arg("cache_v"),
          py::arg("write_pos"), py::arg("cos"), py::arg("sin"));
    m.def("fused_attn_forward",  &fused_attn_forward,
          "Fused causal SDPA, GQA-aware, prefill; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("decode_attn_forward", &decode_attn_forward,
          "Decode attention, S==1; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("small_q_attn_forward",&small_q_attn_forward,
          "Decode attention, S in [1,8] verify; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("o_proj_forward",      &o_proj_forward,      "Output projection (cuBLAS, fp16)");
}
