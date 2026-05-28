// src/kernels/attention/bindings.cpp
// pybind11 module exposing the five attention sub-op launchers to Python.

#include <torch/extension.h>

torch::Tensor qkv_proj_forward(torch::Tensor x, torch::Tensor W_qkv, torch::Tensor b_qkv);
void          rope_forward(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
void          kv_write_forward(torch::Tensor new_k, torch::Tensor new_v,
                               torch::Tensor cache_k, torch::Tensor cache_v,
                               int64_t write_pos);
torch::Tensor fused_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                 int64_t cur_len, double softmax_scale);
torch::Tensor o_proj_forward(torch::Tensor x, torch::Tensor W_o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_proj_forward",   &qkv_proj_forward,   "Fused QKV projection (cuBLAS, fp16)");
    m.def("rope_forward",       &rope_forward,       "In-place RoPE on Q and K (fp16)");
    m.def("kv_write_forward",   &kv_write_forward,   "Scatter new K/V into paged cache (fp16)");
    m.def("fused_attn_forward", &fused_attn_forward, "Fused causal SDPA, GQA-aware (fp16)");
    m.def("o_proj_forward",     &o_proj_forward,     "Output projection (cuBLAS, fp16)");
}
