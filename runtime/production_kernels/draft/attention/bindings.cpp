// pybind11 module exposing the attention sub-op launchers to Python.

#include <torch/extension.h>

torch::Tensor qkv_proj_forward(torch::Tensor x, torch::Tensor W_qkv, torch::Tensor b_qkv);
void          rope_kv_write_forward(torch::Tensor new_k, torch::Tensor new_v,
                                    torch::Tensor cache_k, torch::Tensor cache_v,
                                    int64_t write_pos,
                                    torch::Tensor cos, torch::Tensor sin);
void          rope_kv_write_forward_dev(torch::Tensor new_k, torch::Tensor new_v,
                                        torch::Tensor cache_k, torch::Tensor cache_v,
                                        torch::Tensor write_pos,
                                        torch::Tensor cos, torch::Tensor sin);
torch::Tensor fused_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                 int64_t cur_len, double softmax_scale,
                                 c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor decode_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                  int64_t cur_len, double softmax_scale,
                                  c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor decode_attn_forward_dev(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                      torch::Tensor cur_len, double softmax_scale,
                                      c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor small_q_attn_forward(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                   int64_t cur_len, double softmax_scale,
                                   c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor small_q_attn_forward_dev(torch::Tensor q, torch::Tensor cache_k, torch::Tensor cache_v,
                                       torch::Tensor cur_len, double softmax_scale,
                                       c10::optional<torch::Tensor> cos, c10::optional<torch::Tensor> sin);
torch::Tensor o_proj_forward(torch::Tensor x, torch::Tensor W_o);

torch::Tensor qkv_proj_forward_py(const torch::Tensor& x,
                                  const torch::Tensor& W_qkv,
                                  const torch::Tensor& b_qkv) {
    py::gil_scoped_release release;
    return qkv_proj_forward(x, W_qkv, b_qkv);
}

void rope_kv_write_forward_py(const torch::Tensor& new_k,
                              const torch::Tensor& new_v,
                              const torch::Tensor& cache_k,
                              const torch::Tensor& cache_v,
                              int64_t write_pos,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin) {
    py::gil_scoped_release release;
    rope_kv_write_forward(new_k, new_v, cache_k, cache_v, write_pos, cos, sin);
}

void rope_kv_write_forward_dev_py(const torch::Tensor& new_k,
                                  const torch::Tensor& new_v,
                                  const torch::Tensor& cache_k,
                                  const torch::Tensor& cache_v,
                                  const torch::Tensor& write_pos,
                                  const torch::Tensor& cos,
                                  const torch::Tensor& sin) {
    py::gil_scoped_release release;
    rope_kv_write_forward_dev(new_k, new_v, cache_k, cache_v, write_pos, cos, sin);
}

torch::Tensor fused_attn_forward_py(const torch::Tensor& q,
                                    const torch::Tensor& cache_k,
                                    const torch::Tensor& cache_v,
                                    int64_t cur_len,
                                    double softmax_scale,
                                    c10::optional<torch::Tensor> cos,
                                    c10::optional<torch::Tensor> sin) {
    py::gil_scoped_release release;
    return fused_attn_forward(q, cache_k, cache_v, cur_len, softmax_scale, cos, sin);
}

torch::Tensor decode_attn_forward_py(const torch::Tensor& q,
                                     const torch::Tensor& cache_k,
                                     const torch::Tensor& cache_v,
                                     int64_t cur_len,
                                     double softmax_scale,
                                     c10::optional<torch::Tensor> cos,
                                     c10::optional<torch::Tensor> sin) {
    py::gil_scoped_release release;
    return decode_attn_forward(q, cache_k, cache_v, cur_len, softmax_scale, cos, sin);
}

torch::Tensor small_q_attn_forward_py(const torch::Tensor& q,
                                      const torch::Tensor& cache_k,
                                      const torch::Tensor& cache_v,
                                      int64_t cur_len,
                                      double softmax_scale,
                                      c10::optional<torch::Tensor> cos,
                                      c10::optional<torch::Tensor> sin) {
    py::gil_scoped_release release;
    return small_q_attn_forward(q, cache_k, cache_v, cur_len, softmax_scale, cos, sin);
}

torch::Tensor decode_attn_forward_dev_py(const torch::Tensor& q,
                                         const torch::Tensor& cache_k,
                                         const torch::Tensor& cache_v,
                                         const torch::Tensor& cur_len,
                                         double softmax_scale,
                                         c10::optional<torch::Tensor> cos,
                                         c10::optional<torch::Tensor> sin) {
    py::gil_scoped_release release;
    return decode_attn_forward_dev(q, cache_k, cache_v, cur_len, softmax_scale, cos, sin);
}

torch::Tensor small_q_attn_forward_dev_py(const torch::Tensor& q,
                                          const torch::Tensor& cache_k,
                                          const torch::Tensor& cache_v,
                                          const torch::Tensor& cur_len,
                                          double softmax_scale,
                                          c10::optional<torch::Tensor> cos,
                                          c10::optional<torch::Tensor> sin) {
    py::gil_scoped_release release;
    return small_q_attn_forward_dev(q, cache_k, cache_v, cur_len, softmax_scale, cos, sin);
}

torch::Tensor o_proj_forward_py(const torch::Tensor& x, const torch::Tensor& W_o) {
    py::gil_scoped_release release;
    return o_proj_forward(x, W_o);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_proj_forward", &qkv_proj_forward_py,
          "Fused QKV projection (cuBLAS, fp16)");
    m.def("rope_kv_write_forward", &rope_kv_write_forward_py,
          "RoPE-fused KV write: rotate K + scatter K/V into paged cache (fp16)",
          py::arg("new_k"), py::arg("new_v"), py::arg("cache_k"), py::arg("cache_v"),
          py::arg("write_pos"), py::arg("cos"), py::arg("sin"));
    m.def("rope_kv_write_forward_dev", &rope_kv_write_forward_dev_py,
          "RoPE-fused KV write, device-scalar write_pos (0-d int64 CUDA) for graph capture",
          py::arg("new_k"), py::arg("new_v"), py::arg("cache_k"), py::arg("cache_v"),
          py::arg("write_pos"), py::arg("cos"), py::arg("sin"));
    m.def("fused_attn_forward", &fused_attn_forward_py,
          "Fused causal SDPA, GQA-aware, prefill; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("decode_attn_forward", &decode_attn_forward_py,
          "Decode attention, S==1; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("small_q_attn_forward", &small_q_attn_forward_py,
          "Decode attention, S in [1,8] verify; optional fused RoPE on Q (fp16)",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("decode_attn_forward_dev", &decode_attn_forward_dev_py,
          "Decode attention S==1, device-scalar cur_len (0-d int64 CUDA) for graph capture",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("small_q_attn_forward_dev", &small_q_attn_forward_dev_py,
          "Decode attention S in [1,8], device-scalar cur_len (0-d int64 CUDA) for graph capture",
          py::arg("q"), py::arg("cache_k"), py::arg("cache_v"), py::arg("cur_len"),
          py::arg("softmax_scale"), py::arg("cos") = py::none(), py::arg("sin") = py::none());
    m.def("o_proj_forward", &o_proj_forward_py,
          "Output projection (cuBLAS, fp16)");
}
