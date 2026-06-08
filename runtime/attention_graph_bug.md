# CUDA-graph decode NaN — root cause & fix (RESOLVED)

**Symptom.** Captured decode/verify graphs returned all-NaN (or wrong/zero) logits
past ~22 decoder layers, while eager decode was correct. Surfaced when wiring the
split-KV decode attention into the graph runtime.

**Root cause (NOT the attention kernel).** Stale compiled extensions. Several
`runtime/production_kernels/{draft,target}/*/kernel.cu` had been edited but their
`.so` files were never rebuilt. The stale `residual_ops` `.so` launched
`residual_add` on the **default stream** instead of `getCurrentCUDAStream()`, so
under capture it ran eagerly and recorded **no work** ("CUDA Graph is empty"
warning). On replay the `residual_add` was a no-op, so `hidden = residual_add(...)`
returned an **uninitialized buffer**. That garbage read as wrong/NaN/zero depending
on graph-pool state and depth — hence the apparent ~22-layer threshold and the
red-herring trails (split kernel, scratch allocation, graph-pool aliasing). It was
never the split-KV kernel: that is bit-exact eager and under capture.

**Fix.** Rebuild all kernels so `.so` matches source:
`bash scripts/build_kernels.sh {draft,target} <kernel>` (or `… {draft,target}` for
all). No executor or kernel logic change was needed.

**Verified.** Draft decode graph: NaN-free, bit-exact vs eager (`max|Δ|=0.0`),
~1.7× speedup. Target (28 layers): NaN-free, `max|Δ|≈0.02` (split vs single-block
reduction order), ~1.0× (memory-bound, as expected).

**Prevention.** A stale `.so` fails silently. After editing any `kernel.cu`, always
rebuild; consider a source-vs-`.so` mtime check at extension load time.

Eli's Claude Sessions:
Resume performance modeling discussion: claude -r 86a2ff9f-a4b6-4f38-98da-7f7e559b8d6b

