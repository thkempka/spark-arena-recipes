# DSpark kernel-work image for Qwen3.8-Flash-Next-NVFP4 on DGX Spark (SM121/GB10).
# See start.sh header for the full story.
#
# Patches (both guarded, refuse if source drifted):
#   1. qwen_sparse_attn_backend.py  - SM121 QSA FA4-cute MLIR fallback guard
#   2. rotary_triton.py             - mrope partial-rotary OOB fix (qwen4_exp;
#      from tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark, Dockerfile.qwen38fn-sm121-mrope)
ARG BASE_IMAGE=lmsysorg/sglang:qwen38flashnext
FROM ${BASE_IMAGE}

COPY qsa_fa_fallback.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_fa_fallback.py
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
patch = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + anchor
assert anchor in s, "anchor not found — upstream image layout changed"
assert "qsa_fa_fallback" not in s, "already patched"
open(p, "w").write(s.replace(anchor, patch, 1))
print("qwen_sparse_attn_backend.py patched for SM121")
PYEOF
RUN python3 - <<'PYEOF'
# mrope partial-rotary OOB fix: the fused mrope Triton kernel pads offsets to
# next_power_of_2(head_size)//2 lanes but qwen4_exp is the first mrope model
# with partial rotary (rotary cache row 32 half-lanes vs 128 padded lanes).
# The temporal mask t_mask = ~(h_mask | w_mask) is unbounded above half_rd,
# producing out-of-bounds cache reads on every image call and walking off the
# cache end near the 262K context edge -> candidate origin of the CUDA
# device-side assert that surfaces in _compute_mrope_positions_extend and
# killed the endpoint every ~90 min of load. Bounding t_mask is provably a
# no-op for every full-rotary model (the neighboring w_mask is already bounded).
import glob
from pathlib import Path

cands = [p for p in glob.glob("/sgl-workspace/sglang/**/rotary_triton.py", recursive=True)]
if len(cands) != 1:
    raise SystemExit(f"expected exactly one rotary_triton.py, found: {cands}")
path = Path(cands[0])
src = path.read_text()

old = "t_mask = ~(h_mask | w_mask)"
new = "t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)"
if new in src:
    raise SystemExit("already patched; refusing to double-patch")
n = src.count(old)
if n < 1:
    raise SystemExit("unbounded t_mask pattern not found; source drifted, refusing to patch")
for var in ("cos_offsets", "half_rd"):
    if var not in src:
        raise SystemExit(f"expected variable {var} missing from kernel; refusing to patch")

path.write_text(src.replace(old, new))
print(f"PATCHED {path}: bounded {n} occurrence(s) of t_mask")
PYEOF
