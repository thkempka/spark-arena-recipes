#!/usr/bin/env bash
# fetch-models.sh — pre-pull the HF models/drafters into the sparkrun HF cache.
#
# Usage:  HF_TOKEN=hf_xxx ./fetch-models.sh
# (HF_TOKEN only needed for gated repos; harmless otherwise.)
set -euo pipefail

HF_CACHE="${HF_CACHE:-/cache/huggingface}"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

MODELS=(
  "RadixArk/Qwen3.8-27B-NVFP4"
  "z-lab/Qwen3.8-27B-DFlash2"
  "deepseek-ai/DeepSeek-V4-Flash-0731"
)
# pinned draft revision (from the qwen38 recipes)
DRAFT2_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"

huggingface-cli download --help >/dev/null 2>&1 && HFDL=huggingface-cli \
  || HFDL="python3 -m huggingface_hub"

say() { printf '\033[1;36m[fetch]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

for m in "${MODELS[@]}"; do
  say "downloading $m -> $HF_CACHE"
  $HFDL download "$m" --local-dir "$HF_CACHE" || die "download failed for $m"
done

say "pinning draft revision $DRAFT2_REV (z-lab/Qwen3.8-27B-DFlash2)"
$HFDL download "z-lab/Qwen3.8-27B-DFlash2" --revision "$DRAFT2_REV" --local-dir "$HF_CACHE" || die "draft rev pin failed"

ok() { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
ok "all models cached under $HF_CACHE"
