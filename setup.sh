#!/usr/bin/env bash
# spark-arena-recipes / setup.sh
#
# One-time provisioning so the 4 recipes are runnable on this host:
#   1. pull the two public registry images (digest-pinned, no build)
#   2. build qwen38-dflash2:v1.2.2 offline (public base + sha256-verified overlay)
#      only if not present
#   3. validate QWEN38_API_KEY is set (required by the qwen38 recipes)
#   4. (optional) check HF model caches are present
#
# Usage:  ./setup.sh            # full
#         ./setup.sh --check    # only report status, change nothing
#         QWEN38_API_KEY=... ./setup.sh
#
# Idempotent: images already present are skipped.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# --- pinned artifacts -------------------------------------------------------
B12X_IMAGE="ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x@sha256:af9629c5bc9f5edc1b478a996cf4aa4304311524f31e61ab66e59fce8ff8fd02"
BJKDSPARK_IMAGE="ghcr.io/bjk110/vllm-spark@sha256:d8492e7677cf1b9aaa3344e0e6865efc468454013eee5ebabac85be90af027be"
QWEN38_BASE="lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"  # = lmsysorg/sglang:qwen38-27b
QWEN38_TAG="qwen38-dflash2:v1.2.2"
QWEN38_BUILD_DIR="$DIR/docker/qwen38-dflash2"

HF_MODELS=(
  "RadixArk/Qwen3.8-27B-NVFP4"
  "z-lab/Qwen3.8-27B-DFlash2"
  "deepseek-ai/DeepSeek-V4-Flash-0731"
)

say()  { printf "\033[1;36m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[ ok ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[err ]\033[0m %s\n" "$*" >&2; exit 1; }

image_present() { docker image inspect "$1" >/dev/null 2>&1; }

do_pull() {
  local img="$1"
  if image_present "$img"; then ok "image present: ${img#*@}"; return 0; fi
  say "pulling $img"
  docker pull "$img" || die "docker pull failed for $img"
}

do_build_qwen38() {
  if image_present "$QWEN38_TAG"; then ok "image present: $QWEN38_TAG"; return 0; fi
  [ -f "$QWEN38_BUILD_DIR/build-image.sh" ] || die "missing $QWEN38_BUILD_DIR/build-image.sh (did you clone the full repo?)"
  if ! image_present "$QWEN38_BASE"; then
    say "pulling qwen38 base image (~39 GB, one-time)"
    docker pull "$QWEN38_BASE" || die "base pull failed (try later or 'docker login')"
  fi
  say "building $QWEN38_TAG from pinned base + sha256-verified overlay (offline, ~1 min)"
  BASE_IMAGE="$QWEN38_BASE" TAG="$QWEN38_TAG" "$QWEN38_BUILD_DIR/build-image.sh" \
    || die "qwen38 build failed (see build-image.sh; MANIFEST.sha256 must match)"
  ok "built $QWEN38_TAG"
}

check_api_key() {
  if [ -z "${QWEN38_API_KEY:-}" ]; then
    warn "QWEN38_API_KEY is not set — qwen38 recipes will fail at serve time."
    warn "Set it, e.g.: QWEN38_API_KEY=... ./setup.sh  (or export it / put in a .env)"
    return 1
  fi
  ok "QWEN38_API_KEY is set (${#QWEN38_API_KEY} chars)"
  return 0
}

check_hf_models() {
  local dirs=()
  # candidate HF cache roots: explicit HF_HOME, /cache/huggingface (sparkrun)
  # then the default ~/.cache/huggingface
  [ -n "${HF_HOME:-}" ] && dirs+=("$HF_HOME")
  [ -d /cache/huggingface ] && dirs+=("/cache/huggingface")
  dirs+=("$HOME/.cache/huggingface")
  local missing=0
  for m in "${HF_MODELS[@]}"; do
    local found=0
    for d in "${dirs[@]}"; do
      if [ -d "$d/hub/models--$(echo "$m" | tr / _)" ]; then found=1; break; fi
    done
    if [ $found -eq 1 ]; then ok "HF cache: $m"; else warn "HF model not cached: $m (pull with HF_TOKEN set)"; missing=1; fi
  done
  [ $missing -eq 0 ]
}

echo
say "spark-arena-recipes provisioning"
say "dir: $DIR"
echo

if [ "$CHECK_ONLY" -eq 1 ]; then
  say "check-only mode (no changes)"
  for img in "$B12X_IMAGE" "$BJKDSPARK_IMAGE" "$QWEN38_TAG" "$QWEN38_BASE"; do
    if image_present "$img"; then ok "present: ${img#*@}"; else warn "MISSING: ${img#*@}"; fi
  done
  check_api_key || true
  check_hf_models || true
  exit 0
fi

# --- registry images (digest-pinned) ----------------------------------------
say "1/4 registry images"
do_pull "$B12X_IMAGE"
do_pull "$BJKDSPARK_IMAGE"

# --- qwen38 image (offline build if missing) --------------------------------
say "2/4 qwen38-dflash2 image"
do_build_qwen38

# --- API key ----------------------------------------------------------------
say "3/4 QWEN38_API_KEY"
check_api_key || true

# --- HF model caches --------------------------------------------------------
say "4/4 HF model caches"
check_hf_models || true

echo
ok "setup complete. Run e.g.:"
echo "    QWEN38_API_KEY=\$QWEN38_API_KEY sparkrun run qwen38-dflash2-sglang-a.yaml --solo"
echo "    sparkrun run deepseek-v4-flash-0731-b12x-dspark-vllm-patched.yaml --cluster wc"
