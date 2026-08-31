#!/bin/bash
# glm53-serve.sh - serve GLM-5.3-Flash EXL3 with post-health boot-shape warmup.
#
# sparkrun appends the per-rank distributed flags (--nnodes N --node-rank R
# --master-addr H --master-port P, plus --headless on workers) to the END of
# the recipe command; they arrive here as positional parameters and are
# forwarded verbatim to `vllm serve` via "$@".
#
# Order of operations (matches ~/glm53-miaai-pure start.sh flow):
#   1. apply the 7 overlay patches in place (fresh repo versions are bind-
#      mounted over /opt/glm53 by executor_config.volumes)
#   2. launch vllm serve in the background with ALL args, including the
#      sparkrun-injected distributed flags
#   3. wait for /health
#   4. burn DFlash2 k=7 / sampler / kpool shapes BEFORE real traffic -
#      without this, shape-specialized Triton kernels JIT mid-serve on TP=2
#      and corrupt output after some minutes
#   5. keep the container alive until vllm exits
set -u

# --- 1. patches (each edits the container layer in place, same as start.sh) ---
python3 /opt/glm53/patch_glm_video_placeholders.py && \
python3 /opt/glm53/patch_suppress_stops_in_reasoning.py && \
python3 /opt/glm53/patch_scheduler_decode_floor.py && \
python3 /opt/glm53/patch_glm5_drafter_group.py && \
python3 /opt/glm53/patch_hybrid_prefix_hit.py && \
python3 /opt/glm53/patch_xgrammar_termination.py && \
python3 /opt/glm53/patch_kpool_tail_slotmap.py || {
    echo "glm53-serve: patch application failed" >&2
    exit 1
}

# --- 1b. ablit hook installer (only mounted by the ablit recipe) ---
# Idempotent and inert unless ABLIT=1 (hook no-ops on stock serves), so the
# stock recipe is unaffected. Same guard as start.sh.
if [ -f /opt/glm53/patch_ablit.py ]; then
    python3 /opt/glm53/patch_ablit.py || {
        echo "glm53-serve: patch_ablit failed" >&2
        exit 1
    }
fi

# --- 2. extract port + served name from args (for health check / warmup) ---
PORT=8000
SERVED="GLM-5.3-Flash-EXL3"
prev=""
for a in "$@"; do
    if [ "$prev" = "--port" ]; then PORT="$a"; fi
    if [ "$prev" = "--served-model-name" ]; then SERVED="$a"; fi
    prev="$a"
done

# --- 2b. preflight: the drafter must be a pinned local snapshot present on THIS
# node. A bare repo id resolves via each node's refs/main, which can diverge
# (2026-08-31 incident: TP ranks loaded different drafter checkpoints after an
# upstream re-upload -> 0% draft acceptance + corrupted output mid-generation).
prev=""; DRAFT_MODEL=""
for a in "$@"; do
    if [ "$prev" = "--speculative-config" ]; then
        DRAFT_MODEL=$(printf '%s' "$a" | sed -n 's/.*"model":"\([^"]*\)".*/\1/p')
    fi
    prev="$a"
done
if [ -n "$DRAFT_MODEL" ]; then
    case "$DRAFT_MODEL" in
        /*)
            # absolute snapshot path: must exist on this node
            if [ ! -f "$DRAFT_MODEL/config.json" ]; then
                echo "glm53-serve: FATAL - pinned drafter snapshot missing on this node: $DRAFT_MODEL" >&2
                exit 1
            fi
            echo "glm53-serve: drafter snapshot: $DRAFT_MODEL"
            ;;
        *)
            # repo id: resolved offline via refs/main - log which snapshot is
            # used so cross-rank divergence is visible in the logs
            for hub in "${HF_HOME:-/cache/huggingface}" /root/.cache/huggingface; do
                ref="$hub/hub/models--${DRAFT_MODEL//\//--}/refs/main"
                if [ -f "$ref" ]; then
                    sha=$(cat "$ref")
                    snap="$hub/hub/models--${DRAFT_MODEL//\//--}/snapshots/$sha"
                    if [ ! -f "$snap/config.json" ]; then
                        echo "glm53-serve: FATAL - drafter $DRAFT_MODEL refs/main=$sha but snapshot missing: $snap" >&2
                        exit 1
                    fi
                    echo "glm53-serve: drafter $DRAFT_MODEL resolves via refs/main=$sha"
                    break
                fi
            done
            ;;
    esac
fi

# --- 2c. ablit preflight: fail fast if ABLIT=1 but artifacts are missing on
# THIS node (TP ranks must apply the identical edit; see sync rule) ---
if [ "${ABLIT:-0}" = "1" ]; then
    if [ ! -f /opt/glm53/ablit/LAYER_MAP.json ]; then
        echo "glm53-serve: FATAL - ABLIT=1 but /opt/glm53/ablit/LAYER_MAP.json missing on this node" >&2
        exit 1
    fi
    echo "glm53-serve: ablit ON method=${ABLIT_METHOD:-auto} direction=${ABLIT_DIRECTION:-dealign} layers=${ABLIT_LAYERS:-15-45} alpha=${ABLIT_ALPHA:-3.0} mtp=${ABLIT_INCLUDE_MTP:-1}"
fi

# --- 3. launch vllm serve (all args incl. sparkrun-injected distributed flags) ---
vllm serve "$@" &
VLLM_PID=$!

# --- 4. wait for /health (weight load + graph capture on a 320B MoE is slow) ---
i=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 720 ]; then
        echo "glm53-serve: health wait timed out after 60m" >&2
        break
    fi
    sleep 5
done

# --- 5. boot-shape warmup (nonfatal) ---
# On the headless worker this reaches the head API over the host network and
# warms this rank's Triton cache as well.
GLM53_WARMUP_MAX_CONCURRENCY=4 \
GLM53_WARMUP_DFLASH_K=7 \
GLM53_WARMUP_TRITON_CACHE_DIR=/root/.triton/cache \
    bash /opt/glm53/boot-shape-warmup.sh "http://127.0.0.1:${PORT}" "$SERVED" \
    || echo "glm53-serve: boot-shape-warmup incomplete - uncovered shapes may JIT mid-serve" >&2

# --- 6. stay alive ---
wait "$VLLM_PID"
