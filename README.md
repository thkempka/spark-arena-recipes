# spark-arena-recipes

Self-contained **sparkrun** recipes for LLM inference on NVIDIA DGX Spark (GB10),
prepared for Spark Arena submission and for anyone with a clean checkout of
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker).

**Goal:** clone this repo (+ `spark-vllm-docker`), run `./setup.sh`, and the
recipes are runnable. Everything (images, models, overlays) is either
digest-pinned and pulled from a public registry, or built offline from
sha256-verified, vendored sources. No files outside this repo are required —
with one documented exception: the GLM-5.3-Flash EXL3 recipe additionally
needs the MiaAI-Lab overlay checkout and warm JIT caches on **both** cluster
nodes (see its section below).

---

## Recipes

| Recipe | Model | Runtime | Topology | Port |
|---|---|---|---|---|
| [`qwen38-dflash2-sglang-a.yaml`](qwen38-dflash2-sglang-a.yaml) | Qwen3.8-27B NVFP4 | SGLang + DFlash2 (k=8) | 1× DGX Spark, tp=1 | 8000 |
| [`qwen38-dflash2-sglang-parallel-a.yaml`](qwen38-dflash2-sglang-parallel-a.yaml) | Qwen3.8-27B NVFP4 | SGLang + DFlash2 (k=8) | 2× DGX Spark, tp=2 | 8000 |
| [`deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml`](deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml) | DeepSeek-V4-Flash-0731 | vLLM + DSpark (k=5) | 2× DGX Spark, tp=2, 1M ctx | 8000 |
| [`qwen38-flash-next-nvfp4-sglang-tp2.yaml`](qwen38-flash-next-nvfp4-sglang-tp2.yaml) | Qwen3.8-Flash-Next-NVFP4 | SGLang + NEXTN/MTP4 — **CANONICAL, VERIFIED STABLE**: CUDA graphs off + radix/context cache on (~55 peak / ~33 typ, 94% cache hit) | 2× DGX Spark, tp=2 | 8000 |
| [`deepseek-v4-flash-0731-b12x-dspark-vllm-patched.yaml`](deepseek-v4-flash-0731-b12x-dspark-vllm-patched.yaml) | DeepSeek-V4-Flash-0731 | vLLM B12X + DSpark | 2× DGX Spark, tp=2 | 8000 |
| [`glm53-flash-exl3-vllm-tp2-dflash2.yaml`](glm53-flash-exl3-vllm-tp2-dflash2.yaml) | GLM-5.3-Flash EXL3 4bpw (320B MoE) | vLLM + DFlash2 (k=7, draft tp=2), CUDA graphs, 1M ctx | 2× DGX Spark, tp=2 | 8000 |
| [`…-tp2-graphs-experimental.yaml`](qwen38-flash-next-nvfp4-sglang-tp2-graphs-experimental.yaml) | Qwen3.8-Flash-Next-NVFP4 | ⚠️ *experimental, NOT the submission default* — CUDA graphs ON + radix OFF, ~70 peak but NaN-asserts under load | 2× DGX Spark, tp=2 | 8000 |

All container images are **digest-pinned** (immutable, reproducible). The two
registry images are pulled as-is; the qwen38 image is built offline from a
pinned public base + a vendored, sha256-verified DFlash2 overlay.

---

## Quick start

```bash
# prerequisites
git clone https://github.com/eugr/spark-vllm-docker   # upstream build/laugh scripts
git clone <this-repo> && cd <this-repo>
pipx install sparkrun==0.3.5    # pin the tool version (recipes tuned on 0.3.5)

# provision (pull/build images, validate env; idempotent)
QWEN38_API_KEY="$(cat my-secret)" ./setup.sh

# run
QWEN38_API_KEY="$QWEN38_API_KEY" sparkrun run qwen38-dflash2-sglang-a.yaml --solo
sparkrun run deepseek-v4-flash-0731-b12x-dspark-vllm-patched.yaml --cluster <your-2node-cluster>
sparkrun run deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml --cluster <your-2node-cluster>
```

`./setup.sh --check` reports what is present/missing without changing anything.

---

## Image provenance (pinned)

| Image | Source | Pinned as |
|---|---|---|
| `qwen38-dflash2:v1.2.2` | built offline from `docker/qwen38-dflash2/` | base `lmsysorg/sglang@sha256:febfb971…` + 5 verified files |
| DS b12x | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x` | `@sha256:af9629c5…` |
| DS nvfp4-1m | `ghcr.io/bjk110/vllm-spark` | `@sha256:d8492e76…` |

---

## Credits & references

### Base framework / tooling
- **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — the Docker config, launcher and build scripts (vLLM on DGX Spark, single/multi-node). Every recipe here runs on top of it. Author: eugr.
- **[sparkrun](https://pypi.org/project/sparkrun/)** — the recipe/workload manager (CLI) used to launch all recipes. Pin `sparkrun==0.3.5` for parity.

### Qwen3.8-27B NVFP4 + DFlash2 (SGLang)
- **Model:** `RadixArk/Qwen3.8-27B-NVFP4` (HF). Qwen3.8-27B by Alibaba Qwen team; NVFP4-quantized weights by RadixArk.
- **Drafter:** `z-lab/Qwen3.8-27B-DFlash2`, revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (HF).
- **DFlash2 code:** merged upstream in **[sgl-project/sglang PR #35371](https://github.com/sgl-project/sglang/pull/35371)** ("DFlash2: local convolution + candidate selector", 2026-08-19, commit `c14312a66420b75c`), Apache-2.0. The vendored overlay + manifest live in `docker/qwen38-dflash2/`; see [`ATTRIBUTION.md`](docker/qwen38-dflash2/ATTRIBUTION.md) for full provenance of each of the 5 files.
- **Recipe concept & image build:** **[hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)** — the original qwen38 SGLang + DFlash2 systemd deployment this recipe is ported from (base image, overlay, chat template, api-key handling).
- **Additional upstream contributors referenced in ATTRIBUTION.md:** [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark), [r0b0tlab/qwen38-27b-nvfp4-sm121-sglang](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang).

### Qwen3.8-Flash-Next-NVFP4 (SGLang)
- **Model:** `RadixArk/Qwen3.8-Flash-Next-NVFP4` (HF). ~125B-A3B hybrid MoE + 51B n-gram PLE embedding + in-checkpoint 4B MTP head, arch `qwen4_exp`.
- **Recipe concept & kernel work:** [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) - SM121 Triton QSA fallback, CUDA-graph-safe, memory tuning.
- **Locked config & stability:** [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) and [tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy) - NEXTN/MTP4 flags, GB10 UMA OOM pin (mem 0.80 + 600K-KV), agent-safe defaults, day-0 fixes.
- **Day-0 agent loop fix:** [sgl-project/sglang#36537](https://github.com/sgl-project/sglang/issues/36537) - thinking-off default + qwen3_coder parser, temp <= 0.7.

### DeepSeek-V4-Flash-0731 (vLLM + DSpark)
- **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` (HF), by DeepSeek.
- **nvfp4-1m recipe:** port of **[tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)**, pinned to commit `d728faee9f5a8d5ebafe7bc44bca6c5d8d0d192f` (2026-07-31). The recipe reproduces the DSpark overlay + NVFP4 stage patches at container start (fetched from that repo at launch). Author: tonyd2wild, with built-in fixes documented in the recipe header (cold-prefill garble fix, DSpark draft shared-expert loader fix).
- **b12x recipe:** built on the `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x` image (spark-arena / eugr nightly build line) with the DSpark B12X backend. The `pre_exec` stop-in-reasoning detokenizer patch is embedded (base64) directly in the recipe.

### DSpark / speculative decoding (DeepSeek)
- The DSpark speculative-decode stack is developed by NVIDIA engineers / the spark-arena community on top of vLLM; see the upstream repos above and
  **[NVIDIA DGX Spark forum — DeepSeek V4 Flash threads](https://forums.developer.nvidia.com/)** for tuning notes (RoCE/NCCL, draft acceptance, KV-cache dtype).

---

## GLM-5.3-Flash EXL3 4bpw — 2× DGX Spark, TP2 (`glm53-flash-exl3-vllm-tp2-dflash2.yaml`, rev 4)

Serves `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (320B MoE, EXL3 4bpw) with
DFlash2 k=7 speculative decoding (draft tp=2), CUDA graphs, 1M context,
prefix caching, vision on. Reproduces the MiaAI-Lab `start.sh` reference run
(~49 tok/s peak) as a sparkrun recipe. Status: **validated against the
miaai-pure baseline; long-generation stability incident root-caused and
fixed in rev 4.**

### Rev 4 incident summary (why the sync rule exists)

Two independent failure modes, both caused by **state drifting between the
two cluster nodes**:

1. **Mixed drafter checkpoints.** The DFlash2 drafter was passed as a bare
   repo id; each node resolved it via its *local* `refs/main` after a silent
   upstream re-upload of `incoai/GLM-5.3-Flash-DFlash2` (ws01 had the new
   `bf582e4e`, ws02 the old `7d74cdd8`). The TP-sharded drafter therefore
   mixed two different weight sets: draft acceptance collapsed to 0 after
   ~1k generated tokens and output degenerated into token loops.
   **Fix:** `distribution_config` pins the drafter `revision:` to the
   validated commit `dc77ff1c` (a sha revision never advances `refs/main`);
   `refs/main` is pinned on both nodes and the cache is pruned to that single
   snapshot (weights md5 `065aa1a4…`).
2. **Trailing newline in `refs/main`.** huggingface_hub 1.28 reads the ref
   file with `f.read()` and **no `.strip()`** — a newline written by
   `echo` makes offline repo-id resolution fail (`Invalid repository ID`).
   Always write refs with `printf '%s' <sha>`, never `echo`.

### Sync rule (MANDATORY for tp=2 across nodes)

Every bind-mounted host file in `executor_config.volumes` is resolved
**per node** — a stale copy on the worker silently boots a different
configuration. Before every launch, verify md5 identity across ws01/ws02:

```bash
# overlay patches + warmup script (9 files)
for f in patch_glm_video_placeholders.py patch_suppress_stops_in_reasoning.py \
         patch_scheduler_decode_floor.py patch_glm5_drafter_group.py \
         patch_hybrid_prefix_hit.py patch_xgrammar_termination.py \
         patch_kpool_tail_slotmap.py; do
  md5sum ~/glm53-miaai-pure/overlay/$f
  ssh tk@ws02 md5sum ~/glm53-miaai-pure/overlay/$f
done
md5sum ~/glm53-miaai-pure/scripts/boot-shape-warmup.sh \
       ~/spark-arena-recipes/scripts/glm53-serve.sh
ssh tk@ws02 md5sum ~/glm53-miaai-pure/scripts/boot-shape-warmup.sh \
                   ~/spark-arena-recipes/scripts/glm53-serve.sh
```

`scripts/glm53-serve.sh` (versioned **in this repo**) additionally fails fast
at boot if the pinned drafter snapshot is missing on its node, and logs the
`refs/main` sha it resolved — watch for it diverging between head and worker
in the logs.

### External dependencies (documented exception to self-containment)

| Host path (both nodes) | Origin | Regenerable? |
|---|---|---|
| `~/glm53-miaai-pure/overlay/patch_*.py` (7 files) | MiaAI-Lab repo `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` @ `688b7ab` (tracked) | yes — fresh clone |
| `~/glm53-miaai-pure/scripts/boot-shape-warmup.sh` | same repo @ `a099743` (tracked) | yes — fresh clone |
| `~/spark-arena-recipes/scripts/glm53-serve.sh` | **this repo** (versioned) | yes — this checkout |
| `~/.cache/vllm-glm53-flash/{triton,tilelang}` + root | machine-generated JIT caches from the validated reference run (~95 MB) | no — cold JIT mid-serve on TP=2 has corrupted output; treat as required |

The HF weights (target `024db9f7…`, drafter `dc77ff1c…`) are distributed by
sparkrun itself and excluded from the table above.

---

## Verified: the canonical Qwen3.8-Flash-Next recipe is stable AND uses the context cache

**Status: VERIFIED WORKING — 2026-08-27, ws01+ws02 (live server, not a claim).**

The canonical `qwen38-flash-next-nvfp4-sglang-tp2.yaml` (CUDA graphs OFF via
`--disable-cuda-graph` + radix/context cache ON) was launched at 19:08 UTC and left running
under real agent traffic. This is **the submission default** for this model on 2× DGX Spark.

> **Naming note:** this bulletproof config was developed as `…-tp2-bulletproof.yaml` and then
> promoted in place to the canonical file name `qwen38-flash-next-nvfp4-sglang-tp2.yaml`
> before submission, so the normal-looking name is what gets launched. The superseded
> graphs-ON config is preserved as `…-tp2-graphs-experimental.yaml` (see below).

### Stability — no crash class, no errors

| Check | Result |
|---|---|
| `/health` | `200 OK` |
| Uptime at verification | ~25 min continuous under load (no restart) |
| `inf/nan` / `assert` / `Traceback` / `CUDA error` / `OOM` in full log | **0 occurrences** |
| CUDA graph backend (both phases) | `disabled` — the NaN-assert class is structurally gone |
| Same-day *graphs-on* recipe log (`...-tp2.log`) | `probability tensor contains inf/nan` assert + scheduler Traceback at 17:52:43, on a `cuda graph: True` decode batch (the crash class this variant removes) |

The graphs-on variant hit this assert and killed the scheduler; this one has not. Disabling
the graph-captured decode/sampling path removes the whole NaN-logits assert class.

*(The pre-existing recipe header refers to this as the "70-class" config's crash
signature — it is the same signature, reproduced once here in the retained log.)*

### Context cache — enabled and actually hitting

Startup log confirms the radix tree is live (not disabled):

```
disable_radix_cache=False ... uses_mamba_radix_cache=True
mamba_radix_cache_strategy='extra_buffer', radix_eviction_policy='lru'
Init Unified Radix Cache. Components: (FULL, MAMBA). Tree Core: UnifiedTreeCore
Tree cache initialized: impl=UnifiedRadixCache hybrid_ssm=True
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 600000
Mamba Cache is allocated. max_mamba_cache_size: 97
```

Runtime scheduler lines show real prefix reuse on every request, with `cuda graph: False`
confirming the intended path:

```
#new-seq: 1  #cached-token: 27776  full token usage: 0.05  cuda graph: False
```

`/metrics` (server-side counters, same process):

| Metric | Value | Meaning |
|---|---|---|
| `sglang:cache_hit_rate` | **0.944** | 94.4% of prompt tokens served from cache |
| `sglang:realtime_tokens_total{mode="prefill_cache"}` | **2.32 M** | cached prefill tokens actually skipped |
| `sglang:evicted_tokens_total{cache_type="UnifiedRadixCache"}` | **64** | near-zero eviction → pin is right-sized |
| `sglang:kv_cache_memory_usage_gb` | 7.3 GB | well inside the 0.80 mem-fraction budget |

### Timing proof of the cache effect

Repeated identical 6 420-token prompt (cold = first pass through the radix tree, then warm):

| Run | Wall time | Note |
|---|---|---|
| 1 (cold) | **2.25 s** | full prefill, tree empty |
| 2 (warm) | **0.25 s** | ~9× faster, prefix hit |
| 3 (warm) | **0.26 s** | stable |

**~9× speedup on repeat prompts** — this is the context cache paying off for agent loops.

### Why this is the right pin

- `--max-total-tokens 600000` + `--mem-fraction-static 0.80`: KV pin that does not OOM on
  GB10 unified memory, yet leaves enough headroom that eviction is effectively zero (64 tokens).
- `--max-mamba-cache-size 97`: explicit pin (the pool auto-sizes to 97 anyway), so the
  radix cache has stable mamba state slots.
- CUDA graphs OFF costs some peak tok/s vs. a hypothetical working graphs-on config, but
  it buys **correctness + determinism**: no NaN-assert death, and the cache stays on.
  Net win for an always-on endpoint.

### Reproduce the verification

```bash
# health + uptime
curl -s localhost:8000/health
docker ps --filter name=sparkrun_ --format '{{.Names}} {{.Status}}'

# cache is really on (expect disable_radix_cache=False, UnifiedRadixCache)
grep -oE "disable_radix_cache=[A-Za-z]+|Init Unified Radix Cache.*" \
  /tmp/sparkrun-panel/qwen38-flash-next-nvfp4-sglang-tp2-bulletproof.log  # log named pre-rename

# hit rate + cached/evicted tokens
curl -s localhost:8000/metrics | grep -E "cache_hit_rate|prefill_cache|evicted_tokens"

# cold vs warm single-prompt timing (repeat -> big drop)
curl -s localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"text":"<same long prompt>","sampling_params":{"max_new_tokens":8}}' -o /dev/null -w '%{time_total}\n'

# no crash class in the log (expect 0)
grep -icE "inf/nan|assert|Traceback|CUDA error|out of memory" \
  /tmp/sparkrun-panel/qwen38-flash-next-nvfp4-sglang-tp2-bulletproof.log  # log named pre-rename
```

> The verified launch predates the file rename, so its log is still
> `…-tp2-bulletproof.log`. A relaunch under the canonical name writes
> `…/qwen38-flash-next-nvfp4-sglang-tp2.log` — note that the graphs-ON experimental file
> would produce the *same* log name, so read that historical file as graphs-ON evidence.

### Why the graphs-ON config is NOT the default

`qwen38-flash-next-nvfp4-sglang-tp2-graphs-experimental.yaml` is kept only as a speed upper
bound (~70 peak tok/s). It buys that by turning CUDA graphs ON and the radix cache OFF, which
costs the context cache entirely and reintroduces the `probability tensor contains inf/nan`
scheduler assert on graph-captured decode batches. It is not suitable for an always-on
endpoint and is deliberately not the file an evaluator would launch.

**Operational rule unchanged:** keep agent `temperature <= 0.7`; keep thinking-off default
(`--default-chat-template-kwargs '{"enable_thinking": false}'`) + `--tool-call-parser
qwen3_coder` (sglang#36537); drop the page cache on both nodes before each launch.

## Notes

- **API key:** the qwen38 recipes require `QWEN38_API_KEY` at serve time (env var). It is never stored in the repo.
- **HF models:** DeepSeek-V4-Flash-0731 is ~167 GB and pulled from HF into the sparkrun HF cache (`/cache/huggingface` on Spark nodes); requires `HF_TOKEN` for gated repos if applicable.
- **Network at launch:** `deepseek-v4-flash-0731-dspark-nvfp4-1m-vllm.yaml` fetches its DSpark overlay tarball from GitHub at first container start (digest/commit-pinned). The other three recipes need no runtime fetch.
- **GPU exclusivity:** the qwen38 SGLang recipes need the GPU to themselves (stop any other job first). tp=2 variants must not overlap with other cluster jobs.
