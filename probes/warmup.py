#!/usr/bin/env python3
"""warmup.py — warm the GLM-5.3-Flash DFlash2 engine before benchmarking.

Cold engines JIT-compile _prepare_dflash_inputs_kernel, _topk_topp_kernel and
mhc_pre_big_fuse_with_norm_tilelang on the FIRST requests (latency spikes; a cold
C1 reads ~10 tok/s low — 36.9 vs 46.9 upstream on the same config). This script
fires a few code/reasoning requests at the same shape as probes/bench_c1c6.py
plus one long-context request (sparse-indexer path), so by the time you bench the
JIT table is warm.

Usage: python3 warmup.py --url http://<HEAD_IP>:8000 [--rounds 3] [--long-tokens 2048]
Watch TTFT drop run-over-run; when it plateaus, you're warm.
"""
import argparse, json, random, time, urllib.request

PROMPTS = [
    "Write a Python function that parses an nginx access log line into a dict, with a regex, and explain each group.",
    "Implement a rate limiter class in Python using the token bucket algorithm, then show example usage.",
    "Write a SQL query for the top 5 customers by 90-day revenue, then rewrite it as a window function version.",
    "Explain the difference between TCP slow start and congestion avoidance, then pseudocode both.",
]

def post(url, payload, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def one(url, prompt, max_tokens):
    salt = f"[warm {random.randint(1,10**9)}] "
    p = {"model": "glm-5.3-flash",
         "messages": [{"role": "user", "content": salt + prompt}],
         "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
    t0 = time.time()
    r = post(url, p)
    dt = time.time() - t0
    print(f"  warmup {max_tokens:>4}t prompt: {dt:6.2f}s wall, "
          f"ttft~{r['usage']['completion_tokens'] and dt/max(r['usage']['completion_tokens'],1):.3f}s/t "
          f"({r['usage']['completion_tokens']} ct)", flush=True)
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--long-tokens", type=int, default=2048)
    a = ap.parse_args()

    # health gate
    for _ in range(60):
        try:
            urllib.request.urlopen(a.url + "/health", timeout=5)
            print("health: 200")
            break
        except Exception:
            time.sleep(10)
    else:
        raise SystemExit("engine not healthy after 10 min — check sparkrun log")

    print("warming decode kernels (code/reasoning, temp 1.0)...")
    for i in range(a.rounds):
        one(a.url, PROMPTS[i % len(PROMPTS)], 400 + i * 100)

    print(f"warming long-context path ({a.long_tokens}-token prompt)...")
    long = ("Write a detailed Python implementation and walkthrough of a suffix array "
            "construction and longest-common-prefix computation. " * 40)[:a.long_tokens]
    one(a.url, long, 200)

    print("done. Check the sparkrun log for remaining 'JIT compilation during "
          "inference' warnings — when they stop, the engine is warm.")
    print("then: python3 probes/bench_c1c6.py --url " + a.url)

if __name__ == "__main__":
    main()
