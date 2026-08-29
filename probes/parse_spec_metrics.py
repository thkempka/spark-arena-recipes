#!/usr/bin/env python3
"""parse_spec_metrics.py — summarize SpecDecoding acceptance from a sparkrun log.

Usage: python3 parse_spec_metrics.py /tmp/sparkrun-panel/glm53-...-dflash2.log

Prints: mean acceptance length, avg draft acceptance, accepted/drafted t/s,
per-position acceptance averages (how far past token 1 the drafter survives),
and engine generation throughput windows. Compare the per-position curve
against the pre-fix log ([0.570,0.291,0.093,0.035,...]) to see if the
indexer fix moved target<->drafter agreement.
"""
import re, sys, statistics
from collections import defaultdict

SPEC = re.compile(
    r"SpecDecoding metrics: Mean acceptance length: ([\d.]+), "
    r"Accepted throughput: ([\d.]+) tokens/s, Drafted throughput: ([\d.]+) tokens/s, "
    r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
    r"Per-position acceptance rate: ([0-9., ]+), Avg Draft acceptance rate: ([\d.]+)%"
)
ENGINE = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s")

def main(path: str) -> None:
    mean_len, accept_ratio, acc_tps, draft_tps = [], [], [], []
    per_pos = defaultdict(list)
    eng_gen = []
    n_windows = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = SPEC.search(line)
            if m:
                n_windows += 1
                ml, at, dt, acc, dra, pos, ar = m.groups()
                mean_len.append(float(ml))
                accept_ratio.append(float(ar))
                acc_tps.append(float(at)); draft_tps.append(float(dt))
                for i, p in enumerate(pos.split(",")):
                    per_pos[i].append(float(p.strip()))
            m = ENGINE.search(line)
            if m and float(m.group(1)) > 0:
                eng_gen.append(float(m.group(1)))

    if not mean_len:
        print("no SpecDecoding metric lines found — spec decode NOT active in this log"); return
    print(f"spec-decode metric windows : {n_windows}")
    print(f"mean acceptance length     : {statistics.mean(mean_len):.2f} (of 8 slots; upstream warm/code = ~5.9 @ 74.1%)")
    print(f"avg draft acceptance rate  : {statistics.mean(accept_ratio):.1f}%")
    print(f"accepted throughput        : {statistics.mean(acc_tps):.1f} t/s   (drafted {statistics.mean(draft_tps):.1f} t/s)")
    print(f"engine generation thruput  : {statistics.mean(eng_gen):.1f} t/s  (best window {max(eng_gen):.1f})" if eng_gen else "")
    print("per-position acceptance (0=first draft token):")
    for i in sorted(per_pos):
        avg = statistics.mean(per_pos[i])
        bar = "#" * int(avg * 40)
        print(f"  pos {i}: {avg:5.3f}  {bar}")
    # how much of the 7-token block survives
    p1 = statistics.mean(per_pos[0]) if 0 in per_pos else 0
    p2 = statistics.mean(per_pos[1]) if 1 in per_pos else 0
    p4 = statistics.mean(per_pos[3]) if 3 in per_pos else 0
    print(f"survival: after pos0 {p1:.2f} -> after pos1 {p2:.2f} -> after pos3 {p4:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
