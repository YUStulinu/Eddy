"""Speed: million lattice updates per second (MLUPS), GPU against CPU.

    python -m eddy_validation.bench

Fully periodic boxes of D2Q9 (256^2 .. 4096^2) and D3Q19 (32^3 .. 256^3), the same cell code on
the GPU and on the CPU, one thread and all threads (OpenMP). Also reported: the memory traffic
that speed implies -- each update reads and writes every population, 2 x Q x 4 bytes -- against
the GPU's memory bandwidth, the bound for this kind of code.
"""
from __future__ import annotations

import json
import os
import subprocess

import matplotlib.pyplot as plt

from .common import ROOT, binary

DOCS = ROOT / "docs"


def one(lattice: str, n: int, cpu: bool, threads: int | None = None, steps: int | None = None) -> dict:
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    cells = n * n if lattice == "d2q9" else n**3
    if steps is None:
        # About 2e9 updates on the GPU, 2e8 on the CPU: long enough to time, short enough to wait for.
        steps = max(20, int((2e8 if cpu else 2e9) / cells))
    args = [str(binary()), "bench", "--lattice", lattice, "--n", str(n), "--steps", str(steps)] + (["--cpu"] if cpu else [])
    p = subprocess.run(args, capture_output=True, text=True, env=env, check=True)
    r = json.loads(p.stdout)
    r["threads"] = threads if cpu else None
    print(r, flush=True)
    return r


def main():
    info = json.loads(subprocess.run([str(binary()), "info"], capture_output=True, text=True).stdout)
    threads = os.cpu_count()
    runs = []
    for lattice, sizes in [("d2q9", [256, 512, 1024, 2048, 4096]), ("d3q19", [32, 64, 128, 192, 256])]:
        for n in sizes:
            runs.append(one(lattice, n, cpu=False))
        for n in sizes[1:3]:
            runs.append(one(lattice, n, cpu=True, threads=1))
            runs.append(one(lattice, n, cpu=True, threads=threads))
    out = {"gpu": info["gpu"], "cpu_threads": threads, "runs": runs}
    (ROOT / "results" / "bench.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, lattice in enumerate(["d2q9", "d3q19"]):
        g = [r for r in runs if r["lattice"] == lattice and r["backend"] == "gpu"]
        ax[k].plot([r["cells"] for r in g], [r["mlups"] for r in g], "o-", label=f"GPU ({info['gpu']})")
        for t, style in [(1, "s--"), (threads, "^--")]:
            c = [r for r in runs if r["lattice"] == lattice and r["backend"] == "cpu" and r["threads"] == t]
            ax[k].plot([r["cells"] for r in c], [r["mlups"] for r in c], style, label=f"CPU, {t} thread{'s' if t > 1 else ''}")
        ax[k].set(xscale="log", yscale="log", xlabel="cells", ylabel="MLUPS", title=lattice.upper())
        ax[k].grid(alpha=0.3, which="both")
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "speed.png", dpi=130)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
