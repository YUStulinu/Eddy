"""Running the solver and reading what it writes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "validation" / "reference"


def binary() -> Path:
    for p in [ROOT / "build" / "Release" / "eddy.exe", ROOT / "build" / "eddy", ROOT / "build" / "eddy.exe"]:
        if p.exists():
            return p
    raise FileNotFoundError("build the solver first (cmake --build build --config Release)")


def run(case: str, out: Path, **opts) -> dict:
    """Runs `eddy case` with --key value options; returns its summary.json."""
    out.mkdir(parents=True, exist_ok=True)
    args = [str(binary()), case, "--out", str(out)]
    for k, v in opts.items():
        k = k.replace("_", "-")
        if v is True:
            args.append(f"--{k}")
        elif v is not False and v is not None:
            args += [f"--{k}", str(v)]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{p.stderr}")
    print("  " + p.stdout.strip(), flush=True)
    summary = out / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else {}


class Fields:
    """fields.bin: header (nx, ny, nz, D), flags, rho, u."""

    def __init__(self, path: Path):
        b = path.read_bytes()
        nx, ny, nz, d = np.frombuffer(b[:16], np.int32)
        n = int(nx) * int(ny) * int(nz)
        self.nx, self.ny, self.nz, self.d = int(nx), int(ny), int(nz), int(d)
        o = 16
        self.flags = np.frombuffer(b[o : o + n], np.uint8).reshape(nz, ny, nx)
        o += n
        self.rho = np.frombuffer(b[o : o + 4 * n], np.float32).reshape(nz, ny, nx)
        o += 4 * n
        self.u = np.frombuffer(b[o : o + 4 * n * d], np.float32).reshape(nz, ny, nx, d)


def ghia(which: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Ghia et al. (1982): {Re: (coordinate, velocity)} for Re 100, 400, 1000."""
    rows = [list(map(float, l.split())) for l in (REF / f"ghia_{which}.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    a = np.array(rows)
    out = {}
    for col, re in [(1, 100), (2, 400), (3, 1000)]:
        coord, vel = a[:, 0], a[:, col]
        if which == "v" and re == 400:
            keep = ~np.isclose(coord, 0.9063)  # flagged as probably wrong in the transcription
            coord, vel = coord[keep], vel[keep]
        out[re] = (coord, vel)
    return out


def order(errors: list[float], sizes: list[int]) -> list[float]:
    """Observed convergence order between successive resolutions."""
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(sizes[i + 1] / sizes[i])) for i in range(len(errors) - 1)]
