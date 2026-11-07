"""Runs the validation cases and compares them with reference solutions.

    python -m eddy_validation.validate [--quick]

writes results/validation.json and figures in docs/.

* Lid-driven cavity, Re 100 / 400 / 1000: centre-line velocity profiles against Ghia, Ghia and
  Shin (1982), Tables I and II.
* Force-driven channel (Poiseuille): the analytical parabola; error and convergence order for
  BGK and TRT.
* Square duct, D3Q19: the analytical series solution.
* Flow past a cylinder, Schafer and Turek (1996) 2D-1 (Re 20, steady) and 2D-2 (Re 100,
  periodic): drag and lift coefficients and Strouhal number against the FeatFlow reference values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT, REF, Fields, ghia, order, run

OUT = ROOT / "out" / "validation"
DOCS = ROOT / "docs"


def cavity(res: list[tuple[int, int]]) -> dict:
    """Centre-line profiles of the cavity; errors relative to the lid speed."""
    gu, gv = ghia("u"), ghia("v")
    out = {}
    profiles = {}
    for re, n in res:
        s = run("cavity", OUT / f"cavity-{re}-{n}", n=n, re=re)
        f = Fields(OUT / f"cavity-{re}-{n}" / "fields.bin")
        ulid = s["ulid"]
        u = f.u[0, :, :, 0] / ulid
        v = f.u[0, :, :, 1] / ulid
        # Fluid cells are 1..n; walls halfway, so cell j sits at (j - 0.5) / n. The centre line
        # x = 0.5 falls between cells n/2 and n/2 + 1.
        mid = n // 2
        coords = (np.arange(1, n + 1) - 0.5) / n
        u_line = 0.5 * (u[1 : n + 1, mid] + u[1 : n + 1, mid + 1])
        v_line = 0.5 * (v[mid, 1 : n + 1] + v[mid + 1, 1 : n + 1])
        # Add the walls: u = 1 at the lid, 0 at the bottom; v = 0 at both sides.
        yc = np.concatenate([[0.0], coords, [1.0]])
        uc = np.concatenate([[0.0], u_line, [1.0]])
        vc = np.concatenate([[0.0], v_line, [0.0]])
        y_ref, u_ref = gu[re]
        x_ref, v_ref = gv[re]
        du = np.interp(y_ref, yc, uc) - u_ref
        dv = np.interp(x_ref, yc, vc) - v_ref
        inner_u = (y_ref > 0) & (y_ref < 1)
        inner_v = (x_ref > 0) & (x_ref < 1)
        out[f"re{re}_n{n}"] = {
            "re": re, "n": n, "steps": s["steps"], "seconds": round(s["seconds"], 2),
            "u_rms": float(np.sqrt(np.mean(du[inner_u] ** 2))), "u_max": float(np.abs(du[inner_u]).max()),
            "v_rms": float(np.sqrt(np.mean(dv[inner_v] ** 2))), "v_max": float(np.abs(dv[inner_v]).max()),
        }
        profiles[(re, n)] = (yc, uc, vc)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        colours = {100: "#1f77b4", 400: "#ff7f0e", 1000: "#2ca02c"}
        for (re, n), (yc, uc, vc) in profiles.items():
            ax[0].plot(uc, yc, color=colours[re], lw=1.5, label=f"Eddy Re {re} ({n}²)")
            ax[1].plot(yc, vc, color=colours[re], lw=1.5, label=f"Eddy Re {re} ({n}²)")
        for re in sorted({r for r, _ in profiles}):
            ax[0].plot(gu[re][1], gu[re][0], "o", color=colours[re], mfc="none", label=f"Ghia Re {re}")
            ax[1].plot(gv[re][0], gv[re][1], "o", color=colours[re], mfc="none", label=f"Ghia Re {re}")
        ax[0].set(xlabel="u / U_lid", ylabel="y", title="u on the vertical centre line")
        ax[1].set(xlabel="x", ylabel="v / U_lid", title="v on the horizontal centre line")
        for a in ax:
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(DOCS / "cavity.png", dpi=130)
        plt.close(fig)
    except ImportError:
        pass
    return out


def poiseuille(sizes: list[int]) -> dict:
    out = {}
    for scheme in ["TRT", "BGK"]:
        errs = []
        for n in sizes:
            s = run("poiseuille", OUT / f"poiseuille-{scheme}-{n}", n=n, tau=0.8, bgk=scheme == "BGK")
            f = Fields(OUT / f"poiseuille-{scheme}-{n}" / "fields.bin")
            yy = np.arange(1, n + 1) - 0.5
            exact = s["force"] * yy * (n - yy) / (2 * s["viscosity"])
            got = f.u[0, 1 : n + 1, 1, 0]
            errs.append(float(np.sqrt(np.sum((got - exact) ** 2) / np.sum(exact**2))))
        out[scheme] = {"n": sizes, "l2_error": errs, "order": order(errs, sizes)}
    return out


def duct_exact(n: int, force: float, nu: float, terms: int = 200) -> np.ndarray:
    """Square duct of side n (walls halfway between nodes), body force along x."""
    a = n / 2.0
    c = (np.arange(1, n + 1) - 0.5) - a  # cell centres, from the axis
    y, z = np.meshgrid(c, c, indexing="ij")
    u = np.zeros_like(y)
    for k in range(terms):
        i = 2 * k + 1
        u += (-1) ** k * (1 - np.cosh(i * np.pi * z / (2 * a)) / np.cosh(i * np.pi / 2)) * np.cos(i * np.pi * y / (2 * a)) / i**3
    return 16 * a * a * force / (nu * np.pi**3) * u


def duct(sizes: list[int]) -> dict:
    errs = []
    for n in sizes:
        s = run("duct", OUT / f"duct-{n}", n=n, tau=0.8)
        f = Fields(OUT / f"duct-{n}" / "fields.bin")
        got = f.u[1 : n + 1, 1 : n + 1, 1, 0].T  # (y, z)
        exact = duct_exact(n, s["force"], s["viscosity"])
        errs.append(float(np.sqrt(np.sum((got - exact) ** 2) / np.sum(exact**2))))
    return {"n": sizes, "l2_error": errs, "order": order(errs, sizes)}


def cylinder(diameters: list[int]) -> dict:
    ref = json.loads((REF / "schaefer_turek.json").read_text())
    out = {"reference": ref["source"]}
    series = {}
    for re in [20, 100]:
        rows = []
        for d in diameters:
            s = run("cylinder", OUT / f"cylinder-{re}-{d}", d=d, re=re)
            data = np.genfromtxt(OUT / f"cylinder-{re}-{d}" / "forces.csv", delimiter=",", names=True)
            t, cd, cl = data["time"], data["cd"], data["cl"]
            if re == 20:
                last = t > t[-1] * 0.9
                row = {"d": d, "cd": float(cd[last].mean()), "cl": float(cl[last].mean())}
                r = ref["2D-1"]
                row["cd_error"] = row["cd"] / r["cd"] - 1
                row["cl_error"] = row["cl"] / r["cl"] - 1
            else:
                # The last half of the run, once shedding is established.
                last = t > t[-1] * 0.5
                tt, l = t[last], cl[last] - cl[last].mean()
                ups = tt[1:][(l[:-1] < 0) & (l[1:] >= 0)]  # upward zero crossings
                period = float(np.mean(np.diff(ups))) if len(ups) > 2 else float("nan")
                row = {"d": d, "cd_max": float(cd[last].max()), "cl_max": float(cl[last].max()), "st": 1.0 / period, "periods": len(ups) - 1}
                r = ref["2D-2"]
                row["cd_max_error"] = row["cd_max"] / r["cd_max"] - 1
                row["cl_max_error"] = row["cl_max"] / r["cl_max"] - 1
                row["st_error"] = row["st"] / r["st"] - 1
                series[d] = (t, cd, cl)
            row["mlups"] = round(s["mlups"], 1)
            row["seconds"] = round(s["seconds"], 1)
            rows.append(row)
        out[f"re{re}"] = rows
    try:
        import matplotlib.pyplot as plt

        d = max(series)
        t, cd, cl = series[d]
        keep = t > t[-1] - 10
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t[keep], cl[keep], lw=1.2, label="lift coefficient")
        ax.plot(t[keep], cd[keep], lw=1.2, label="drag coefficient")
        ax.axhline(ref["2D-2"]["cl_max"], color="gray", ls=":", lw=1)
        ax.axhline(ref["2D-2"]["cd_max"], color="gray", ls=":", lw=1, label="reference maxima")
        ax.set(xlabel="time (D / U)", title=f"Cylinder at Re 100 (D = {d} cells): the last 10 time units")
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DOCS / "cylinder.png", dpi=130)
        plt.close(fig)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small resolutions only (for CI)")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    results = {}
    if args.quick:
        results["poiseuille"] = poiseuille([8, 16])
        results["cavity"] = cavity([(100, 64)])
    else:
        results["cavity"] = cavity([(100, 128), (400, 128), (1000, 128), (1000, 256)])
        results["poiseuille"] = poiseuille([8, 16, 32, 64])
        results["duct"] = duct([8, 16, 32])
        results["cylinder"] = cylinder([20, 40, 80])
    path = ROOT / "results" / ("validation-quick.json" if args.quick else "validation.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
