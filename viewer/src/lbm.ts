// The parts of the browser simulation that are plain arithmetic: lattice constants, units,
// obstacle shapes and colour maps. The GPU code is in sim.ts.

/** D2Q9, in the same order as the C++ solver. */
export const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
export const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
export const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];
export const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

export function feq(i: number, rho: number, ux: number, uy: number): number {
  const cu = CX[i] * ux + CY[i] * uy;
  return W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy));
}

export interface Rates {
  viscosity: number;
  omegaPlus: number;
  omegaMinus: number;
  clamped: boolean;
}

/**
 * Relaxation rates for a Reynolds number Re = U L / nu, with TRT's magic parameter 3/16.
 * Very low viscosities are unstable at this resolution: nu is kept at or above 0.004 (and the
 * Reynolds number shown is then the one actually simulated).
 */
export function rates(re: number, u: number, length: number, magic = 3 / 16): Rates {
  let viscosity = (u * length) / re;
  const clamped = viscosity < 0.004;
  if (clamped) viscosity = 0.004;
  const tau = 3 * viscosity + 0.5;
  const tauMinus = magic / (tau - 0.5) + 0.5;
  return { viscosity, omegaPlus: 1 / tau, omegaMinus: 1 / tauMinus, clamped };
}

export type Shape = "cylinder" | "plate" | "airfoil" | "two-cylinders" | "none";

/** A solid mask (1 = solid), row-major from the bottom row, for a domain of nx x ny cells. */
export function shapeMask(shape: Shape, nx: number, ny: number): Uint8Array {
  const m = new Uint8Array(nx * ny);
  const d = Math.round(ny / 6);
  const cx = Math.round(nx / 5);
  const cy = ny / 2;
  const set = (x: number, y: number) => {
    if (x >= 0 && x < nx && y >= 0 && y < ny) m[y * nx + x] = 1;
  };
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const px = x + 0.5, py = y + 0.5;
      switch (shape) {
        case "cylinder":
          // Slightly off-centre, so the wake sheds without waiting for round-off.
          if ((px - cx) ** 2 + (py - cy - 0.5) ** 2 <= (d / 2) ** 2) set(x, y);
          break;
        case "two-cylinders":
          if ((px - cx) ** 2 + (py - cy - d) ** 2 <= (d / 2.4) ** 2 || (px - cx - 2 * d) ** 2 + (py - cy + d) ** 2 <= (d / 2.4) ** 2) set(x, y);
          break;
        case "plate":
          if (Math.abs(px - cx) <= 1.5 && Math.abs(py - cy) <= d) set(x, y);
          break;
        case "airfoil": {
          // NACA 0012 at 12 degrees: the thickness distribution around a tilted chord.
          const chord = 3 * d;
          const a = (-12 * Math.PI) / 180;
          const rx = (px - (cx - chord / 3)) * Math.cos(a) + (py - cy) * Math.sin(a);
          const ry = -(px - (cx - chord / 3)) * Math.sin(a) + (py - cy) * Math.cos(a);
          const t = rx / chord;
          if (t >= 0 && t <= 1) {
            const half = 5 * 0.12 * chord * (0.2969 * Math.sqrt(t) - 0.126 * t - 0.3516 * t * t + 0.2843 * t ** 3 - 0.1015 * t ** 4);
            if (Math.abs(ry) <= half) set(x, y);
          }
          break;
        }
        case "none":
          break;
      }
    }
  }
  return m;
}

/** A mask from an image's pixels (RGBA, top row first): dark pixels are solid. */
export function maskFromImage(rgba: Uint8ClampedArray, width: number, height: number, nx: number, ny: number): Uint8Array {
  const m = new Uint8Array(nx * ny);
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const ix = Math.min(width - 1, Math.floor(((x + 0.5) / nx) * width));
      const iy = Math.min(height - 1, Math.floor(((ny - 1 - y + 0.5) / ny) * height));
      const k = 4 * (iy * width + ix);
      const lum = 0.299 * rgba[k] + 0.587 * rgba[k + 1] + 0.114 * rgba[k + 2];
      if (rgba[k + 3] > 127 && lum < 128) m[y * nx + x] = 1;
    }
  }
  return m;
}

/** Perceptual colour for 0..1 (an approximation of viridis), as [r, g, b] in 0..1. */
export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const r = 0.2777 + x * (0.105 + x * (-0.3308 + x * (-4.6342 + x * (6.2283 + x * (4.7764 - x * 5.4355)))));
  const g = 0.0054 + x * (1.4046 + x * (0.2148 + x * (-5.7991 + x * (14.1799 + x * (-13.7451 + x * 4.6459)))));
  const b = 0.334 + x * (1.3845 + x * (0.095 + x * (-19.3324 + x * (56.6906 + x * (-65.353 + x * 26.3124)))));
  const c = (v: number) => Math.min(1, Math.max(0, v));
  return [c(r), c(g), c(b)];
}
