import { describe, expect, it } from "vitest";
import { CX, CY, OPP, W, feq, maskFromImage, rates, shapeMask, viridis } from "./lbm";

describe("lattice", () => {
  it("is D2Q9: weights sum to one, opposites reverse, the equilibrium carries density and momentum", () => {
    expect(W.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    for (let i = 0; i < 9; i++) {
      expect(CX[OPP[i]] + CX[i]).toBe(0);
      expect(CY[OPP[i]] + CY[i]).toBe(0);
    }
    const rho = 1.02, ux = 0.05, uy = -0.03;
    let s = 0, mx = 0, my = 0;
    for (let i = 0; i < 9; i++) {
      const f = feq(i, rho, ux, uy);
      s += f;
      mx += f * CX[i];
      my += f * CY[i];
    }
    expect(s).toBeCloseTo(rho, 12);
    expect(mx).toBeCloseTo(rho * ux, 12);
    expect(my).toBeCloseTo(rho * uy, 12);
  });
});

describe("units", () => {
  it("turns a Reynolds number into relaxation rates (TRT, Lambda 3/16)", () => {
    const r = rates(100, 0.1, 40);
    expect(r.viscosity).toBeCloseTo(0.04, 12);
    expect(r.clamped).toBe(false);
    const tauP = 1 / r.omegaPlus, tauM = 1 / r.omegaMinus;
    expect((tauP - 0.5) * (tauM - 0.5)).toBeCloseTo(3 / 16, 12);
  });
  it("keeps the viscosity in the stable range", () => {
    const r = rates(1e6, 0.1, 40);
    expect(r.clamped).toBe(true);
    expect(r.viscosity).toBe(0.004);
  });
});

describe("shapes", () => {
  it("draws a cylinder of about the expected area", () => {
    const nx = 300, ny = 120;
    const m = shapeMask("cylinder", nx, ny);
    const d = Math.round(ny / 6);
    const area = m.reduce((a, b) => a + b, 0);
    expect(area).toBeGreaterThan(0.9 * Math.PI * (d / 2) ** 2);
    expect(area).toBeLessThan(1.1 * Math.PI * (d / 2) ** 2);
    expect(shapeMask("none", nx, ny).every((v) => v === 0)).toBe(true);
    expect(shapeMask("airfoil", nx, ny).some((v) => v === 1)).toBe(true);
  });
  it("reads dark pixels of an image as solid, flipping rows", () => {
    // 2x2 image, top-left black.
    const img = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
    const m = maskFromImage(img, 2, 2, 2, 2);
    expect(Array.from(m)).toEqual([0, 0, 1, 0]); // the bottom row first
  });
});

describe("colour", () => {
  it("runs from dark purple to yellow", () => {
    const [r0, , b0] = viridis(0);
    const [r1, g1, b1] = viridis(1);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(0.9);
    expect(g1).toBeGreaterThan(0.8);
    expect(b1).toBeLessThan(0.3);
  });
});
