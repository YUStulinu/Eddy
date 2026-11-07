import "./style.css";
import { maskFromImage, rates, shapeMask, type Shape } from "./lbm";
import { Simulation } from "./sim";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("canvas");
const NX = 640, NY = 240, U = 0.1;
const LENGTH = Math.round(NY / 6); // the obstacle's size, for the Reynolds number

const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
let sim: Simulation;
let paused = false;
let mlups = 0;

function fail(msg: string) {
  $("stage").innerHTML = `<p class="error">${msg}</p>`;
  throw new Error(msg);
}

if (!gl) fail("This browser has no WebGL2.");
try {
  sim = new Simulation(gl!, NX, NY);
} catch (e) {
  fail(String((e as Error).message));
}

function re(): number {
  return Math.round(10 ** Number($<HTMLInputElement>("re").value));
}

function settings() {
  const r = rates(re(), U, LENGTH);
  return { omegaPlus: r.omegaPlus, omegaMinus: r.omegaMinus, uIn: U, periodicY: !$<HTMLInputElement>("walls").checked, rates: r };
}

function load(shape: Shape) {
  sim.setMask(shapeMask(shape, NX, NY));
  sim.reset(U);
}

$<HTMLSelectElement>("shape").onchange = (e) => load((e.target as HTMLSelectElement).value as Shape);
$("reset").onclick = () => sim.reset(U);
$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Run" : "Pause";
};
$<HTMLInputElement>("image").onchange = async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, bmp.width, bmp.height);
  sim.setMask(maskFromImage(img.data, img.width, img.height, NX, NY));
  sim.reset(U);
};

// Drawing obstacles: left button adds solid, right button removes it.
let drawing = 0;
function paint(ev: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - r.left) / r.width) * NX);
  const y = Math.floor((1 - (ev.clientY - r.top) / r.height) * NY);
  const mask = sim.getMask().slice();
  const rad = 4;
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) {
      const px = x + dx, py = y + dy;
      if (dx * dx + dy * dy <= rad * rad && px > 1 && px < NX - 2 && py >= 0 && py < NY) mask[py * NX + px] = drawing === 1 ? 1 : 0;
    }
  sim.setMask(mask);
}
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  drawing = e.button === 2 ? 2 : 1;
  canvas.setPointerCapture(e.pointerId);
  paint(e);
});
canvas.addEventListener("pointermove", (e) => {
  if (drawing) paint(e);
});
canvas.addEventListener("pointerup", () => (drawing = 0));

function resize() {
  const stage = $("stage");
  const w = stage.clientWidth;
  const h = Math.round((w * NY) / NX);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
}
window.addEventListener("resize", resize);
resize();

let last = performance.now();
let stepsSince = 0;
let frames = 0;
function frame(now: number) {
  const s = settings();
  const spf = Number($<HTMLInputElement>("spf").value);
  $("spf-value").textContent = String(spf);
  $("re-value").textContent = s.rates.clamped ? `${re()} (limited to ${Math.round((U * LENGTH) / s.rates.viscosity)})` : String(re());
  if (!paused) {
    sim.step(spf, s);
    stepsSince += spf;
  }
  const mode = $<HTMLSelectElement>("mode").value as "speed" | "vorticity";
  sim.draw(canvas.width, canvas.height, mode, mode === "speed" ? 0.18 : 0.02, $<HTMLInputElement>("particles").checked, paused ? 0 : spf);
  frames++;
  if (now - last > 1000) {
    mlups = (stepsSince * NX * NY) / ((now - last) / 1000) / 1e6;
    // A blow-up shows as NaN; start again rather than show garbage.
    const [ux] = sim.probe(NX - 3, NY >> 1);
    if (!Number.isFinite(ux)) sim.reset(U);
    $("stats").innerHTML = "";
    const add = (k: string, v: string) => $("stats").append(Object.assign(document.createElement("dt"), { textContent: k }), Object.assign(document.createElement("dd"), { textContent: v }));
    add("grid", `${NX} × ${NY}`);
    add("speed", `${mlups.toFixed(0)} million cell updates/s`);
    add("frames", `${((frames * 1000) / (now - last)).toFixed(0)} per second`);
    add("time steps", sim.steps.toLocaleString());
    add("viscosity", s.rates.viscosity.toFixed(4));
    last = now;
    stepsSince = 0;
    frames = 0;
  }
  requestAnimationFrame(frame);
}
load("cylinder");
requestAnimationFrame(frame);
