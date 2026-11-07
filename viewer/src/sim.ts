// D2Q9 lattice Boltzmann in WebGL2 fragment shaders: the same scheme as the C++ solver (pull
// streaming, halfway bounce-back, TRT collision), one draw call per time step.
//
// The nine populations live in three RGBA32F textures (f0..f3, f4..f7, f8) rendered with three
// colour attachments, ping-ponged between two sets. A fourth pass writes velocity (RG32F) for
// display and for the tracer particles, whose positions are another ping-ponged texture.

import { W } from "./lbm";

const VS = `#version 300 es
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }`;

const LATTICE = `
const ivec2 C[9] = ivec2[9](ivec2(0,0), ivec2(1,0), ivec2(0,1), ivec2(-1,0), ivec2(0,-1),
                            ivec2(1,1), ivec2(-1,1), ivec2(-1,-1), ivec2(1,-1));
const int OPP[9] = int[9](0, 3, 4, 1, 2, 7, 8, 5, 6);
const float W[9] = float[9](4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);
uniform sampler2D F0, F1, F2;
float pop(ivec2 p, int i) {
  if (i < 4) return texelFetch(F0, p, 0)[i];
  if (i < 8) return texelFetch(F1, p, 0)[i - 4];
  return texelFetch(F2, p, 0).x;
}
float feq(int i, float rho, vec2 u) {
  float cu = dot(vec2(C[i]), u);
  return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}`;

const STEP = `#version 300 es
precision highp float;
precision highp int;
${LATTICE}
uniform sampler2D SOLID;
uniform ivec2 N;
uniform float OMEGA_P, OMEGA_M, U_IN;
uniform int PERIODIC_Y;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
bool solid(ivec2 q) {
  if (q.y < 0 || q.y >= N.y) return PERIODIC_Y == 0;
  return texelFetch(SOLID, q, 0).r > 0.5;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float f[9];
  if (texelFetch(SOLID, p, 0).r > 0.5) {
    for (int i = 0; i < 9; i++) f[i] = W[i];
  } else if (p.x == 0) {
    // Inflow: the equilibrium at the inlet speed and the density of the next cell in.
    float rho = 0.0;
    for (int i = 0; i < 9; i++) rho += pop(p + ivec2(1, 0), i);
    for (int i = 0; i < 9; i++) f[i] = feq(i, rho, vec2(U_IN, 0.0));
  } else if (p.x == N.x - 1) {
    // Outflow: zero gradient, the populations of the cell upstream.
    for (int i = 0; i < 9; i++) f[i] = pop(p - ivec2(1, 0), i);
  } else {
    for (int i = 0; i < 9; i++) {
      ivec2 q = p - C[i];
      if (PERIODIC_Y == 1) q.y = (q.y + N.y) % N.y;
      f[i] = solid(q) ? pop(p, OPP[i]) : pop(q, i);
    }
    float rho = 0.0;
    vec2 m = vec2(0.0);
    for (int i = 0; i < 9; i++) { rho += f[i]; m += f[i] * vec2(C[i]); }
    vec2 u = m / rho;
    float post[9];
    for (int i = 0; i < 9; i++) {
      int j = OPP[i];
      float ei = feq(i, rho, u), ej = feq(j, rho, u);
      post[i] = f[i] - OMEGA_P * (0.5 * (f[i] + f[j]) - 0.5 * (ei + ej)) - OMEGA_M * (0.5 * (f[i] - f[j]) - 0.5 * (ei - ej));
    }
    for (int i = 0; i < 9; i++) f[i] = post[i];
  }
  o0 = vec4(f[0], f[1], f[2], f[3]);
  o1 = vec4(f[4], f[5], f[6], f[7]);
  o2 = vec4(f[8], 0.0, 0.0, 0.0);
}`;

const MACRO = `#version 300 es
precision highp float;
${LATTICE}
uniform sampler2D SOLID;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(SOLID, p, 0).r > 0.5) { o = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float rho = 0.0; vec2 m = vec2(0.0);
  for (int i = 0; i < 9; i++) { float v = pop(p, i); rho += v; m += v * vec2(C[i]); }
  o = vec4(m / rho, rho, 0.0);
}`;

const SHOW = `#version 300 es
precision highp float;
uniform sampler2D U;
uniform sampler2D SOLID;
uniform vec2 VIEW;      // canvas size
uniform ivec2 N;
uniform int MODE;       // 0 speed, 1 vorticity
uniform float SCALE;    // colour range
out vec4 o;
vec3 viridis(float x) {
  x = clamp(x, 0.0, 1.0);
  float r = 0.2777 + x * (0.105 + x * (-0.3308 + x * (-4.6342 + x * (6.2283 + x * (4.7764 - x * 5.4355)))));
  float g = 0.0054 + x * (1.4046 + x * (0.2148 + x * (-5.7991 + x * (14.1799 + x * (-13.7451 + x * 4.6459)))));
  float b = 0.334 + x * (1.3845 + x * (0.095 + x * (-19.3324 + x * (56.6906 + x * (-65.353 + x * 26.3124)))));
  return clamp(vec3(r, g, b), 0.0, 1.0);
}
vec2 vel(ivec2 q) { return texelFetch(U, clamp(q, ivec2(0), N - 1), 0).xy; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy / VIEW * vec2(N));
  if (texelFetch(SOLID, p, 0).r > 0.5) { o = vec4(0.12, 0.12, 0.14, 1.0); return; }
  if (MODE == 0) {
    o = vec4(viridis(length(vel(p)) / SCALE), 1.0);
  } else {
    float w = 0.5 * ((vel(p + ivec2(1, 0)).y - vel(p - ivec2(1, 0)).y) - (vel(p + ivec2(0, 1)).x - vel(p - ivec2(0, 1)).x));
    float t = clamp(w / SCALE, -1.0, 1.0);
    vec3 c = t > 0.0 ? mix(vec3(0.97), vec3(0.75, 0.1, 0.12), t) : mix(vec3(0.97), vec3(0.1, 0.3, 0.75), -t);
    o = vec4(c, 1.0);
  }
}`;

const PARTICLE_STEP = `#version 300 es
precision highp float;
uniform sampler2D P;
uniform sampler2D U;
uniform sampler2D SOLID;
uniform ivec2 N;
uniform float DT;
uniform float SEED;
out vec4 o;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + SEED) * 43758.5453); }
void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  vec4 s = texelFetch(P, id, 0);
  vec2 x = s.xy;
  // Midpoint rule through the velocity field (bilinear by hand: the texture is not filterable).
  vec2 u1 = texelFetch(U, clamp(ivec2(x), ivec2(0), N - 1), 0).xy;
  vec2 xm = x + 0.5 * DT * u1;
  vec2 u2 = texelFetch(U, clamp(ivec2(xm), ivec2(0), N - 1), 0).xy;
  x += DT * u2;
  float age = s.z + 1.0;
  bool out_ = x.x < 1.0 || x.x >= float(N.x) - 1.0 || x.y < 0.0 || x.y >= float(N.y) || age > s.w;
  if (!out_ && texelFetch(SOLID, ivec2(x), 0).r > 0.5) out_ = true;
  if (out_) {
    // Reborn at a random point, mostly near the inlet so streaks fill the whole channel.
    float r = hash(vec2(id) + x);
    x = vec2(r < 0.6 ? 2.0 + 6.0 * hash(vec2(id) * 1.3) : hash(vec2(id) * 2.1) * float(N.x), hash(vec2(id) * 3.7 + x.yx) * float(N.y));
    age = 0.0;
  }
  o = vec4(x, age, s.w);
}`;

const PARTICLE_VS = `#version 300 es
precision highp float;
uniform sampler2D P;
uniform ivec2 N;
uniform int SIDE;
out float alpha;
void main() {
  ivec2 id = ivec2(gl_VertexID % SIDE, gl_VertexID / SIDE);
  vec4 s = texelFetch(P, id, 0);
  gl_Position = vec4(s.xy / vec2(N) * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.6;
  alpha = clamp(min(s.z / 20.0, (s.w - s.z) / 40.0), 0.0, 0.85);
}`;

const PARTICLE_FS = `#version 300 es
precision highp float;
in float alpha;
uniform vec3 COLOR;
out vec4 o;
void main() { o = vec4(COLOR, alpha); }`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
  return p;
}

function texture(gl: WebGL2RenderingContext, w: number, h: number, internal: number, format: number, type: number, data: ArrayBufferView | null) {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
  return t;
}

export interface Settings {
  omegaPlus: number;
  omegaMinus: number;
  uIn: number;
  periodicY: boolean;
}

export class Simulation {
  readonly nx: number;
  readonly ny: number;
  steps = 0;
  private gl: WebGL2RenderingContext;
  private f: WebGLTexture[][] = [];
  private fbo: WebGLFramebuffer[] = [];
  private cur = 0;
  private solid: WebGLTexture;
  private vel: WebGLTexture;
  private velFbo: WebGLFramebuffer;
  private parts: WebGLTexture[] = [];
  private partFbo: WebGLFramebuffer[] = [];
  private partCur = 0;
  private readonly side = 128;
  private progs: Record<string, WebGLProgram>;
  private quad: WebGLVertexArrayObject;
  private mask: Uint8Array;

  constructor(gl: WebGL2RenderingContext, nx: number, ny: number) {
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("this browser cannot render to float textures (EXT_color_buffer_float)");
    this.gl = gl;
    this.nx = nx;
    this.ny = ny;
    this.progs = {
      step: compile(gl, VS, STEP),
      macro: compile(gl, VS, MACRO),
      show: compile(gl, VS, SHOW),
      pstep: compile(gl, VS, PARTICLE_STEP),
      pdraw: compile(gl, PARTICLE_VS, PARTICLE_FS),
    };
    this.quad = gl.createVertexArray()!;
    gl.bindVertexArray(this.quad);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.mask = new Uint8Array(nx * ny);
    this.solid = texture(gl, nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, this.mask);
    for (let k = 0; k < 2; k++) {
      const set = [0, 1, 2].map(() => texture(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT, null));
      this.f.push(set);
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      set.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
      this.fbo.push(fb);
    }
    this.vel = texture(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
    this.velFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.vel, 0);
    const seed = new Float32Array(this.side * this.side * 4);
    for (let i = 0; i < this.side * this.side; i++) {
      seed[4 * i] = Math.random() * nx;
      seed[4 * i + 1] = Math.random() * ny;
      seed[4 * i + 2] = Math.random() * 200;
      seed[4 * i + 3] = 200 + Math.random() * 400; // lifetime in frames
    }
    for (let k = 0; k < 2; k++) {
      const t = texture(gl, this.side, this.side, gl.RGBA32F, gl.RGBA, gl.FLOAT, seed);
      this.parts.push(t);
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      this.partFbo.push(fb);
    }
    this.reset(0.1);
  }

  /** Fluid at rest everywhere... except moving at the inflow speed, to start quickly. */
  reset(u: number) {
    const gl = this.gl;
    const n = this.nx * this.ny;
    const data = [new Float32Array(4 * n), new Float32Array(4 * n), new Float32Array(4 * n)];
    const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    for (let c = 0; c < n; c++) {
      const s = this.mask[c] ? 0 : u;
      for (let i = 0; i < 9; i++) {
        const cu = cx[i] * s;
        const v = W[i] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * s * s);
        data[i < 4 ? 0 : i < 8 ? 1 : 2][4 * c + (i < 8 ? i % 4 : 0)] = v;
      }
    }
    for (let k = 0; k < 2; k++)
      for (let j = 0; j < 3; j++) {
        gl.bindTexture(gl.TEXTURE_2D, this.f[k][j]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.nx, this.ny, gl.RGBA, gl.FLOAT, data[j]);
      }
    this.steps = 0;
  }

  setMask(mask: Uint8Array) {
    this.mask = mask;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.solid);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.nx, this.ny, gl.RED, gl.UNSIGNED_BYTE, mask.map((v) => (v ? 255 : 0)));
  }

  getMask(): Uint8Array {
    return this.mask;
  }

  private bindF(prog: WebGLProgram, set: WebGLTexture[], unit0 = 0) {
    const gl = this.gl;
    ["F0", "F1", "F2"].forEach((name, i) => {
      gl.activeTexture(gl.TEXTURE0 + unit0 + i);
      gl.bindTexture(gl.TEXTURE_2D, set[i]);
      gl.uniform1i(gl.getUniformLocation(prog, name), unit0 + i);
    });
  }

  private bindTex(prog: WebGLProgram, name: string, t: WebGLTexture, unit: number) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.uniform1i(gl.getUniformLocation(prog, name), unit);
  }

  step(n: number, s: Settings) {
    const gl = this.gl;
    const p = this.progs.step;
    gl.useProgram(p);
    gl.bindVertexArray(this.quad);
    gl.viewport(0, 0, this.nx, this.ny);
    gl.uniform2i(gl.getUniformLocation(p, "N"), this.nx, this.ny);
    gl.uniform1f(gl.getUniformLocation(p, "OMEGA_P"), s.omegaPlus);
    gl.uniform1f(gl.getUniformLocation(p, "OMEGA_M"), s.omegaMinus);
    gl.uniform1f(gl.getUniformLocation(p, "U_IN"), s.uIn);
    gl.uniform1i(gl.getUniformLocation(p, "PERIODIC_Y"), s.periodicY ? 1 : 0);
    this.bindTex(p, "SOLID", this.solid, 3);
    for (let k = 0; k < n; k++) {
      const next = 1 - this.cur;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[next]);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
      this.bindF(p, this.f[this.cur]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.cur = next;
      this.steps++;
    }
  }

  /** Velocity field into the velocity texture; then the particles move through it. */
  private updateVelocity() {
    const gl = this.gl;
    const p = this.progs.macro;
    gl.useProgram(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.nx, this.ny);
    this.bindF(p, this.f[this.cur]);
    this.bindTex(p, "SOLID", this.solid, 3);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private moveParticles(dt: number) {
    const gl = this.gl;
    const p = this.progs.pstep;
    gl.useProgram(p);
    const next = 1 - this.partCur;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.partFbo[next]);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.side, this.side);
    this.bindTex(p, "P", this.parts[this.partCur], 0);
    this.bindTex(p, "U", this.vel, 1);
    this.bindTex(p, "SOLID", this.solid, 2);
    gl.uniform2i(gl.getUniformLocation(p, "N"), this.nx, this.ny);
    gl.uniform1f(gl.getUniformLocation(p, "DT"), dt);
    gl.uniform1f(gl.getUniformLocation(p, "SEED"), Math.random() * 100);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.partCur = next;
  }

  draw(width: number, height: number, mode: "speed" | "vorticity", scale: number, particles: boolean, stepsPerFrame: number) {
    const gl = this.gl;
    this.updateVelocity();
    if (particles) this.moveParticles(stepsPerFrame);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    const p = this.progs.show;
    gl.useProgram(p);
    this.bindTex(p, "U", this.vel, 0);
    this.bindTex(p, "SOLID", this.solid, 1);
    gl.uniform2f(gl.getUniformLocation(p, "VIEW"), width, height);
    gl.uniform2i(gl.getUniformLocation(p, "N"), this.nx, this.ny);
    gl.uniform1i(gl.getUniformLocation(p, "MODE"), mode === "speed" ? 0 : 1);
    gl.uniform1f(gl.getUniformLocation(p, "SCALE"), scale);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (particles) {
      const q = this.progs.pdraw;
      gl.useProgram(q);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      this.bindTex(q, "P", this.parts[this.partCur], 0);
      gl.uniform2i(gl.getUniformLocation(q, "N"), this.nx, this.ny);
      gl.uniform1i(gl.getUniformLocation(q, "SIDE"), this.side);
      // White streaks over the dark speed map, dark ones over the pale vorticity map.
      const c = mode === "speed" ? [1, 1, 1] : [0.1, 0.1, 0.13];
      gl.uniform3f(gl.getUniformLocation(q, "COLOR"), c[0], c[1], c[2]);
      gl.drawArrays(gl.POINTS, 0, this.side * this.side);
      gl.disable(gl.BLEND);
    }
  }

  /** Velocity at one cell (for the readout under the cursor); also detects a blow-up. */
  probe(x: number, y: number): [number, number, number] {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velFbo);
    const out = new Float32Array(4);
    gl.readPixels(Math.min(this.nx - 1, Math.max(0, x)), Math.min(this.ny - 1, Math.max(0, y)), 1, 1, gl.RGBA, gl.FLOAT, out);
    return [out[0], out[1], out[2]];
  }
}
