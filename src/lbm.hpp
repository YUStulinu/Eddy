// The lattice Boltzmann method, one cell at a time. Everything here is __host__ __device__:
// the CPU and GPU solvers run exactly this code, so their results can be compared and their
// speeds measured on equal terms.
//
// Populations are stored structure-of-arrays (f[i * cells + cell]), two copies, streamed by
// "pull": a cell gathers the post-collision populations of its neighbours, then collides.
//
// What is stored is not f_i but f_i - w_i, the departure from the fluid at rest with density 1.
// In single precision that matters: f_i is about 0.1 while the part that carries the flow is
// 1e-3 to 1e-6 of it, and a body force of 1e-6 added to 0.1 every step keeps only one or two
// digits. Stored shifted, those digits survive. Density is 1 + the sum of the stored values.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EDDY_HD __host__ __device__ __forceinline__
#define EDDY_UNROLL _Pragma("unroll")
#else
#define EDDY_HD inline
#define EDDY_UNROLL
#endif

namespace eddy {

enum CellType : uint8_t {
    FLUID = 0,
    WALL = 1,      // no-slip, halfway bounce-back
    MOVING = 2,    // a wall moving with Params::wall_u (the lid of a cavity)
    OBSTACLE = 3,  // a wall whose force is measured
    INLET = 4,     // prescribed velocity
    OUTLET = 5,    // prescribed density (pressure)
};

EDDY_HD bool is_solid(uint8_t t) { return t == WALL || t == MOVING || t == OBSTACLE; }

// D2Q9: rest, 4 axis, 4 diagonal directions.
struct D2Q9 {
    static constexpr int D = 2, Q = 9;
    EDDY_HD static int cx(int i) { constexpr int v[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; return v[i]; }
    EDDY_HD static int cy(int i) { constexpr int v[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1}; return v[i]; }
    EDDY_HD static int cz(int) { return 0; }
    EDDY_HD static int opp(int i) { constexpr int v[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; return v[i]; }
    EDDY_HD static float w(int i) { return i == 0 ? 4.f / 9.f : (i < 5 ? 1.f / 9.f : 1.f / 36.f); }
};

// D3Q19: rest, 6 axis, 12 edge-diagonal directions.
struct D3Q19 {
    static constexpr int D = 3, Q = 19;
    EDDY_HD static int cx(int i) { constexpr int v[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0}; return v[i]; }
    EDDY_HD static int cy(int i) { constexpr int v[19] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1}; return v[i]; }
    EDDY_HD static int cz(int i) { constexpr int v[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1}; return v[i]; }
    EDDY_HD static int opp(int i) { constexpr int v[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17}; return v[i]; }
    EDDY_HD static float w(int i) { return i == 0 ? 1.f / 3.f : (i < 7 ? 1.f / 18.f : 1.f / 36.f); }
};

struct Params {
    int nx = 1, ny = 1, nz = 1;
    int periodic[3] = {0, 0, 0};
    float omega_plus = 1.f;   // even relaxation rate (1/tau): sets the viscosity
    float omega_minus = 1.f;  // odd rate; equal to omega_plus for BGK, from the magic parameter for TRT
    float force[3] = {0, 0, 0};   // body force per unit volume (Guo's scheme)
    float wall_u[3] = {0, 0, 0};  // velocity of MOVING walls
    float outlet_rho = 1.f;
    float inlet_scale = 1.f;  // multiplies the inlet profile (to ramp a flow up gently)
    const float* inlet_u = nullptr;  // INLET velocities, 3 per (y, z) row of the x = const face
};

// The equilibrium, shifted like the stored populations: feq - w, written with drho = rho - 1 so
// that nothing large is subtracted.
EDDY_HD float feq(float w, float drho, float cu, float uu) {
    return w * (drho + (1.f + drho) * (3.f * cu + 4.5f * cu * cu - 1.5f * uu));
}

EDDY_HD int wrap(int v, int n, bool periodic) {
    if (periodic) return (v + n) % n;
    return v;
}

// Moments of a cell's (shifted) populations: density - 1 and velocity (with half the force, as Guo's scheme
// requires for second-order accuracy).
template <class L>
EDDY_HD void moments(const float* f, const Params& p, float& drho, float u[3]) {
    drho = 0.f;
    float m[3] = {0.f, 0.f, 0.f};
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) {
        drho += f[i];
        m[0] += f[i] * L::cx(i);
        m[1] += f[i] * L::cy(i);
        m[2] += f[i] * L::cz(i);
    }
    for (int d = 0; d < 3; d++) u[d] = (m[d] + 0.5f * p.force[d]) / (1.f + drho);
}

// The same moments from post-collision populations (what the solver stores between steps): their
// momentum already includes this step's force F, so the velocity is (sum f c - F/2) / rho.
template <class L>
EDDY_HD void moments_post(const float* f, const Params& p, float& drho, float u[3]) {
    drho = 0.f;
    float m[3] = {0.f, 0.f, 0.f};
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) {
        drho += f[i];
        m[0] += f[i] * L::cx(i);
        m[1] += f[i] * L::cy(i);
        m[2] += f[i] * L::cz(i);
    }
    for (int d = 0; d < 3; d++) u[d] = (m[d] - 0.5f * p.force[d]) / (1.f + drho);
}

// Collide populations in place: BGK or TRT, plus Guo's forcing term (split into its even and odd
// parts so TRT relaxes each with its own rate).
template <class L>
EDDY_HD void collide(float* f, float drho, const float u[3], const Params& p) {
    const float uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const float uF = u[0] * p.force[0] + u[1] * p.force[1] + u[2] * p.force[2];
    float post[L::Q];
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) {
        const int j = L::opp(i);
        const float cu = L::cx(i) * u[0] + L::cy(i) * u[1] + L::cz(i) * u[2];
        const float cF = L::cx(i) * p.force[0] + L::cy(i) * p.force[1] + L::cz(i) * p.force[2];
        const float ei = feq(L::w(i), drho, cu, uu);
        const float ej = feq(L::w(j), drho, -cu, uu);
        const float fp = 0.5f * (f[i] + f[j]), fm = 0.5f * (f[i] - f[j]);
        const float ep = 0.5f * (ei + ej), em = 0.5f * (ei - ej);
        const float s_even = L::w(i) * (9.f * cu * cF - 3.f * uF);
        const float s_odd = L::w(i) * 3.f * cF;
        post[i] = f[i] - p.omega_plus * (fp - ep) - p.omega_minus * (fm - em) + (1.f - 0.5f * p.omega_plus) * s_even +
                  (1.f - 0.5f * p.omega_minus) * s_odd;
    }
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) f[i] = post[i];
}

EDDY_HD long cell_index(int x, int y, int z, const Params& p) { return (static_cast<long>(z) * p.ny + y) * p.nx + x; }

// Velocity and density a boundary cell takes from its inner neighbour (non-equilibrium
// extrapolation, Guo, Zheng and Shi 2002): f = feq(boundary) + (f(neighbour) - feq(neighbour)).
template <class L>
EDDY_HD void open_boundary(float* f, const float* src, long cells, long nb, uint8_t type, int y, int z, const Params& p) {
    float fn[L::Q];
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) fn[i] = src[i * cells + nb];
    float rn, un[3];
    moments_post<L>(fn, p, rn, un);
    float rb, ub[3];
    if (type == INLET) {
        rb = rn;
        const float* v = p.inlet_u + 3 * (static_cast<long>(z) * p.ny + y);
        ub[0] = p.inlet_scale * v[0], ub[1] = p.inlet_scale * v[1], ub[2] = p.inlet_scale * v[2];
    } else {
        rb = p.outlet_rho - 1.f;
        ub[0] = un[0], ub[1] = un[1], ub[2] = un[2];
    }
    const float uun = un[0] * un[0] + un[1] * un[1] + un[2] * un[2];
    const float uub = ub[0] * ub[0] + ub[1] * ub[1] + ub[2] * ub[2];
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) {
        const float cun = L::cx(i) * un[0] + L::cy(i) * un[1] + L::cz(i) * un[2];
        const float cub = L::cx(i) * ub[0] + L::cy(i) * ub[1] + L::cz(i) * ub[2];
        f[i] = feq(L::w(i), rb, cub, uub) + (fn[i] - feq(L::w(i), rn, cun, uun));
    }
}

// One time step of one cell: pull from src, collide, write to dst. Optionally write density and
// velocity (only when asked: it costs memory traffic).
template <class L>
EDDY_HD void step_cell(long c, const float* src, float* dst, const uint8_t* flags, const Params& p, float* rho_out, float* u_out) {
    const long cells = static_cast<long>(p.nx) * p.ny * p.nz;
    const uint8_t type = flags[c];
    if (is_solid(type)) return;
    const int x = static_cast<int>(c % p.nx);
    const int y = static_cast<int>((c / p.nx) % p.ny);
    const int z = static_cast<int>(c / (static_cast<long>(p.nx) * p.ny));
    float f[L::Q];
    if (type == INLET || type == OUTLET) {
        const int nx = type == INLET ? x + 1 : x - 1;  // inner neighbour along x
        open_boundary<L>(f, src, cells, cell_index(nx, y, z, p), type, y, z, p);
    } else {
EDDY_UNROLL
        for (int i = 0; i < L::Q; i++) {
            const int sx = wrap(x - L::cx(i), p.nx, p.periodic[0]);
            const int sy = wrap(y - L::cy(i), p.ny, p.periodic[1]);
            const int sz = wrap(z - L::cz(i), p.nz, p.periodic[2]);
            const bool outside = sx < 0 || sx >= p.nx || sy < 0 || sy >= p.ny || sz < 0 || sz >= p.nz;
            const uint8_t st = outside ? WALL : flags[cell_index(sx, sy, sz, p)];
            if (is_solid(st)) {
                // Halfway bounce-back: what left towards the wall comes back reversed. A moving
                // wall adds the momentum it imparts, 2 w rho (c.u_w) / cs^2.
                float v = src[L::opp(i) * cells + c];
                if (st == MOVING) {
                    const float cu = L::cx(i) * p.wall_u[0] + L::cy(i) * p.wall_u[1] + L::cz(i) * p.wall_u[2];
                    v += 6.f * L::w(i) * cu;
                }
                f[i] = v;
            } else {
                f[i] = src[i * cells + cell_index(sx, sy, sz, p)];
            }
        }
    }
    float drho, u[3];
    moments<L>(f, p, drho, u);
    if (rho_out) {
        rho_out[c] = 1.f + drho;
        for (int d = 0; d < L::D; d++) u_out[c * L::D + d] = u[d];
    }
    collide<L>(f, drho, u, p);
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) dst[i * cells + c] = f[i];
}

// The momentum a fluid cell hands to neighbouring OBSTACLE cells in one step (momentum exchange,
// Ladd 1994): each population bounced back by the obstacle transfers 2 c_i f_i.
template <class L>
EDDY_HD void obstacle_force(long c, const float* f, const uint8_t* flags, const Params& p, float out[3]) {
    out[0] = out[1] = out[2] = 0.f;
    if (is_solid(flags[c])) return;
    const long cells = static_cast<long>(p.nx) * p.ny * p.nz;
    const int x = static_cast<int>(c % p.nx);
    const int y = static_cast<int>((c / p.nx) % p.ny);
    const int z = static_cast<int>(c / (static_cast<long>(p.nx) * p.ny));
    for (int i = 1; i < L::Q; i++) {
        const int nx = wrap(x + L::cx(i), p.nx, p.periodic[0]);
        const int ny = wrap(y + L::cy(i), p.ny, p.periodic[1]);
        const int nz = wrap(z + L::cz(i), p.nz, p.periodic[2]);
        if (nx < 0 || nx >= p.nx || ny < 0 || ny >= p.ny || nz < 0 || nz >= p.nz) continue;
        if (flags[cell_index(nx, ny, nz, p)] != OBSTACLE) continue;
        const float m = 2.f * (f[i * cells + c] + L::w(i));
        out[0] += m * L::cx(i);
        out[1] += m * L::cy(i);
        out[2] += m * L::cz(i);
    }
}

}  // namespace eddy
