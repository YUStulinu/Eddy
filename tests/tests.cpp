// Tests of the solver's building blocks and invariants. Run on the CPU; the GPU check runs only
// when a CUDA device is present.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "solver.hpp"

using namespace eddy;

static int failures = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
            failures++;                                     \
        }                                                   \
    } while (0)

template <class L>
static void equilibrium_moments(const char* name) {
    // The equilibrium carries the density and momentum it was built from; weights sum to one.
    const float rho = 1.03f, u[3] = {0.05f, -0.02f, L::D == 3 ? 0.01f : 0.f};
    double s = 0, mx = 0, my = 0, mz = 0, w = 0;
    const float uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    for (int i = 0; i < L::Q; i++) {
        const float cu = L::cx(i) * u[0] + L::cy(i) * u[1] + L::cz(i) * u[2];
        const float e = feq(L::w(i), rho - 1.f, cu, uu) + L::w(i);  // feq is shifted by w
        s += e, mx += e * L::cx(i), my += e * L::cy(i), mz += e * L::cz(i), w += L::w(i);
        CHECK(L::opp(L::opp(i)) == i && L::cx(L::opp(i)) == -L::cx(i) && L::cz(L::opp(i)) == -L::cz(i), "%s opposite of %d", name, i);
    }
    CHECK(std::fabs(w - 1) < 1e-6, "%s weights sum to %g", name, w);
    CHECK(std::fabs(s - rho) < 1e-5, "%s density %g", name, s);
    CHECK(std::fabs(mx - rho * u[0]) < 1e-6 && std::fabs(my - rho * u[1]) < 1e-6 && std::fabs(mz - rho * u[2]) < 1e-6,
          "%s momentum %g %g %g", name, mx, my, mz);
}

static Geometry closed_box(int n) {
    Geometry g(n, n);
    for (int i = 0; i < n; i++) g.at(i, 0) = g.at(i, n - 1) = g.at(0, i) = g.at(n - 1, i) = WALL;
    for (int x = 1; x < n - 1; x++) g.at(x, n - 1) = MOVING;
    return g;
}

static void mass_is_conserved() {
    // Stationary walls: bounce-back and collision conserve mass. At rest nothing is rounded and
    // the mass stays exactly the same; driven by a force, single-precision rounding makes it drift
    // by about 2e-8 a step (below float's epsilon of 6e-8), nothing more.
    Geometry still = closed_box(40);
    for (auto& f : still.flags)
        if (f == MOVING) f = WALL;
    Physics ph;
    ph.viscosity = 0.02f;
    auto s = make_solver("d2q9", Backend::CPU, still, ph);
    double m0 = s->mass();
    s->step(2000);
    CHECK(s->mass() == m0, "mass at rest %g -> %g", m0, s->mass());
    ph.force = {1e-5f, 0, 0};
    s = make_solver("d2q9", Backend::CPU, still, ph);
    m0 = s->mass();
    s->step(2000);
    double m1 = s->mass();
    CHECK(std::fabs(m1 - m0) / m0 < 1e-4, "mass %g -> %g", m0, m1);
    // A moving lid gains or loses a little mass at its two corners, where it meets a still wall
    // (a known property of the momentum-corrected bounce-back); it stays small.
    ph.force = {0, 0, 0};
    ph.wall_u = {0.1f, 0, 0};
    s = make_solver("d2q9", Backend::CPU, closed_box(40), ph);
    m0 = s->mass();
    s->step(2000);
    m1 = s->mass();
    CHECK(std::fabs(m1 - m0) / m0 < 1e-4, "mass with a moving lid %g -> %g", m0, m1);
}

// A force-driven channel: with TRT and Lambda = 3/16 the parabola is reproduced to round-off,
// whatever the viscosity; BGK's walls drift with tau.
static double poiseuille_error(Collision c, double tau, int n) {
    const double nu = (tau - 0.5) / 3, umax = 0.02;
    Geometry g(3, n + 2);
    g.periodic = {1, 0, 0};
    for (int x = 0; x < 3; x++) g.at(x, 0) = g.at(x, n + 1) = WALL;
    Physics ph;
    ph.viscosity = static_cast<float>(nu);
    ph.collision = c;
    const double force = 8 * nu * umax / (static_cast<double>(n) * n);
    ph.force = {static_cast<float>(force), 0, 0};
    auto s = make_solver("d2q9", Backend::CPU, g, ph);
    s->step(static_cast<int>(8 * n * n / nu));
    std::vector<float> rho, u;
    s->fields(rho, u);
    double err = 0, ref = 0;
    for (int y = 1; y <= n; y++) {
        const double yy = y - 0.5, exact = force * yy * (n - yy) / (2 * nu);
        const double got = u[(static_cast<size_t>(y) * 3 + 1) * 2];
        err += (got - exact) * (got - exact), ref += exact * exact;
    }
    return std::sqrt(err / ref);
}

static void poiseuille() {
    const double trt_a = poiseuille_error(Collision::TRT, 0.6, 8), trt_b = poiseuille_error(Collision::TRT, 1.5, 8);
    const double bgk_b = poiseuille_error(Collision::BGK, 1.5, 8);
    CHECK(trt_a < 2e-4 && trt_b < 2e-4, "TRT errors %g %g", trt_a, trt_b);
    CHECK(bgk_b > 10 * trt_b, "BGK at tau 1.5 should be visibly worse: %g vs %g", bgk_b, trt_b);
}

// In a fully periodic box with a block in it, at steady state the body force pushing the fluid is
// all taken by the block: the momentum-exchange force must equal force x fluid cells.
static void force_balance() {
    const int n = 40;
    Geometry g(n, n);
    g.periodic = {1, 1, 0};
    int fluid = 0;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            if (x >= 15 && x < 25 && y >= 15 && y < 25) g.at(x, y) = OBSTACLE;
            else fluid++;
        }
    Physics ph;
    ph.viscosity = 0.1f;
    ph.force = {1e-5f, 0, 0};
    auto s = make_solver("d2q9", Backend::CPU, g, ph);
    s->step(20000);
    const auto f = s->force();
    const double expect = 1e-5 * fluid;
    CHECK(std::fabs(f[0] - expect) / expect < 0.01, "drag %g, body force %g", f[0], expect);
    CHECK(std::fabs(f[1]) < 0.01 * expect, "lift %g", f[1]);
}

static void open_boundaries_carry_the_inflow() {
    // An empty channel with a parabolic inlet and a pressure outlet: the flow develops into the
    // parabola everywhere, and what comes in goes out.
    const int nx = 60, ny = 20;
    const double um = 0.05;
    Geometry g(nx, ny + 2);
    for (int x = 0; x < nx; x++) g.at(x, 0) = g.at(x, ny + 1) = WALL;
    g.inlet_u.assign(3 * (ny + 2), 0.f);
    for (int y = 1; y <= ny; y++) {
        g.at(0, y) = INLET;
        g.at(nx - 1, y) = OUTLET;
        const double yy = y - 0.5;
        g.inlet_u[3 * y] = static_cast<float>(4 * um * yy * (ny - yy) / (ny * ny));
    }
    Physics ph;
    ph.viscosity = 0.05f;
    auto s = make_solver("d2q9", Backend::CPU, g, ph);
    s->step(20000);
    std::vector<float> rho, u;
    s->fields(rho, u);
    double worst = 0;
    for (int y = 1; y <= ny; y++) {
        const double yy = y - 0.5, exact = 4 * um * yy * (ny - yy) / (ny * ny);
        worst = std::max(worst, std::fabs(u[(static_cast<size_t>(y) * nx + nx / 2) * 2] - exact));
    }
    CHECK(worst < 0.01 * um, "mid-channel profile off by %g", worst);
}

static void cpu_and_gpu_agree() {
    if (!gpu_available()) {
        std::printf("skip: no CUDA device\n");
        return;
    }
    Physics ph;
    ph.viscosity = 0.02f;
    ph.wall_u = {0.1f, 0, 0};
    auto c = make_solver("d2q9", Backend::CPU, closed_box(64), ph);
    auto g = make_solver("d2q9", Backend::GPU, closed_box(64), ph);
    c->step(500);
    g->step(500);
    std::vector<float> r1, u1, r2, u2;
    c->fields(r1, u1);
    g->fields(r2, u2);
    double worst = 0;
    for (size_t i = 0; i < u1.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(u1[i] - u2[i])));
    CHECK(worst < 1e-5, "CPU and GPU velocities differ by %g", worst);
    // D3Q19 too.
    Geometry b(16, 16, 16);
    b.periodic = {1, 1, 1};
    Physics p3;
    p3.viscosity = 0.05f;
    p3.force = {1e-5f, 0, 0};
    auto c3 = make_solver("d3q19", Backend::CPU, b, p3);
    auto g3 = make_solver("d3q19", Backend::GPU, b, p3);
    c3->step(100);
    g3->step(100);
    c3->fields(r1, u1);
    g3->fields(r2, u2);
    worst = 0;
    for (size_t i = 0; i < u1.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(u1[i] - u2[i])));
    CHECK(worst < 1e-6, "D3Q19 CPU and GPU differ by %g", worst);
}

int main() {
    equilibrium_moments<D2Q9>("D2Q9");
    equilibrium_moments<D3Q19>("D3Q19");
    mass_is_conserved();
    poiseuille();
    force_balance();
    open_boundaries_carry_the_inflow();
    cpu_and_gpu_agree();
    std::printf(failures ? "%d failure(s)\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
