#include <cmath>
#include <stdexcept>

#include "solver.hpp"

namespace eddy {

Params make_params(const Geometry& g, const Physics& ph) {
    Params p;
    p.nx = g.nx, p.ny = g.ny, p.nz = g.nz;
    for (int d = 0; d < 3; d++) {
        p.periodic[d] = g.periodic[d];
        p.force[d] = ph.force[d];
        p.wall_u[d] = ph.wall_u[d];
    }
    const float tau = 3.f * ph.viscosity + 0.5f;
    p.omega_plus = 1.f / tau;
    if (ph.collision == Collision::BGK) {
        p.omega_minus = p.omega_plus;
    } else {
        // Lambda = (tau+ - 1/2)(tau- - 1/2).
        const float tau_minus = ph.magic / (tau - 0.5f) + 0.5f;
        p.omega_minus = 1.f / tau_minus;
    }
    p.outlet_rho = ph.outlet_rho;
    p.inlet_scale = ph.inlet_scale;
    return p;
}

template <class L>
class CpuSolver : public Solver {
  public:
    CpuSolver(Geometry g, const Physics& ph) : g_(std::move(g)) {
        const long n = g_.cells();
        a_.assign(static_cast<size_t>(n) * L::Q, 0.f);
        // Start at rest, density 1: all stored (shifted) populations zero.
        b_ = a_;
        set_physics(ph);
    }

    void set_physics(const Physics& ph) override {
        p_ = make_params(g_, ph);
        p_.inlet_u = g_.inlet_u.empty() ? nullptr : g_.inlet_u.data();
    }

    void step(int n) override {
        const long cells = g_.cells();
        for (int s = 0; s < n; s++) {
            const float* src = a_.data();
            float* dst = b_.data();
            const uint8_t* flags = g_.flags.data();
            const Params p = p_;
#pragma omp parallel for schedule(static)
            for (long c = 0; c < cells; c++) step_cell<L>(c, src, dst, flags, p, nullptr, nullptr);
            std::swap(a_, b_);
            steps++;
        }
    }

    void fields(std::vector<float>& rho, std::vector<float>& u) override {
        const long n = g_.cells();
        rho.assign(n, 0.f);
        u.assign(n * L::D, 0.f);
        for (long c = 0; c < n; c++) {
            if (is_solid(g_.flags[c])) continue;
            float f[L::Q];
            for (int i = 0; i < L::Q; i++) f[i] = a_[i * n + c];
            float r, v[3];
            moments_post<L>(f, p_, r, v);
            rho[c] = 1.f + r;
            for (int d = 0; d < L::D; d++) u[c * L::D + d] = v[d];
        }
    }

    std::array<double, 3> force() override {
        double fx = 0, fy = 0, fz = 0;
        const long n = g_.cells();
#pragma omp parallel for reduction(+ : fx, fy, fz)
        for (long c = 0; c < n; c++) {
            float o[3];
            obstacle_force<L>(c, a_.data(), g_.flags.data(), p_, o);
            fx += o[0], fy += o[1], fz += o[2];
        }
        return {fx, fy, fz};
    }

    double mass() override {
        double m = 0;
        const long n = g_.cells();
        for (long c = 0; c < n; c++) {
            if (is_solid(g_.flags[c])) continue;
            m += 1.0;
            for (int i = 0; i < L::Q; i++) m += a_[i * n + c];
        }
        return m;
    }

    const Geometry& geometry() const override { return g_; }
    int lattice_q() const override { return L::Q; }

  private:
    Geometry g_;
    Params p_;
    std::vector<float> a_, b_;
};

#ifndef EDDY_CUDA
bool gpu_available() { return false; }
std::string gpu_name() { return ""; }
#endif

std::unique_ptr<Solver> make_gpu_solver(const std::string& lattice, Geometry g, const Physics& ph);

std::unique_ptr<Solver> make_solver(const std::string& lattice, Backend backend, Geometry g, const Physics& ph) {
    if (g.flags.size() != static_cast<size_t>(g.cells())) throw std::runtime_error("geometry flags do not match its size");
    if (backend == Backend::GPU) {
        if (!gpu_available()) throw std::runtime_error("no CUDA device (or built without CUDA)");
        return make_gpu_solver(lattice, std::move(g), ph);
    }
    if (lattice == "d2q9") return std::make_unique<CpuSolver<D2Q9>>(std::move(g), ph);
    if (lattice == "d3q19") return std::make_unique<CpuSolver<D3Q19>>(std::move(g), ph);
    throw std::runtime_error("unknown lattice " + lattice);
}

#ifndef EDDY_CUDA
std::unique_ptr<Solver> make_gpu_solver(const std::string&, Geometry, const Physics&) {
    throw std::runtime_error("built without CUDA");
}
#endif

}  // namespace eddy
