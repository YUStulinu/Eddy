// A lattice Boltzmann simulation on the CPU (OpenMP) or the GPU (CUDA), running the same cell code.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "lbm.hpp"

namespace eddy {

enum class Backend { CPU, GPU };

// The domain: a cell type per cell, and the inlet velocity profile on the x = 0 face.
struct Geometry {
    int nx = 1, ny = 1, nz = 1;
    std::vector<uint8_t> flags;
    std::vector<float> inlet_u;  // 3 per (y, z) row; empty if there is no inlet
    std::array<int, 3> periodic = {0, 0, 0};

    Geometry() = default;
    Geometry(int nx_, int ny_, int nz_ = 1) : nx(nx_), ny(ny_), nz(nz_), flags(static_cast<size_t>(nx_) * ny_ * nz_, FLUID) {}
    long cells() const { return static_cast<long>(nx) * ny * nz; }
    uint8_t& at(int x, int y, int z = 0) { return flags[(static_cast<size_t>(z) * ny + y) * nx + x]; }
};

enum class Collision { BGK, TRT };

struct Physics {
    float viscosity = 0.1f;  // lattice units: nu = (1/omega - 1/2) / 3
    Collision collision = Collision::TRT;
    float magic = 3.f / 16.f;  // TRT's Lambda; 3/16 puts bounce-back walls exactly halfway
    std::array<float, 3> force = {0, 0, 0};
    std::array<float, 3> wall_u = {0, 0, 0};
    float outlet_rho = 1.f;
    float inlet_scale = 1.f;  // multiplies the geometry's inlet profile
};

// The two relaxation rates for a viscosity.
Params make_params(const Geometry& g, const Physics& ph);

bool gpu_available();
std::string gpu_name();

class Solver {
  public:
    virtual ~Solver() = default;
    // Advance n steps.
    virtual void step(int n) = 0;
    // Density and velocity (D components per cell) of the current state.
    virtual void fields(std::vector<float>& rho, std::vector<float>& u) = 0;
    // Force the fluid exerts on OBSTACLE cells now (lattice units).
    virtual std::array<double, 3> force() = 0;
    // Total mass (sum of all populations of fluid cells).
    virtual double mass() = 0;
    virtual void set_physics(const Physics& ph) = 0;
    virtual const Geometry& geometry() const = 0;
    virtual int lattice_q() const = 0;
    long steps = 0;
};

// lattice: "d2q9" or "d3q19".
std::unique_ptr<Solver> make_solver(const std::string& lattice, Backend backend, Geometry g, const Physics& ph);

}  // namespace eddy
