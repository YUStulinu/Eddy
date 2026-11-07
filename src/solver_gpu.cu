// The CUDA solver: one thread per cell, the same step_cell as the CPU.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "solver.hpp"

namespace eddy {

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_)); \
    } while (0)

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string gpu_name() {
    if (!gpu_available()) return "";
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
}

template <class L>
__global__ void k_step(const float* __restrict__ src, float* __restrict__ dst, const uint8_t* __restrict__ flags, Params p, long cells) {
    const long c = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (c < cells) step_cell<L>(c, src, dst, flags, p, nullptr, nullptr);
}

template <class L>
__global__ void k_fields(const float* __restrict__ f, const uint8_t* __restrict__ flags, Params p, long cells, float* rho, float* u) {
    const long c = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    if (c >= cells) return;
    if (is_solid(flags[c])) {
        rho[c] = 0.f;
        for (int d = 0; d < L::D; d++) u[c * L::D + d] = 0.f;
        return;
    }
    float g[L::Q];
EDDY_UNROLL
    for (int i = 0; i < L::Q; i++) g[i] = f[i * cells + c];
    float r, v[3];
    moments_post<L>(g, p, r, v);
    rho[c] = 1.f + r;
    for (int d = 0; d < L::D; d++) u[c * L::D + d] = v[d];
}

template <class L>
__global__ void k_force(const float* __restrict__ f, const uint8_t* __restrict__ flags, Params p, long cells, double* out) {
    __shared__ double s[3][256];
    const long c = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    float o[3] = {0, 0, 0};
    if (c < cells) obstacle_force<L>(c, f, flags, p, o);
    for (int d = 0; d < 3; d++) s[d][threadIdx.x] = o[d];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            for (int d = 0; d < 3; d++) s[d][threadIdx.x] += s[d][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int d = 0; d < 3; d++) atomicAdd(&out[d], s[d][0]);
}

template <class L>
__global__ void k_mass(const float* __restrict__ f, const uint8_t* __restrict__ flags, long cells, double* out) {
    __shared__ double s[256];
    const long c = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    double m = 0;
    if (c < cells && !is_solid(flags[c]))
        for (int i = 0; i < L::Q; i++) m += f[i * cells + c] + L::w(i);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

template <class L>
class GpuSolver : public Solver {
  public:
    GpuSolver(Geometry g, const Physics& ph) : g_(std::move(g)) {
        const long n = g_.cells();
        std::vector<float> init(static_cast<size_t>(n) * L::Q, 0.f);  // at rest (shifted populations)
        const size_t bytes = init.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&a_, bytes));
        CUDA_CHECK(cudaMalloc(&b_, bytes));
        CUDA_CHECK(cudaMemcpy(a_, init.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_, init.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&flags_, n));
        CUDA_CHECK(cudaMemcpy(flags_, g_.flags.data(), n, cudaMemcpyHostToDevice));
        if (!g_.inlet_u.empty()) {
            CUDA_CHECK(cudaMalloc(&inlet_, g_.inlet_u.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(inlet_, g_.inlet_u.data(), g_.inlet_u.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&scratch_, 3 * sizeof(double)));
        set_physics(ph);
    }

    ~GpuSolver() override {
        cudaFree(a_);
        cudaFree(b_);
        cudaFree(flags_);
        cudaFree(inlet_);
        cudaFree(scratch_);
    }

    void set_physics(const Physics& ph) override {
        p_ = make_params(g_, ph);
        p_.inlet_u = inlet_;
    }

    int blocks() const { return static_cast<int>((g_.cells() + 255) / 256); }

    void step(int n) override {
        for (int s = 0; s < n; s++) {
            k_step<L><<<blocks(), 256>>>(a_, b_, flags_, p_, g_.cells());
            std::swap(a_, b_);
            steps++;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void fields(std::vector<float>& rho, std::vector<float>& u) override {
        const long n = g_.cells();
        float *dr, *du;
        CUDA_CHECK(cudaMalloc(&dr, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&du, n * L::D * sizeof(float)));
        k_fields<L><<<blocks(), 256>>>(a_, flags_, p_, n, dr, du);
        rho.resize(n);
        u.resize(n * L::D);
        CUDA_CHECK(cudaMemcpy(rho.data(), dr, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(u.data(), du, n * L::D * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dr);
        cudaFree(du);
    }

    std::array<double, 3> force() override {
        CUDA_CHECK(cudaMemset(scratch_, 0, 3 * sizeof(double)));
        k_force<L><<<blocks(), 256>>>(a_, flags_, p_, g_.cells(), scratch_);
        std::array<double, 3> out{};
        CUDA_CHECK(cudaMemcpy(out.data(), scratch_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
        return out;
    }

    double mass() override {
        CUDA_CHECK(cudaMemset(scratch_, 0, sizeof(double)));
        k_mass<L><<<blocks(), 256>>>(a_, flags_, g_.cells(), scratch_);
        double m = 0;
        CUDA_CHECK(cudaMemcpy(&m, scratch_, sizeof(double), cudaMemcpyDeviceToHost));
        return m;
    }

    const Geometry& geometry() const override { return g_; }
    int lattice_q() const override { return L::Q; }

  private:
    Geometry g_;
    Params p_;
    float *a_ = nullptr, *b_ = nullptr, *inlet_ = nullptr;
    uint8_t* flags_ = nullptr;
    double* scratch_ = nullptr;
};

std::unique_ptr<Solver> make_gpu_solver(const std::string& lattice, Geometry g, const Physics& ph) {
    if (lattice == "d2q9") return std::make_unique<GpuSolver<D2Q9>>(std::move(g), ph);
    if (lattice == "d3q19") return std::make_unique<GpuSolver<D3Q19>>(std::move(g), ph);
    throw std::runtime_error("unknown lattice " + lattice);
}

}  // namespace eddy
