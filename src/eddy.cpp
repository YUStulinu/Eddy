// eddy: lattice Boltzmann flows on the GPU or CPU.
//
//   eddy cavity    --n 128 --re 100 [--ulid 0.1]                lid-driven cavity to steady state
//   eddy poiseuille --n 32 [--tau 0.8] [--bgk]                   force-driven channel
//   eddy duct      --n 24                                        3D square duct (D3Q19)
//   eddy cylinder  --d 20 --re 20|100 [--umax 0.05] [--time T] [--ramp 5]    Schafer-Turek benchmark 2D-1 / 2D-2
//   eddy image     --png obstacles.png [--re 200]                channel flow past a drawn shape
//   eddy bench     --lattice d2q9|d3q19 --n 512 [--steps 200]    million lattice updates per second
//   common: --cpu (else the GPU if there is one), --out DIR
//
// Steady cases write fields.bin (nx ny nz D int32, then flags u8, rho f32, u f32 x D) and
// summary.json; the cylinder writes forces.csv too.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "solver.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

using namespace eddy;
namespace fs = std::filesystem;

struct Args {
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    double num(const std::string& k, double def) const { return has(k) ? std::stod(kv.at(k)) : def; }
    std::string str(const std::string& k, const std::string& def) const { return has(k) ? kv.at(k) : def; }
};

static Backend backend_of(const Args& a) {
    if (a.has("cpu")) return Backend::CPU;
    return gpu_available() ? Backend::GPU : Backend::CPU;
}

static void write_fields(const fs::path& path, Solver& s, int dims) {
    std::vector<float> rho, u;
    s.fields(rho, u);
    const Geometry& g = s.geometry();
    std::ofstream o(path, std::ios::binary);
    const int32_t hdr[4] = {g.nx, g.ny, g.nz, dims};
    o.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    o.write(reinterpret_cast<const char*>(g.flags.data()), g.flags.size());
    o.write(reinterpret_cast<const char*>(rho.data()), rho.size() * sizeof(float));
    o.write(reinterpret_cast<const char*>(u.data()), u.size() * sizeof(float));
}

static void write_json(const fs::path& path, const std::map<std::string, std::string>& kv) {
    std::ofstream o(path);
    o << "{";
    bool first = true;
    for (auto& [k, v] : kv) {
        o << (first ? "" : ",") << "\n  \"" << k << "\": " << v;
        first = false;
    }
    o << "\n}\n";
}

static std::string q(const std::string& s) { return "\"" + s + "\""; }
static std::string num(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

// Runs until the velocity field stops changing: the relative L2 change over `every` steps below
// `tol`. Returns the steps taken.
static long run_to_steady(Solver& s, double scale, double tol, long max_steps, int every = 1000) {
    std::vector<float> rho, u, prev;
    s.fields(rho, prev);
    while (s.steps < max_steps) {
        s.step(every);
        s.fields(rho, u);
        // Relative L2 change of the velocity field: unlike the largest change, it is not held up by
        // the round-off noise of the few cells at a singular corner (a cavity's lid ends).
        double num = 0, den = 0;
        for (size_t i = 0; i < u.size(); i++) {
            const double du = static_cast<double>(u[i]) - prev[i];
            num += du * du;
            den += static_cast<double>(u[i]) * u[i];
        }
        const double d = std::sqrt(num / std::max(den, 1e-300)) * scale;  // scale cancels below
        if (!std::isfinite(d)) throw std::runtime_error("the simulation blew up");
        if (std::getenv("EDDY_TRACE")) std::fprintf(stderr, "step %ld change %.3g\n", s.steps, d / scale);
        if (d / scale < tol) break;
        prev.swap(u);
    }
    return s.steps;
}

static Physics physics(const Args& a, double viscosity) {
    Physics ph;
    ph.viscosity = static_cast<float>(viscosity);
    ph.collision = a.has("bgk") ? Collision::BGK : Collision::TRT;
    return ph;
}

static int cavity(const Args& a, const fs::path& out) {
    const int n = static_cast<int>(a.num("n", 128));
    const double re = a.num("re", 100), ulid = a.num("ulid", 0.1);
    Geometry g(n + 2, n + 2);
    for (int x = 0; x < n + 2; x++) {
        g.at(x, 0) = WALL;
        g.at(x, n + 1) = MOVING;
    }
    for (int y = 0; y < n + 2; y++) g.at(0, y) = g.at(n + 1, y) = WALL;
    Physics ph = physics(a, ulid * n / re);
    ph.wall_u = {static_cast<float>(ulid), 0, 0};
    auto s = make_solver("d2q9", backend_of(a), g, ph);
    auto t0 = std::chrono::steady_clock::now();
    // Float round-off leaves a velocity noise of about 1e-6 (absolute) that never settles, so
    // "steady" means a relative L2 change below 5e-5 per 1000 steps.
    const long steps = run_to_steady(*s, ulid, a.num("tol", 5e-5), static_cast<long>(a.num("max-steps", 2e6)));
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    write_fields(out / "fields.bin", *s, 2);
    write_json(out / "summary.json", {{"case", q("cavity")}, {"n", num(n)}, {"re", num(re)}, {"ulid", num(ulid)},
                                     {"viscosity", num(ph.viscosity)}, {"steps", num(steps)}, {"seconds", num(secs)},
                                     {"collision", q(a.has("bgk") ? "BGK" : "TRT")}});
    std::printf("cavity n=%d Re=%g: %ld steps, %.1f s\n", n, re, steps, secs);
    return 0;
}

static int poiseuille(const Args& a, const fs::path& out) {
    const int n = static_cast<int>(a.num("n", 32));
    const double tau = a.num("tau", 0.8), umax = a.num("umax", 0.05);
    const double nu = (tau - 0.5) / 3.0;
    Geometry g(4, n + 2);
    g.periodic = {1, 0, 0};
    for (int x = 0; x < 4; x++) g.at(x, 0) = g.at(x, n + 1) = WALL;
    Physics ph = physics(a, nu);
    // Walls halfway between nodes: channel width n; u(y) = g y (n - y) / (2 nu).
    ph.force = {static_cast<float>(8.0 * nu * umax / (n * static_cast<double>(n))), 0, 0};
    auto s = make_solver("d2q9", backend_of(a), g, ph);
    // Transients decay like exp(-pi^2 nu t / n^2): 20 diffusion times leave nothing of them.
    const long steps = static_cast<long>(a.num("steps", 20.0 * n * n / nu));
    s->step(static_cast<int>(steps));
    write_fields(out / "fields.bin", *s, 2);
    write_json(out / "summary.json", {{"case", q("poiseuille")}, {"n", num(n)}, {"tau", num(tau)}, {"viscosity", num(nu)},
                                     {"force", num(ph.force[0])}, {"umax", num(umax)}, {"steps", num(steps)},
                                     {"collision", q(a.has("bgk") ? "BGK" : "TRT")}});
    std::printf("poiseuille n=%d tau=%g: %ld steps\n", n, tau, steps);
    return 0;
}

static int duct(const Args& a, const fs::path& out) {
    const int n = static_cast<int>(a.num("n", 24));
    const double tau = a.num("tau", 0.8), umax = a.num("umax", 0.05);
    const double nu = (tau - 0.5) / 3.0;
    Geometry g(4, n + 2, n + 2);
    g.periodic = {1, 0, 0};
    for (int z = 0; z < n + 2; z++)
        for (int y = 0; y < n + 2; y++)
            for (int x = 0; x < 4; x++)
                if (y == 0 || z == 0 || y == n + 1 || z == n + 1) g.at(x, y, z) = WALL;
    Physics ph = physics(a, nu);
    // The centre-line velocity of a square duct is about 0.5895 * g a^2 / nu (a: half-width)...
    // the exact profile is computed in validation; here the force just sets a moderate speed.
    const double half = n / 2.0;
    ph.force = {static_cast<float>(umax * nu / (0.2947 * half * half)), 0, 0};
    auto s = make_solver("d3q19", backend_of(a), g, ph);
    const long steps = static_cast<long>(a.num("steps", 20.0 * n * n / nu));
    s->step(static_cast<int>(steps));
    write_fields(out / "fields.bin", *s, 3);
    write_json(out / "summary.json", {{"case", q("duct")}, {"n", num(n)}, {"tau", num(tau)}, {"viscosity", num(nu)},
                                     {"force", num(ph.force[0])}, {"steps", num(steps)},
                                     {"collision", q(a.has("bgk") ? "BGK" : "TRT")}});
    std::printf("duct n=%d: %ld steps\n", n, steps);
    return 0;
}

// A channel with a parabolic inflow and a pressure outlet; `solid(x, y)` marks obstacles.
template <class F>
static Geometry channel(int nx, int ny, double umax, F solid) {
    Geometry g(nx, ny + 2);
    for (int x = 0; x < nx; x++) g.at(x, 0) = g.at(x, ny + 1) = WALL;
    g.inlet_u.assign(3 * (ny + 2), 0.f);
    for (int y = 1; y <= ny; y++) {
        g.at(0, y) = INLET;
        g.at(nx - 1, y) = OUTLET;
        const double yy = y - 0.5;  // walls halfway: the fluid spans 0..ny
        g.inlet_u[3 * y] = static_cast<float>(4.0 * umax * yy * (ny - yy) / (static_cast<double>(ny) * ny));
    }
    for (int y = 1; y <= ny; y++)
        for (int x = 1; x < nx - 1; x++)
            if (solid(x, y)) g.at(x, y) = OBSTACLE;
    return g;
}

static int cylinder(const Args& a, const fs::path& out) {
    // Schafer & Turek (1996): channel 2.2 x 0.41, cylinder of diameter 0.1 centred at (0.2, 0.2),
    // parabolic inflow of peak Um; Re = U D / nu with U = 2 Um / 3. Lengths here in units of D/d.
    const int d = static_cast<int>(a.num("d", 20));
    // Peak inflow 0.05 (Mach 0.09): at 0.1 the drag is still a few per cent off (compressibility).
    const double re = a.num("re", 20), um = a.num("umax", 0.05);
    const int ny = static_cast<int>(std::lround(4.1 * d)), nx = 22 * d;
    const double cx = 2.0 * d, cy = 2.0 * d, r = d / 2.0;
    Geometry g = channel(nx, ny, um, [&](int x, int y) {
        const double px = x, py = y - 0.5;
        return (px - cx) * (px - cx) + (py - cy) * (py - cy) <= r * r;
    });
    const double ubar = 2.0 * um / 3.0;
    const double nu = ubar * d / re;
    Physics ph = physics(a, nu);
    auto s = make_solver("d2q9", backend_of(a), g, ph);
    // Simulated time in units of D / U.
    const double t_end = a.num("time", re < 50 ? 60 : 200);
    const long steps = static_cast<long>(t_end * d / ubar);
    const int every = std::max(1, static_cast<int>(a.num("sample", 10)));
    std::ofstream csv(out / "forces.csv");
    csv << "step,time,cd,cl\n";
    const double norm = 2.0 / (ubar * ubar * d);  // rho = 1
    auto t0 = std::chrono::steady_clock::now();
    // The inflow starts gently (sin^2 over `ramp` time units): an impulsive start sends a strong
    // pressure wave back and forth between inlet and outlet, which takes long to die out.
    const double ramp = a.num("ramp", 5);
    const long ramp_steps = static_cast<long>(ramp * d / ubar);
    while (s->steps < steps) {
        if (s->steps <= ramp_steps) {
            const double k = ramp_steps > 0 ? std::sin(0.5 * 3.14159265358979 * std::min(1.0, static_cast<double>(s->steps) / ramp_steps)) : 1.0;
            ph.inlet_scale = static_cast<float>(k * k);
            s->set_physics(ph);
        }
        s->step(every);
        auto f = s->force();
        if (!std::isfinite(f[0])) throw std::runtime_error("the simulation blew up");
        csv << s->steps << "," << num(s->steps * ubar / d) << "," << num(f[0] * norm) << "," << num(f[1] * norm) << "\n";
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    write_fields(out / "fields.bin", *s, 2);
    write_json(out / "summary.json", {{"case", q("cylinder")}, {"d", num(d)}, {"re", num(re)}, {"umax", num(um)}, {"ubar", num(ubar)},
                                     {"viscosity", num(nu)}, {"nx", num(nx)}, {"ny", num(ny)}, {"steps", num(steps)},
                                     {"seconds", num(secs)}, {"collision", q(a.has("bgk") ? "BGK" : "TRT")},
                                     {"mlups", num(static_cast<double>(steps) * g.cells() / secs / 1e6)}});
    std::printf("cylinder d=%d Re=%g: %ld steps (%.0f cells), %.1f s\n", d, re, steps, static_cast<double>(g.cells()), secs);
    return 0;
}

static int image(const Args& a, const fs::path& out) {
    const std::string png = a.str("png", "");
    int w, h, ch;
    unsigned char* img = stbi_load(png.c_str(), &w, &h, &ch, 1);
    if (!img) throw std::runtime_error("cannot read " + png);
    const double um = a.num("umax", 0.1), re = a.num("re", 200);
    // Dark pixels are solid; the image's height is the channel's.
    Geometry g = channel(w, h, um, [&](int x, int y) { return img[(h - y) * w + x] < 128; });
    stbi_image_free(img);
    const double nu = (2.0 * um / 3.0) * h / re;
    auto s = make_solver("d2q9", backend_of(a), g, physics(a, nu));
    const long steps = static_cast<long>(a.num("steps", 20000));
    const int frames = static_cast<int>(a.num("frames", 1));
    for (int k = 1; k <= frames; k++) {
        s->step(static_cast<int>(steps / frames));
        write_fields(out / ("frame" + std::to_string(k) + ".bin"), *s, 2);
    }
    std::printf("image %dx%d: %ld steps, %d frames\n", w, h, s->steps, frames);
    return 0;
}

static int bench(const Args& a, const fs::path& out) {
    const std::string lat = a.str("lattice", "d2q9");
    const int n = static_cast<int>(a.num("n", lat == "d2q9" ? 1024 : 128));
    const int steps = static_cast<int>(a.num("steps", 100));
    Geometry g = lat == "d2q9" ? Geometry(n, n) : Geometry(n, n, n);
    g.periodic = {1, 1, 1};
    Physics ph;
    ph.viscosity = 0.05f;
    ph.force = {1e-6f, 0, 0};
    auto s = make_solver(lat, backend_of(a), g, ph);
    s->step(std::max(1, steps / 10));  // warm up
    auto t0 = std::chrono::steady_clock::now();
    s->step(steps);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double mlups = static_cast<double>(steps) * g.cells() / secs / 1e6;
    // Bytes moved per cell update: read and write every population, read the flags (neighbours'
    // flags mostly hit in cache).
    const int qn = lat == "d2q9" ? 9 : 19;
    const double gbps = mlups * 1e6 * (2.0 * qn * 4 + 1) / 1e9;
    std::printf("{\"lattice\": \"%s\", \"n\": %d, \"cells\": %ld, \"backend\": \"%s\", \"steps\": %d, \"seconds\": %.4f, \"mlups\": %.2f, \"effective_gb_s\": %.2f}\n",
                lat.c_str(), n, g.cells(), backend_of(a) == Backend::GPU ? "gpu" : "cpu", steps, secs, mlups, gbps);
    (void)out;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: eddy cavity|poiseuille|duct|cylinder|image|bench|info [--key value ...]\n");
        return 2;
    }
    Args a;
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
            a.kv[k] = argv[++i];
        } else {
            a.kv[k] = "1";
        }
    }
    const std::string cmd = argv[1];
    const fs::path out = a.str("out", "out/" + cmd);
    try {
        if (cmd == "info") {
            std::printf("{\"gpu\": \"%s\"}\n", gpu_name().c_str());
            return 0;
        }
        if (cmd == "bench") return bench(a, out);
        fs::create_directories(out);
        if (cmd == "cavity") return cavity(a, out);
        if (cmd == "poiseuille") return poiseuille(a, out);
        if (cmd == "duct") return duct(a, out);
        if (cmd == "cylinder") return cylinder(a, out);
        if (cmd == "image") return image(a, out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
