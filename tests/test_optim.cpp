#include <keel/format.h>
#include <keel/math.h>
#include <keel/optim.h>
#include <keel/string.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: %s", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: %s (got %.15g, expected %.15g, tol %.3g)", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_brent_root_and_fzero() {
    auto f = [](keel::f64 x) { return x * x - 2.0; };
    auto r1 = keel::optim::brent_root(f, 0.0, 2.0);
    check(r1.ok(), "brent_root ok");
    check_near(r1.x, keel::sqrt(2.0), 1e-14, "brent_root sqrt(2)");

    auto r2 = keel::optim::fzero([](keel::f64 x) { return keel::cos(x) - x; }, 0.5);
    check(r2.ok(), "fzero cos(x)-x ok");
    check_near(r2.x, 0.7390851332151607, 1e-14, "fzero Dottie number");

    // Symmetric even function starting at 0: checks both left/right previous intervals
    auto r_even = keel::optim::fzero([](keel::f64 x) { return x * x - 4.0; }, 0.0);
    check(r_even.ok(), "fzero symmetric x^2-4 from 0");
    check_near(keel::abs(r_even.x), 2.0, 1e-14, "fzero |x|==2");

    // Large root accuracy (tests exact machine eps in tol1)
    auto r_big = keel::optim::brent_root([](keel::f64 x) { return x - 1234.567890123456; }, 1000.0, 1500.0, 1e-15);
    check(r_big.ok(), "brent_root large root ok");
    check_near(r_big.x, 1234.567890123456, 1e-12, "brent_root large root precision");

    // NaN objective must NOT masquerade as a sign change!
    auto r_nan = keel::optim::brent_root([](keel::f64 x) { return (x > 0.5) ? keel::kNaN : -1.0; }, 0.0, 1.0);
    check(r_nan.status == keel::optim::Status::InvalidInput, "brent_root NaN endpoint rejected");

    auto br_nan = keel::optim::find_bracket([](keel::f64 x) { return (x > 0.0) ? keel::kNaN : -1.0; }, 0.0, 10);
    check(br_nan.status == keel::optim::Status::NoBracket, "find_bracket NaN does not fake sign change");

    // No sign change
    auto bad = keel::optim::brent_root([](keel::f64 x) { return x * x + 1.0; }, -1.0, 1.0);
    check(bad.status == keel::optim::Status::NoSignChange, "brent_root NoSignChange");

    auto nobr = keel::optim::find_bracket([](keel::f64 x) { return x * x + 1.0; }, 0.0, 10);
    check(nobr.status == keel::optim::Status::NoBracket, "find_bracket NoBracket");
}

void test_brent_min() {
    auto f = [](keel::f64 x) { return (x - 1.25) * (x - 1.25) + 3.0; };
    auto r = keel::optim::brent_min(f, -2.0, 5.0, 1e-10);
    check(r.ok(), "brent_min ok");
    check_near(r.x, 1.25, 1e-7, "brent_min argmin");
    check_near(r.fval, 3.0, 1e-12, "brent_min fval");

    // Tight tolerance near 0 (previously clamped by hardcoded 1e-10 floor)
    auto f0 = [](keel::f64 x) { return x * x; };
    auto r0 = keel::optim::brent_min(f0, -1.0, 2.0, 1e-12);
    check(r0.ok(), "brent_min near zero ok");
    check_near(r0.x, 0.0, 1e-11, "brent_min tight tol near zero");

    // Hitting max_iter MUST report Status::MaxIterations
    auto r_max = keel::optim::brent_min(f, -2.0, 5.0, 1e-12, 2);
    check(r_max.status == keel::optim::Status::MaxIterations, "brent_min MaxIterations status");

    check(keel::optim::brent_min(f, 2.0, 1.0).status == keel::optim::Status::InvalidInput, "brent_min bad bounds");
}

void test_nelder_mead() {
    auto rosen = [](keel::ArrayRef<keel::f64> x) {
        const keel::f64 a = 1.0 - x[0];
        const keel::f64 b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    const keel::f64 x0[2] = {-1.2, 1.0};
    auto r = keel::optim::nelder_mead(rosen, x0, 1e-8, 2000);
    check(r.ok(), "nelder_mead rosenbrock ok");
    check(r.x.size() == 2, "nelder_mead dim == 2");
    check_near(r.x[0], 1.0, 1e-4, "nelder_mead x[0]");
    check_near(r.x[1], 1.0, 1e-4, "nelder_mead x[1]");
    check_near(r.fval, 0.0, 1e-7, "nelder_mead fval");

    // Hitting max_iter MUST report Status::MaxIterations
    auto r_max = keel::optim::nelder_mead(rosen, x0, 1e-12, 5);
    check(r_max.status == keel::optim::Status::MaxIterations, "nelder_mead MaxIterations status");

    // Non-finite x0 rejected
    const keel::f64 x_nan[2] = {1.0, keel::kNaN};
    check(keel::optim::nelder_mead(rosen, x_nan).status == keel::optim::Status::InvalidInput, "nelder_mead NaN x0");
    check(keel::optim::nelder_mead(rosen, {}).status == keel::optim::Status::InvalidInput, "nelder_mead empty x0");
}

void dump_reference() {
    // 1. Roots: transcendental, polynomial, cubic
    auto r1 = keel::optim::brent_root([](keel::f64 x) { return keel::exp(-x) - x; }, 0.0, 2.0, 1e-15);
    auto r2 = keel::optim::fzero([](keel::f64 x) { return x * x * x - 2.0 * x - 5.0; }, 2.0, 1e-15);
    auto r3 = keel::optim::brent_root([](keel::f64 x) { return keel::sin(x) - 0.5 * x; }, 1.0, 3.0, 1e-15);
    keel::println("root_exp_x: %.17g", r1.x);
    keel::println("root_cubic: %.17g", r2.x);
    keel::println("root_sin_halfx: %.17g", r3.x);

    // 2. Bounded 1D minima
    auto m1 = keel::optim::brent_min([](keel::f64 x) { return keel::exp(x) - 3.0 * x; }, 0.0, 3.0, 1e-10);
    auto m2 = keel::optim::brent_min([](keel::f64 x) { return x * keel::sin(x) + x * x * 0.25; }, -2.0, 4.0, 1e-10);
    keel::println("min_exp_3x: %.17g %.17g", m1.x, m1.fval);
    keel::println("min_xsinx: %.17g %.17g", m2.x, m2.fval);

    // 3. Nelder-Mead 2D & 3D
    auto rosen = [](keel::ArrayRef<keel::f64> x) {
        const keel::f64 a = 1.0 - x[0];
        const keel::f64 b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    const keel::f64 x0_r[2] = {-1.2, 1.0};
    auto nm1 = keel::optim::nelder_mead(rosen, x0_r, 1e-10, 4000);
    keel::println("nm_rosen: %.17g %.17g %.17g", nm1.x[0], nm1.x[1], nm1.fval);

    auto himmel_like = [](keel::ArrayRef<keel::f64> x) {
        return (x[0] - 2.0) * (x[0] - 2.0) + 3.0 * (x[1] + 1.5) * (x[1] + 1.5) +
               0.5 * (x[2] - 0.75) * (x[2] - 0.75) + 0.2 * (x[0] - 2.0) * (x[1] + 1.5);
    };
    const keel::f64 x0_3[3] = {0.0, 0.0, 0.0};
    auto nm2 = keel::optim::nelder_mead(himmel_like, x0_3, 1e-10, 4000);
    keel::println("nm_quad3d: %.17g %.17g %.17g %.17g", nm2.x[0], nm2.x[1], nm2.x[2], nm2.fval);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
    test_brent_root_and_fzero();
    test_brent_min();
    test_nelder_mead();
    if (g_failures == 0) {
        keel::println("ALL keel::optim TESTS PASSED");
        return 0;
    }
    keel::eprintln("%d FAILURE(S)", g_failures);
    return 1;
}
