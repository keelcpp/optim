#include <keel/format.h>
#include <keel/math.h>
#include <keel/optim.h>

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
    check_near(r1.x, keel::sqrt(2.0), 1e-13, "brent_root sqrt(2)");

    auto r2 = keel::optim::fzero([](keel::f64 x) { return keel::cos(x) - x; }, 0.5);
    check(r2.ok(), "fzero cos(x)-x ok");
    check_near(r2.x, 0.7390851332151607, 1e-13, "fzero Dottie number");

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

    check(keel::optim::brent_min(f, 2.0, 1.0).status == keel::optim::Status::InvalidInput, "brent_min bad bounds");
}

void test_nelder_mead() {
    // Rosenbrock function: min at (1, 1) with f = 0
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

    check(keel::optim::nelder_mead(rosen, {}).status == keel::optim::Status::InvalidInput, "nelder_mead empty x0");
}

} // namespace

int main() {
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
