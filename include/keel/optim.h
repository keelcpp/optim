#pragma once

#include "keel/array.h"
#include "keel/core.h"
#include "keel/function_ref.h"
#include "keel/math.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::optim {

using ScalarFn = FunctionRef<f64(f64)>;
using VectorFn = FunctionRef<f64(ArrayRef<f64>)>;

enum class Status : u8 {
    Ok = 0,
    InvalidInput,    ///< Non-finite bounds/guess, a >= b, or empty x0
    InvalidCallback, ///< Objective callback is empty
    NoBracket,       ///< Outward search failed to locate a sign change
    NoSignChange,    ///< f(a) and f(b) have the same sign
    MaxIterations,   ///< Iteration limit exceeded before convergence
};

struct BracketResult {
    Status status = Status::Ok;
    f64 a = 0.0;
    f64 b = 0.0;

    [[nodiscard]] constexpr bool ok() const noexcept { return status == Status::Ok; }
};

struct ScalarResult {
    Status status = Status::Ok;
    f64 x = 0.0;
    f64 fval = 0.0;
    i32 iterations = 0;

    [[nodiscard]] constexpr bool ok() const noexcept { return status == Status::Ok; }
};

struct VectorResult {
    Status status = Status::Ok;
    Array<f64> x;
    f64 fval = 0.0;
    i32 iterations = 0;

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }
};

/// Expand an interval outward from `x0` until a sign change is found.
/// Returns `a == b` if an exact root is hit during expansion.
[[nodiscard]] BracketResult find_bracket(ScalarFn fn, f64 x0, i32 max_expansions = 60);

/// Brent's method root finder on `[a, b]` (requires `f(a)` and `f(b)` to have
/// opposite signs, or one endpoint to be an exact root).
[[nodiscard]] ScalarResult brent_root(ScalarFn fn, f64 a, f64 b, f64 tol = 1e-15, i32 max_iter = 200);

/// Scalar root finder starting from initial guess `x0` (auto-brackets then runs `brent_root`).
[[nodiscard]] ScalarResult fzero(ScalarFn fn, f64 x0, f64 tol = 1e-15, i32 max_iter = 200);

/// Bounded 1D minimization on `[a, b]` via Brent's golden-section + parabolic hybrid.
[[nodiscard]] ScalarResult brent_min(ScalarFn fn, f64 a, f64 b, f64 tol = 1e-6, i32 max_iter = 200);

/// Unconstrained multi-dimensional minimization via Nelder-Mead simplex.
[[nodiscard]] VectorResult nelder_mead(VectorFn fn, ArrayRef<f64> x0, f64 tol = 1e-4, i32 max_iter = 500,
                                       Allocator* alloc = nullptr);

} // namespace keel::optim
