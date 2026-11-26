# keel::optim

Zero-STL C++20 derivative-free root-finding and optimization library built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **`keel::optim::find_bracket`** — Outward geometric sign-change bracketing around an initial guess
- **`keel::optim::brent_root`** / **`keel::optim::fzero`** — Brent's method scalar root finder (secant + inverse quadratic interpolation + bisection)
- **`keel::optim::brent_min`** — Brent's bounded 1D minimization (golden-section + parabolic interpolation hybrid)
- **`keel::optim::nelder_mead`** — Unconstrained multi-dimensional Nelder–Mead simplex minimization
- Non-owning `keel::FunctionRef` callbacks — works cleanly with `-fno-exceptions -fno-rtti`
- Zero STL headers in public headers and implementation
- Differential reference test suite (`tests/verify_reference.py`) validated against SciPy and MATLAB (`fzero`, `fminbnd`, `fminsearch`)

## Usage

```cpp
#include <keel/optim.h>

auto root = keel::optim::fzero([](keel::f64 x) { return x * x - 2.0; }, 1.0);
// root.ok() == true, root.x ~= 1.4142135623730951
```

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-fno-exceptions -fno-rtti` build:

```bash
cmake -S . -B build-noeh -DCMAKE_BUILD_TYPE=Release -DKEEL_OPTIM_NO_EXCEPTIONS_RTTI=ON
cmake --build build-noeh -j
ctest --test-dir build-noeh --output-on-failure
```
