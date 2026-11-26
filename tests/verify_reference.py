#!/usr/bin/env python3
"""Differential reference verification for keel::optim against SciPy and MATLAB."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import optimize


def parse_dump(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = [float(tok) for tok in rest.strip().split()] if rest.strip() else []
        out[key.strip()] = vals
    return out


def check_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, failures: list[str]) -> None:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        failures.append(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    err = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
    status = "PASS" if err <= tol else "FAIL"
    print(f"  [{status}] {name:<28s} rel_err={err:.3e} (tol={tol:.1e})")
    if err > tol:
        failures.append(f"{name}: rel_err={err:.3e} > tol={tol:.1e}")


def run_matlab_checks(dump: dict[str, list[float]], failures: list[str]) -> None:
    if not shutil.which("matlab"):
        print("  [SKIP] MATLAB not found on PATH")
        return

    script = r"""
format longg
r1 = fzero(@(x) exp(-x) - x, [0 2]);
r2 = fzero(@(x) x^3 - 2*x - 5, 2);
r3 = fzero(@(x) sin(x) - 0.5*x, [1 3]);
opts_b = optimset('TolX', 1e-11);
[mx1, mf1] = fminbnd(@(x) exp(x) - 3*x, 0, 3, opts_b);
[mx2, mf2] = fminbnd(@(x) x*sin(x) + 0.25*x^2, -2, 4, opts_b);
opts_s = optimset('TolX', 1e-11, 'TolFun', 1e-11, 'MaxIter', 4000, 'MaxFunEvals', 8000);
[nx1, nf1] = fminsearch(@(x) (1 - x(1))^2 + 100*(x(2) - x(1)^2)^2, [-1.2, 1.0], opts_s);
fprintf('m_r1: %.17g\n', r1);
fprintf('m_r2: %.17g\n', r2);
fprintf('m_r3: %.17g\n', r3);
fprintf('m_m1: %.17g %.17g\n', mx1, mf1);
fprintf('m_m2: %.17g %.17g\n', mx2, mf2);
fprintf('m_nm1: %.17g %.17g %.17g\n', nx1(1), nx1(2), nf1);
"""
    proc = subprocess.run(["matlab", "-batch", script], capture_output=True, text=True, check=True)
    md = parse_dump(proc.stdout)
    check_close("matlab/root_exp_x", dump["root_exp_x"], md["m_r1"], 1e-14, failures)
    check_close("matlab/root_cubic", dump["root_cubic"], md["m_r2"], 1e-14, failures)
    check_close("matlab/root_sin_halfx", dump["root_sin_halfx"], md["m_r3"], 1e-14, failures)
    check_close("matlab/min_exp_3x", dump["min_exp_3x"], md["m_m1"], 1e-7, failures)
    check_close("matlab/min_xsinx", dump["min_xsinx"], md["m_m2"], 1e-7, failures)
    check_close("matlab/nm_rosen", dump["nm_rosen"], md["m_nm1"], 1e-7, failures)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_optim>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    dump = parse_dump(proc.stdout)
    failures: list[str] = []

    print("Running keel::optim reference verification against SciPy/MATLAB...")

    r1 = optimize.brentq(lambda x: np.exp(-x) - x, 0.0, 2.0, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    r2 = optimize.brentq(lambda x: x**3 - 2.0 * x - 5.0, 1.0, 3.0, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    r3 = optimize.brentq(lambda x: np.sin(x) - 0.5 * x, 1.0, 3.0, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    check_close("scipy/root_exp_x", dump["root_exp_x"], [r1], 1e-14, failures)
    check_close("scipy/root_cubic", dump["root_cubic"], [r2], 1e-14, failures)
    check_close("scipy/root_sin_halfx", dump["root_sin_halfx"], [r3], 1e-14, failures)

    m1 = optimize.minimize_scalar(lambda x: np.exp(x) - 3.0 * x, bounds=(0.0, 3.0), method="bounded", options={"xatol": 1e-12})
    m2 = optimize.minimize_scalar(
        lambda x: x * np.sin(x) + 0.25 * x * x, bounds=(-2.0, 4.0), method="bounded", options={"xatol": 1e-12}
    )
    check_close("scipy/min_exp_3x", dump["min_exp_3x"], [m1.x, m1.fun], 1e-7, failures)
    check_close("scipy/min_xsinx", dump["min_xsinx"], [m2.x, m2.fun], 1e-7, failures)

    check_close("exact/nm_rosen", dump["nm_rosen"], [1.0, 1.0, 0.0], 1e-7, failures)
    check_close("exact/nm_quad3d", dump["nm_quad3d"], [2.0, -1.5, 0.75, 0.0], 1e-7, failures)

    run_matlab_checks(dump, failures)

    if failures:
        print(f"\nFAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL keel::optim REFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
