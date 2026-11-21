#include "keel/optim.h"

namespace keel::optim {

BracketResult find_bracket(ScalarFn fn, f64 x0, i32 max_expansions) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0, 0.0};
    if (!is_finite(x0)) return {Status::InvalidInput, 0.0, 0.0};
    const i32 cap = (max_expansions > 0) ? max_expansions : 60;

    f64 step = (x0 == 0.0) ? 0.02 : (abs(x0) * 0.02);
    if (step == 0.0) step = 0.02;

    f64 a = x0;
    f64 b = x0;
    f64 fa = fn(a);
    if (fa == 0.0) return {Status::Ok, a, a};
    f64 fb = fa;

    f64 scale = 1.0;
    for (i32 i = 0; i < cap; ++i) {
        const f64 s = step * scale;
        scale *= 2.0;
        const f64 a_prev = a;
        const f64 fa_prev = fa;
        a = x0 - s;
        b = x0 + s;
        fa = fn(a);
        if (fa == 0.0) return {Status::Ok, a, a};
        fb = fn(b);
        if (fb == 0.0) return {Status::Ok, b, b};
        if ((fa < 0.0) != (fb < 0.0)) return {Status::Ok, a, b};
        if ((fa_prev < 0.0) != (fa < 0.0)) return {Status::Ok, a, a_prev};
    }
    return {Status::NoBracket, a, b};
}

ScalarResult brent_root(ScalarFn fn, f64 a, f64 b, f64 tol, i32 max_iter) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0, 0.0, 0};
    if (!is_finite(a) || !is_finite(b)) return {Status::InvalidInput, 0.0, 0.0, 0};
    const f64 use_tol = (tol > 0.0 && is_finite(tol)) ? tol : 1e-15;
    const i32 cap = (max_iter > 0) ? max_iter : 200;
    constexpr f64 kEps = 1e-15;

    f64 fa = fn(a);
    f64 fb = fn(b);
    if (fa == 0.0) return {Status::Ok, a, 0.0, 0};
    if (fb == 0.0) return {Status::Ok, b, 0.0, 0};
    if ((fa < 0.0) == (fb < 0.0)) return {Status::NoSignChange, b, fb, 0};

    f64 c = a;
    f64 fc = fa;
    f64 d = b - a;
    f64 e = d;

    for (i32 it = 0; it < cap; ++it) {
        if ((fb < 0.0) == (fc < 0.0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (abs(fc) < abs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        const f64 tol1 = 2.0 * kEps * abs(b) + 0.5 * use_tol;
        const f64 xm = 0.5 * (c - b);
        if (abs(xm) <= tol1 || fb == 0.0) {
            return {Status::Ok, b, fb, it};
        }

        if (abs(e) >= tol1 && abs(fa) > abs(fb)) {
            const f64 s = fb / fa;
            f64 p = 0.0;
            f64 q = 0.0;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                const f64 r = fb / fc;
                const f64 sa = fa / fc;
                p = s * (2.0 * xm * sa * (sa - r) - (b - a) * (r - 1.0));
                q = (sa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = abs(p);
            const f64 min1 = 3.0 * xm * q - abs(tol1 * q);
            const f64 min2 = abs(e * q);
            if (2.0 * p < min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (abs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0 ? abs(tol1) : -abs(tol1));
        }
        fb = fn(b);
    }
    return {Status::MaxIterations, b, fb, cap};
}

ScalarResult fzero(ScalarFn fn, f64 x0, f64 tol, i32 max_iter) {
    const BracketResult br = find_bracket(fn, x0);
    if (!br.ok()) return {br.status, x0, 0.0, 0};
    if (br.a == br.b) return {Status::Ok, br.a, 0.0, 0};
    f64 lo = br.a;
    f64 hi = br.b;
    if (lo > hi) swap(lo, hi);
    return brent_root(fn, lo, hi, tol, max_iter);
}

ScalarResult brent_min(ScalarFn fn, f64 a, f64 b, f64 tol, i32 max_iter) {
    if (fn.empty()) return {Status::InvalidCallback, 0.0, 0.0, 0};
    if (!is_finite(a) || !is_finite(b) || a >= b) return {Status::InvalidInput, 0.0, 0.0, 0};
    const f64 use_tol = (tol > 0.0 && is_finite(tol)) ? tol : 1e-6;
    const i32 cap = (max_iter > 0) ? max_iter : 200;
    const f64 gold = 0.5 * (3.0 - sqrt(5.0));
    constexpr f64 kEps = 1e-10;

    f64 x = a + gold * (b - a);
    f64 w = x;
    f64 v = x;
    f64 fx = fn(x);
    f64 fw = fx;
    f64 fv = fx;
    f64 d = 0.0;
    f64 e = 0.0;

    for (i32 it = 0; it < cap; ++it) {
        const f64 m = 0.5 * (a + b);
        const f64 t1 = use_tol * abs(x) + kEps;
        const f64 t2 = 2.0 * t1;
        if (abs(x - m) <= (t2 - 0.5 * (b - a))) {
            return {Status::Ok, x, fx, it};
        }

        bool use_gold = true;
        if (abs(e) > t1) {
            const f64 r1 = (x - w) * (fx - fv);
            f64 q = (x - v) * (fx - fw);
            f64 p = (x - v) * q - (x - w) * r1;
            q = 2.0 * (q - r1);
            if (q > 0.0) p = -p;
            q = abs(q);
            const f64 etemp = e;
            e = d;
            if (abs(p) < abs(0.5 * q * etemp) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const f64 u = x + d;
                if ((u - a) < t2 || (b - u) < t2) d = (m >= x ? t1 : -t1);
                use_gold = false;
            }
        }
        if (use_gold) {
            e = (x >= m ? (a - x) : (b - x));
            d = gold * e;
        }
        const f64 u = (abs(d) >= t1) ? (x + d) : (x + (d >= 0.0 ? t1 : -t1));
        const f64 fu = fn(u);

        if (fu <= fx) {
            if (u >= x) {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if (u < x) {
                a = u;
            } else {
                b = u;
            }
            if (fu <= fw || w == x) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    return {Status::Ok, x, fx, cap};
}

VectorResult nelder_mead(VectorFn fn, ArrayRef<f64> x0, f64 tol, i32 max_iter, Allocator* alloc) {
    VectorResult res{Status::Ok, Array<f64>(alloc), 0.0, 0};
    if (fn.empty()) {
        res.status = Status::InvalidCallback;
        return res;
    }
    const usize n = x0.size();
    if (n == 0) {
        res.status = Status::InvalidInput;
        return res;
    }
    const f64 use_tol = (tol > 0.0 && is_finite(tol)) ? tol : 1e-4;
    const i32 cap = (max_iter > 0) ? max_iter : 500;

    constexpr f64 kAlpha = 1.0; // reflection
    constexpr f64 kGamma = 2.0; // expansion
    constexpr f64 kRho = 0.5;   // contraction
    constexpr f64 kSigma = 0.5; // shrink

    auto eval_at = [&](const f64* ptr) -> f64 { return fn(ArrayRef<f64>(ptr, n)); };

    Array<f64> sx(alloc);
    sx.resize((n + 1) * n);
    Array<f64> fv(alloc);
    fv.resize(n + 1);

    for (usize j = 0; j < n; ++j) sx[j] = x0[j];
    fv[0] = eval_at(sx.data());
    for (usize i = 1; i <= n; ++i) {
        for (usize j = 0; j < n; ++j) sx[i * n + j] = x0[j];
        const f64 xi = x0[i - 1];
        sx[i * n + (i - 1)] = (xi != 0.0) ? (1.05 * xi) : 0.00025;
        fv[i] = eval_at(sx.data() + i * n);
    }

    Array<usize> ord(alloc);
    ord.resize(n + 1);
    Array<f64> new_sx(alloc);
    new_sx.resize((n + 1) * n);
    Array<f64> new_fv(alloc);
    new_fv.resize(n + 1);
    Array<f64> centroid(alloc);
    centroid.resize(n);
    Array<f64> xr(alloc);
    xr.resize(n);
    Array<f64> xe(alloc);
    xe.resize(n);
    Array<f64> xc(alloc);
    xc.resize(n);

    i32 iters = 0;
    for (i32 it = 0; it < cap; ++it) {
        iters = it;
        // Sort simplex vertices ascending by fv (insertion sort on n + 1 indices).
        for (usize i = 0; i <= n; ++i) ord[i] = i;
        for (usize i = 1; i <= n; ++i) {
            const usize key = ord[i];
            const f64 key_val = fv[key];
            usize j = i;
            while (j > 0 && key_val < fv[ord[j - 1]]) {
                ord[j] = ord[j - 1];
                --j;
            }
            ord[j] = key;
        }

        for (usize i = 0; i <= n; ++i) {
            const f64* src = sx.data() + ord[i] * n;
            f64* dst = new_sx.data() + i * n;
            for (usize j = 0; j < n; ++j) dst[j] = src[j];
            new_fv[i] = fv[ord[i]];
        }
        swap(sx, new_sx);
        swap(fv, new_fv);

        const f64 fspread = fv[n] - fv[0];
        f64 xspread = 0.0;
        for (usize i = 1; i <= n; ++i) {
            for (usize j = 0; j < n; ++j) {
                xspread = max(xspread, abs(sx[i * n + j] - sx[j]));
            }
        }
        if (fspread <= use_tol && xspread <= use_tol) break;

        for (usize j = 0; j < n; ++j) centroid[j] = 0.0;
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) centroid[j] += sx[i * n + j];
        }
        const f64 inv_n = 1.0 / static_cast<f64>(n);
        for (usize j = 0; j < n; ++j) centroid[j] *= inv_n;

        for (usize j = 0; j < n; ++j) {
            xr[j] = centroid[j] + kAlpha * (centroid[j] - sx[n * n + j]);
        }
        const f64 fxr = eval_at(xr.data());

        if (fxr < fv[0]) {
            for (usize j = 0; j < n; ++j) xe[j] = centroid[j] + kGamma * (xr[j] - centroid[j]);
            const f64 fxe = eval_at(xe.data());
            if (fxe < fxr) {
                for (usize j = 0; j < n; ++j) sx[n * n + j] = xe[j];
                fv[n] = fxe;
            } else {
                for (usize j = 0; j < n; ++j) sx[n * n + j] = xr[j];
                fv[n] = fxr;
            }
        } else if (fxr < fv[n - 1]) {
            for (usize j = 0; j < n; ++j) sx[n * n + j] = xr[j];
            fv[n] = fxr;
        } else {
            const bool outside = fxr < fv[n];
            const f64* base = outside ? xr.data() : (sx.data() + n * n);
            for (usize j = 0; j < n; ++j) xc[j] = centroid[j] + kRho * (base[j] - centroid[j]);
            const f64 fxc = eval_at(xc.data());
            const f64 fcompare = outside ? fxr : fv[n];
            if (fxc <= fcompare) {
                for (usize j = 0; j < n; ++j) sx[n * n + j] = xc[j];
                fv[n] = fxc;
            } else {
                for (usize i = 1; i <= n; ++i) {
                    for (usize j = 0; j < n; ++j) {
                        sx[i * n + j] = sx[j] + kSigma * (sx[i * n + j] - sx[j]);
                    }
                    fv[i] = eval_at(sx.data() + i * n);
                }
            }
        }
    }

    res.x.resize(n);
    for (usize j = 0; j < n; ++j) res.x[j] = sx[j];
    res.fval = fv[0];
    res.iterations = iters;
    return res;
}

} // namespace keel::optim
