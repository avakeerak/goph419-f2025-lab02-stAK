
import numpy as np

def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel'):
    """
    Solve Ax = b using Gauss-Seidel or Jacobi iterative methods.

    Parameters
    ----------
    A : array_like
        Coefficient matrix (n x n).
    b : array_like
        RHS vector (n,) or matrix (n, m).
    x0 : array_like or None
        Initial guess (same shape as b) or (n,1) to be repeated.
    tol : float
        Relative error tolerance for convergence.
    alg : str
        the algorithm to be used, defaults to Gauss-Seidel Method if 'jacobi' is not specified.

    Returns
    -------
    numpy.ndarray
        Solution array with same shape as b.
    """

    # Convert inputs to arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # -----------------------
    # Shape / validity checks
    # -----------------------
    A_shape = A.shape
    if len(A_shape) != 2:
        raise ValueError(f"Coefficient matrix A has shape {A_shape}, must be 2D")

    n = A_shape[0]
    if n != A_shape[1]:
        raise ValueError(f"Coefficient matrix A must be square, got {A_shape}")

    b_shape = b.shape
    if len(b_shape) not in {1, 2}:
        raise ValueError("b must be 1D or 2D array")

    if b_shape[0] != n:
        raise ValueError(f"A has {n} rows but b has {b_shape[0]} rows")

    if b.ndim == 1:
        b = b.reshape(n, 1)          # number of RHS
    
    m = b.shape[1]

    alg_flag = alg.strip().lower()
    if alg_flag not in ("seidel", "jacobi"):
        raise ValueError("alg must be 'seidel' or 'jacobi'")
    use_jacobi = (alg_flag == "jacobi")

    # --- Handle x0 ---
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x0 = np.array(x0, dtype=float)

        if x0.ndim == 1:
            if x0.shape[0] != n:
                raise ValueError("x0 has wrong number of rows")
            x = np.tile(x0.reshape(n, 1), (1, m))
        else:
            if x0.shape == (n, m):
                x = x0.copy()
            elif x0.shape == (n, 1):
                x = np.tile(x0, (1, m))
            else:
                raise ValueError("x0 shape incompatible with b")

    # --- Precompute components ---
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Zero diagonal in A")

    LU = A - np.diagflat(D)

    max_iter = 50000

    iteration = 1

    for k in range(max_iter):
        x_old = x.copy()

        if use_jacobi:
            x = (b - LU @ x_old) / D[:, None]
        else:  # Gauss–Seidel
            for i in range(n):
                row_sum = A[i, :i] @ x[:i] + A[i, i+1:] @ x_old[i+1:]
                x[i] = (b[i] - row_sum) / A[i, i]

        # convergence check
        rel_err = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-14)
        if rel_err < tol:
            return x

        if iteration >= max_iter:
            print("Gauss-Seidel/Jacobi did not converge")
            break
        else:
            iteration += 1

    # match user’s b shape
    if len(b_shape) == 1:
        return x.reshape(b_shape)

    return x

def spline_function(xd, yd, order=3):
    """
    Generate a spline interpolation function of order 1, 2, or 3.
    Uses custom Gauss–Seidel/Jacobi solver for all linear solves.

    Supports:
        order = 1 → piecewise linear
        order = 2 → piecewise quadratic
        order = 3 → cubic spline (NOT-A-KNOT, SciPy-compatible)
    """

    # ---- Validation ----
    xd = np.asarray(xd, dtype=float).flatten()
    yd = np.asarray(yd, dtype=float).flatten()

    if xd.size != yd.size:
        raise ValueError("xd and yd must have the same number of elements.")

    if np.unique(xd).size != xd.size:
        raise ValueError("xd cannot contain repeated values.")

    if not np.all(np.sort(xd) == xd):
        raise ValueError("xd must be strictly increasing.")

    if order not in (1, 2, 3):
        raise ValueError("order must be one of {1, 2, 3}.")

    n = len(xd)

    # ======================================================================
    # ORDER 1: Piecewise Linear
    # ======================================================================
    if order == 1:
        def f_linear(x):
            x = np.asarray(x)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError("Extrapolation not allowed.")

            idx = np.searchsorted(xd, x, side='right') - 1
            idx = np.clip(idx, 0, n-2)

            x0, x1 = xd[idx], xd[idx+1]
            y0, y1 = yd[idx], yd[idx+1]

            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        return f_linear

    # ======================================================================
    # ORDER 2: Piecewise Quadratic (local least polynomial)
    # ======================================================================
    if order == 2:

        coeffs = []

        for i in range(n - 2):
            x0, x1, x2 = xd[i], xd[i+1], xd[i+2]
            y0, y1, y2 = yd[i], yd[i+1], yd[i+2]

            A = np.array([
                [x0**2, x0, 1],
                [x1**2, x1, 1],
                [x2**2, x2, 1]
            ])

            bvec = np.array([[y0], [y1], [y2]])

            abc = gauss_iter_solve(A, bvec, alg="seidel")
            coeffs.append(abc[:,0])

        coeffs = np.array(coeffs)

        def f_quad(x):
            x = np.asarray(x)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError("Extrapolation not allowed.")

            idx = np.searchsorted(xd, x) - 1
            idx = np.clip(idx, 0, n-3)

            a = coeffs[idx, 0]
            b = coeffs[idx, 1]
            c = coeffs[idx, 2]

            return a*x**2 + b*x + c

        return f_quad

    # ======================================================================
    # ORDER 3: Cubic Spline – NOT-A-KNOT (SciPy-compatible!)
    # ======================================================================
    if order == 3:

        h = np.diff(xd)
        d = np.diff(yd) / h

        # Build system A m = b for second derivatives m
        A = np.zeros((n, n))
        bvec = np.zeros((n, 1))

        # ---------------------------------------------------------
        # NOT-A-KNOT LEFT BOUNDARY
        #     h1*m0 - (h0 + h1)*m1 + h0*m2 = 0
        # ---------------------------------------------------------
        A[0,0] =  h[1]
        A[0,1] = -(h[0] + h[1])
        A[0,2] =  h[0]

        # ---------------------------------------------------------
        # INTERIOR EQUATIONS
        # ---------------------------------------------------------
        for i in range(1, n-1):
            A[i,i-1] = h[i-1]
            A[i,i]   = 2*(h[i-1] + h[i])
            A[i,i+1] = h[i]
            bvec[i,0] = 6*(d[i] - d[i-1]) if i < n-1 else 0.0

        # ---------------------------------------------------------
        # NOT-A-KNOT RIGHT BOUNDARY
        #     h[n-2]*m[n-1] - (h[n-3]+h[n-2])*m[n-2] + h[n-3]*m[n-3] = 0
        # ---------------------------------------------------------
        A[-1,-1] =  h[-2]
        A[-1,-2] = -(h[-3] + h[-2])
        A[-1,-3] =  h[-3]

        # Solve using YOUR Gauss–Seidel solver
        m = gauss_iter_solve(A, bvec, alg="seidel")[:,0]

        def f_cubic(x):
            x = np.asarray(x)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError("Extrapolation not allowed.")

            idx = np.searchsorted(xd, x, side="right") - 1
            idx = np.clip(idx, 0, n-2)

            x0 = xd[idx]
            x1 = xd[idx+1]
            y0 = yd[idx]
            y1 = yd[idx+1]
            h_i = x1 - x0

            m0 = m[idx]
            m1 = m[idx+1]

            t = (x1 - x) / h_i
            u = (x - x0) / h_i

            return (
                m0*(t**3)*h_i**2/6 +
                m1*(u**3)*h_i**2/6 +
                (y0 - m0*h_i**2/6)*t +
                (y1 - m1*h_i**2/6)*u
            )

        return f_cubic




import numpy as np
from scipy.interpolate import UnivariateSpline

# -------------------------
# Helper generators
# -------------------------
def midpoints(x):
    """Return midpoints between consecutive entries of x."""
    return 0.5 * (x[:-1] + x[1:])

# -------------------------
# 1) Exact recovery tests
# -------------------------
def test_linear_quadratic_cubic_exact_recovery():
    x = np.linspace(0, 10, 25)      # knots
    x_dense = np.linspace(x[0], x[-1], 1000)   # dense grid for exact-match checks
    x_mids = midpoints(x)           # points strictly between knots for 'should NOT match'

    # ----- Linear -----
    y_linear = 3 * x - 2
    f_lin = spline_function(x, y_linear, order=1)
    # linear spline should reproduce linear data everywhere
    if not np.allclose(f_lin(x_dense), 3*x_dense - 2, atol=1e-12):
        raise ValueError("Order=1 spline did not recover linear data exactly.")

    # higher-order splines should also reproduce linear exactly
    for ord_ in (2, 3):
        f_h = spline_function(x, y_linear, order=ord_)
        if not np.allclose(f_h(x_dense), 3*x_dense - 2, atol=1e-12):
            raise ValueError(f"Order={ord_} spline did not recover linear data exactly.")

    # ----- Quadratic -----
    y_quad = x**2 - 4*x + 1

    # order=1 should NOT match quadratic at midpoints (interior points)
    f1 = spline_function(x, y_quad, order=1)
    err1 = np.max(np.abs(f1(x_mids) - (x_mids**2 - 4*x_mids + 1)))
    # require a non-negligible difference (not exact). Machine-eps-level matches are OK,
    # but we want to catch "exact interpolation" at knots only; midpoints should differ.
    if err1 < 1e-12:
        raise ValueError("Order=1 spline unexpectedly matched quadratic data exactly at interior points.")

    # order=2 should match quadratic everywhere (dense grid)
    f2 = spline_function(x, y_quad, order=2)
    if not np.allclose(f2(x_dense), x_dense**2 - 4*x_dense + 1, atol=1e-12):
        raise ValueError("Order=2 spline did not recover quadratic data exactly.")

    # order=3 should also reproduce quadratic exactly
    f3 = spline_function(x, y_quad, order=3)
    if not np.allclose(f3(x_dense), x_dense**2 - 4*x_dense + 1, atol=1e-12):
        raise ValueError("Order=3 spline did not recover quadratic data exactly.")

    # ----- Cubic -----
    y_cubic = 2*x**3 - 5*x**2 + x - 7

    # order=1 should NOT match cubic at midpoints
    f1c = spline_function(x, y_cubic, order=1)
    err1c = np.max(np.abs(f1c(x_mids) - (2*x_mids**3 - 5*x_mids**2 + x_mids - 7)))
    if err1c < 1e-12:
        raise ValueError("Order=1 spline unexpectedly matched cubic data exactly at interior points.")

    # order=2 should NOT match cubic at midpoints
    f2c = spline_function(x, y_cubic, order=2)
    err2c = np.max(np.abs(f2c(x_mids) - (2*x_mids**3 - 5*x_mids**2 + x_mids - 7)))
    if err2c < 1e-12:
        raise ValueError("Order=2 spline unexpectedly matched cubic data exactly at interior points.")

    # order=3 should match cubic everywhere
    f3c = spline_function(x, y_cubic, order=3)
    if not np.allclose(f3c(x_dense), 2*x_dense**3 - 5*x_dense**2 + x_dense - 7, atol=1e-12):
        raise ValueError("Order=3 spline did not recover cubic data exactly.")


# -------------------------
# 2) Compare with UnivariateSpline (k=3, s=0, ext='raise')
# -------------------------
def test_compare_to_univariatespline():
    x = np.linspace(0, 5, 40)
    # test points — avoid exact knots so we compare interpolation, not trivial knot-equality
    x_test = np.linspace(x[0] + 1e-6, x[-1] - 1e-6, 200)

    # --- Cubic polynomial (degree 3) — should be reproduced exactly by a cubic spline ---
    y_cubic = 2*x**3 - 5*x**2 + x - 7
    f_custom = spline_function(x, y_cubic, order=3)
    f_scipy = UnivariateSpline(x, y_cubic, k=3, s=0, ext='raise')

    y_custom = f_custom(x_test)
    y_scipy = f_scipy(x_test)
    if not np.allclose(y_custom, y_scipy, atol=1e-12, rtol=1e-12):
        err = np.max(np.abs(y_custom - y_scipy))
        raise ValueError(f"Custom spline did not match UnivariateSpline for cubic polynomial. max error = {err:.2e}")

    # --- Exponential (smooth) — approximate; compare within reasonable tolerance ---
    y_exp = np.exp(0.4 * x)
    f_custom2 = spline_function(x, y_exp, order=3)
    f_scipy2 = UnivariateSpline(x, y_exp, k=3, s=0, ext='raise')

    y_custom2 = f_custom2(x_test)
    y_scipy2 = f_scipy2(x_test)
    if not np.allclose(y_custom2, y_scipy2, atol=1e-6, rtol=1e-6):
        err2 = np.max(np.abs(y_custom2 - y_scipy2))
        raise ValueError(f"Custom spline did not match UnivariateSpline for exponential. max error = {err2:.2e}")


test_linear_quadratic_cubic_exact_recovery()
test_compare_to_univariatespline()