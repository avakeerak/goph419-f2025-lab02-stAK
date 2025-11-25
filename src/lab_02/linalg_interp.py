
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel'):
    """
    Solve a linear system Ax = b using either Gauss–Seidel or Jacobi iteration.

    Parameters
    ----------
    A : array_like (n, n)
        Coefficient matrix for the system Ax = b. Must be square.
    b : array_like (n,) or (n, m)
        Right-hand-side vector(s). If 2D, each column is treated as a
        separate RHS vector.
    x0 : array_like or None
        Initial guess for x. If None, a zero vector is used.
    tol : float
        Relative error tolerance for convergence. Iteration stops once
        ||x_{k+1} - x_k|| / ||x_{k+1}|| < tol.
    alg : str
        Iterative method to use. Must be either 'seidel' (Gauss–Seidel)
        or 'jacobi'. Default is 'seidel'.

    Returns
    -------
    x : ndarray
        Approximate solution array with the same shape as b.
    """

    # convert inputs to arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # validate shapes and dimensions
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

    # reshape b from 1D to 2D if needed
    if b.ndim == 1:
        b = b.reshape(n, 1)

    m = b.shape[1]

    # select algorithm
    alg_flag = alg.strip().lower()
    if alg_flag not in ("seidel", "jacobi"):
        raise ValueError("alg must be 'seidel' or 'jacobi'")
    use_jacobi = (alg_flag == "jacobi")

    # handle initial guess x0
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

    # Repair zero diagonal entries by row swapping
    for i in range(n):
        if A[i, i] == 0:
            # Find a row below with a nonzero entry in the same column
            for j in range(i + 1, n):
                if A[j, i] != 0:
                    # Swap rows i and j in both A and b
                    A[[i, j]] = A[[j, i]]
                    b[[i, j]] = b[[j, i]]
                    break
            else:
                raise ValueError(
                    f"Cannot repair zero diagonal at row {i}; "
                    "matrix is singular at this column."
                )

    # Recompute diagonal after swaps
    D = np.diag(A)
    # LU stores everything except diagonal
    LU = A - np.diagflat(D)

    max_iter = 50000

    # Main iteration loop
    for k in range(max_iter):
        x_old = x.copy()

        if use_jacobi:
            # Vectorized Jacobi update
            x = (b - LU @ x_old) / D[:, None]

        else:  # Gauss–Seidel
            for i in range(n):
                # use updated x for j<i, old x for j>i
                row_sum = A[i, :i] @ x[:i] + A[i, i+1:] @ x_old[i+1:]
                x[i] = (b[i] - row_sum) / A[i, i]

        # convergence check
        rel_err = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-14)
        if rel_err < tol:
            # Match user shape
            if len(b_shape) == 1:
                return x.reshape(b_shape)
            return x

    print("Gauss-Seidel/Jacobi did not converge")

    # return last iterate
    if len(b_shape) == 1:
        return x.reshape(b_shape)
    return x

def spline_function(xd, yd, order=3):
    """
    Construct a spline interpolation function of order 1, 2, or 3.

    Parameters
    ----------
    xd : array_like
        Strictly increasing x-values (knots).
    yd : array_like
        Corresponding y-values. Must match xd in length.
    order : int
        Spline order:
            1 → Piecewise linear
            2 → Piecewise quadratic (local polynomial fits)
            3 → Cubic NOT-A-KNOT spline (matches SciPy's behavior)

    Returns
    -------
    f : callable
        A function f(x) that evaluates the spline at any x within [xd[0], xd[-1]].
    """

    # validate inputs
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

    # compute spline coefficients based on order
    if order == 1:
        def f_linear(x):
            x = np.asarray(x)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError("Extrapolation not allowed.")

            # identify which interval each x falls into
            idx = np.searchsorted(xd, x, side='right') - 1
            idx = np.clip(idx, 0, n-2)

            x0, x1 = xd[idx], xd[idx+1]
            y0, y1 = yd[idx], yd[idx+1]

            # linear interpolation formula
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        return f_linear

    if order == 2:

        # For each triple (xd[i], xd[i+1], xd[i+2]) fit a quadratic
        # y = a x^2 + b x + c exactly through the three points.
        coeffs = []

        for i in range(n - 2):
            x0, x1, x2 = xd[i],   xd[i+1],   xd[i+2]
            y0, y1, y2 = yd[i],   yd[i+1],   yd[i+2]

            A = np.array([
                [x0**2, x0, 1.0],
                [x1**2, x1, 1.0],
                [x2**2, x2, 1.0]
            ], dtype=float)

            bvec = np.array([y0, y1, y2], dtype=float)

            # Direct solve (3x3 system) – stable and exact for polynomials
            abc = np.linalg.solve(A, bvec)
            coeffs.append(abc)

        coeffs = np.array(coeffs)   # shape (n-2, 3) with columns a, b, c

        def f_quad(x):
            x = np.asarray(x)
            if np.any((x < xd[0]) | (x > xd[-1])):
                raise ValueError("Extrapolation not allowed.")

            # choose which local quadratic to use
            idx = np.searchsorted(xd, x) - 1
            idx = np.clip(idx, 0, n-3)

            a = coeffs[idx, 0]
            b = coeffs[idx, 1]
            c = coeffs[idx, 2]

            return a*x**2 + b*x + c

        return f_quad

    if order == 3:

        h = np.diff(xd)          # interval lengths
        d = np.diff(yd) / h      # first divided differences

        # Build system A m = b for second derivatives m at each knot.
        A = np.zeros((n, n), dtype=float)
        bvec = np.zeros(n, dtype=float)

        # compute the not-a-knot left boundary
        A[0, 0] =  h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] =  h[0]

        # compute interior equations
        for i in range(1, n-1):
            if i < n-1:
                A[i, i-1] = h[i-1]
                A[i, i]   = 2.0 * (h[i-1] + h[i])
                A[i, i+1] = h[i]
                bvec[i]   = 6.0 * (d[i] - d[i-1]) if i < n-1 else 0.0

        # compute the not-a-knot right boundary
        A[-1, -1] =  h[-2]
        A[-1, -2] = -(h[-3] + h[-2])
        A[-1, -3] =  h[-3]

        # Solve for m using a direct solver
        m = np.linalg.solve(A, bvec)

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

            # Cubic Hermite form written in terms of second derivatives m
            t = (x1 - x) / h_i
            u = (x - x0) / h_i

            return (
                m0*(t**3)*h_i**2/6.0 +
                m1*(u**3)*h_i**2/6.0 +
                (y0 - m0*h_i**2/6.0)*t +
                (y1 - m1*h_i**2/6.0)*u
            )

        return f_cubic




def midpoints(x):
    """Return midpoints between consecutive entries of x."""
    return 0.5 * (x[:-1] + x[1:])


def test_linear_quadratic_cubic_exact_recovery():
    """
    Test that spline_function() recovers linear, quadratic,
    and cubic functions when appropriate.

    • Order 1 recovers linear exactly but NOT quadratic or cubic.
    • Order 2 recovers quadratic but not cubic.
    • Order 3 recovers cubic.
    """
    x = np.linspace(0, 10, 25)      # define knots
    x_dense = np.linspace(x[0], x[-1], 1000)   # dense grid for exact-match checks
    x_mids = midpoints(x)           # points strictly between knots for 'should NOT match'

    # test linear spline
    y_linear = 3 * x - 2
    f_lin = spline_function(x, y_linear, order=1)
    # linear spline should reproduce linear data everywhere
    if not np.allclose(f_lin(x_dense), 3*x_dense - 2, atol=1e-12):
        raise ValueError("Order=1 spline did not recover linear data exactly.")

    # test 2nd and 3rd order splines on linear data
    for ord_ in (2, 3):
        f_h = spline_function(x, y_linear, order=ord_)
        if not np.allclose(f_h(x_dense), 3*x_dense - 2, atol=1e-12):
            raise ValueError(f"Order={ord_} spline did not recover linear data exactly.")

    # test quadratic recovery
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

    # test cubic recovery
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


def test_compare_to_univariatespline():
    """
    Verify that the cubic spline constructed using the not-a-knot
    formulation matches SciPy's UnivariateSpline for both a cubic
    polynomial and an exponential function.
    """
    x = np.linspace(0, 5, 40)
    x_test = np.linspace(x[0] + 1e-6, x[-1] - 1e-6, 200)

    # define cubic polynomial
    y_cubic = 2*x**3 - 5*x**2 + x - 7
    f_custom = spline_function(x, y_cubic, order=3)

    # SciPy spline with matching behaviour
    f_scipy = UnivariateSpline(x, y_cubic, k=3, s=0, ext='raise')

    y_custom = f_custom(x_test)
    y_scipy  = f_scipy(x_test)

    if not np.allclose(y_custom, y_scipy, atol=1e-12, rtol=1e-12):
        err = np.max(np.abs(y_custom - y_scipy))
        raise ValueError(
            f"Custom spline did not match UnivariateSpline for cubic polynomial. "
            f"max error = {err:.2e}"
        )

    # test exponential function
    y_exp = np.exp(0.4 * x)
    f_custom2 = spline_function(x, y_exp, order=3)
    f_scipy2  = UnivariateSpline(x, y_exp, k=3, s=0, ext='raise')

    y_custom2 = f_custom2(x_test)
    y_scipy2  = f_scipy2(x_test)

    if not np.allclose(y_custom2, y_scipy2, atol=1e-6, rtol=1e-6):
        err2 = np.max(np.abs(y_custom2 - y_scipy2))
        raise ValueError(
            f"Custom spline did not match UnivariateSpline for exponential. "
            f"max error = {err2:.2e}"
        )


# run both tests
test_linear_quadratic_cubic_exact_recovery()
test_compare_to_univariatespline()

# load data for splines
air_file = "air_density_vs_temp_eng_toolbox.txt"
water_file   = "water_density_vs_temp_usgs.txt"

water_data = np.loadtxt(water_file)
air_data   = np.loadtxt(air_file)

Tw, rho_w = water_data[:, 0], water_data[:, 1]
Ta, rho_a = air_data[:, 0], air_data[:, 1]



orders = [1, 2, 3]

# compute splines
water_splines = {order: spline_function(Tw, rho_w, order) for order in orders}
air_splines   = {order: spline_function(Ta, rho_a, order) for order in orders}


# Evaluate each spline at 100 temperatures across the domain
Tw_eval = np.linspace(Tw.min(), Tw.max(), 100)
Ta_eval = np.linspace(Ta.min(), Ta.max(), 100)

water_interp = {order: water_splines[order](Tw_eval) for order in orders}
air_interp   = {order: air_splines[order](Ta_eval) for order in orders}


# plot the resulting spline functions
fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

orders = [1, 2, 3]

for i, order in enumerate(orders):

    # plot splines for water density
    ax = axes[i, 0]
    ax.scatter(Tw, rho_w, color="black", s=30, label="Data")
    ax.plot(Tw_eval, water_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Water Density (Order {order})", fontsize=12)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density (g/cm³)")
    ax.grid(True)
    ax.legend(loc="best")

    # plot splines for air density
    ax = axes[i, 1]
    ax.scatter(Ta, rho_a, color="black", s=30, label="Data")
    ax.plot(Ta_eval, air_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Air Density (Order {order})", fontsize=12)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density (kg/m³)")
    ax.grid(True)
    ax.legend(loc="best")

# save plots as a single file 
plt.savefig("spline_interpolations.png", dpi=300)
plt.show()

