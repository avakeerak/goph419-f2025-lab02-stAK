import numpy as np

from lab_02.linalg_interp import (gauss_iter_solve, spline_function)

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
    
# use if __name__ == "__main__": guard to run tests when this file is executed directly
if __name__ == "__main__":
    test_linear_quadratic_cubic_exact_recovery()
    test_compare_to_univariatespline()

