
def gauss_iter_solve(A, b, x0=None, tol=1e-10, alg='jacobi'):
    """
    Solve the linear system Ax = b using iterative methods: Jacobi or Gauss-Seidel.

    Parameters:
    A : 2D array-like
        Coefficient matrix.
    b : 1D array-like
        Right-hand side vector.
    x0 : 1D array-like, optional
        Initial guess for the solution. If None, a zero vector is used.
    tol : float, optional
        Tolerance for convergence. Default is 1e-10.
    alg : str, optional
        Algorithm to use: 'jacobi' or 'gauss-seidel'. Default is 'jacobi'.

    Returns:
    x : 1D array
        Approximate solution vector.
    """
    import numpy as np

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if x0 is None:
        x0 = np.zeros(n)

    x = np.copy(x0)
    max_iterations = 10000

    for iteration in range(max_iterations):
        x_new = np.copy(x)

        for i in range(n):
            sum_ax = np.dot(A[i, :], x_new) - A[i, i] * x_new[i]
            if alg == 'jacobi':
                x_new[i] = (b[i] - sum_ax) / A[i, i]
            elif alg == 'gauss-seidel':
                x_new[i] = (b[i] - sum_ax) / A[i, i]
            else:
                raise ValueError("Algorithm must be 'jacobi' or 'gauss-seidel'.")

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    raise ValueError("Maximum iterations reached without convergence.")

def forward_substitution(L, b):
    """ Solve a system Lx = b where L is a lower triangular coefficient matrix
    and b is the right hand side vector, or a matrix where each column is a right hand side vector.
    
    Parameters
    ----------
    L : array_like
        Lower triangular matrix, size = (n, n)
    b : array_like
        Right-hand side(s), size = (, n) or (n, m)
        where m is the number of right-hand sides.
    Returns
    -------
    numpy.ndarray
        The vector or matrix of solutions x.
        This will have the same shape as b.

    """

        # state in documentation that we are assuming that the system is lower traingular; therefore
        # we do not need to check

    L = np.array(L)
    b = np.array(b)