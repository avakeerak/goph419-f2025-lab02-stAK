
import numpy as np
import warnings

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

    if len(b_shape) == 1:
        b = np.reshape(b, (n, 1))  

    m = b_shape[1]      # number of RHS

    alg_flag = alg.strip().lower()
    if alg_flag not in ("seidel", "jacobi"):
        raise ValueError("alg must be either 'seidel' or 'jacobi'.")

    # --------------------------------------
    # Handle initial guess x0 (None or array)
    # --------------------------------------
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x0 = np.array(x0, dtype=float)

        if len(x0.shape) not in {1, 2}:
            raise ValueError("x0 must be 1D or 2D")

        if x0.shape[0] != n:
            raise ValueError("x0 must have same number of rows as A and b")

        if x0.ndim == 1:
            x = x0.reshape(n, 1).repeat(m, axis=1)
        else:  # 2D
            if x0.shape == (n, m):
                x = x0.copy()
            elif x0.shape == (n, 1):
                x = x0.repeat(m, axis=1)
            else:
                raise ValueError("x0 shape incompatible with b")


    # Precompute diagonal
    diagA = np.diag(A)
    if np.any(diagA == 0):
        raise ValueError("Zero diagonal entry in A — cannot iterate")

    LU = A - np.diagflat(D)

    max_iter = 50_000
    iteration = 0

    # for Jacobi (x_new depends on previous x only)
    x_old = x.copy()

    # ==========================
    #      ITERATIVE LOOP
    # ==========================
    for k in range(max_iter):
        x_old = x.copy()

        if use_jacobi:
            # Jacobi: use old x everywhere
            x = (b - LplusU @ x_old) / D[:, None]

        else:
            # Gauss–Seidel: update in-place using newest values
            for i in range(n):
                # Compute A[i,:]·x skipping diagonal term
                row_sum = A[i, :i] @ x[:i] + A[i, i+1:] @ x_old[i+1:]
                x[i] = (b[i] - row_sum) / A[i, i]

        # ---------------------
        # Check convergence
        # ---------------------
        rel_err = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-14)
        if rel_err < tol:
            return x

        if iteration >= max_iter:
            warnings.warn("Gauss-Seidel/Jacobi did not converge", RuntimeWarning)
            break

    # match user’s b shape
    if len(b_shape) == 1:
        return x.reshape(b_shape)

    return x

def forward_substitution(L, b):
    """ Solve a system Lx = b where L is a lower triangular coefficient matrix
    and b is the right hand side vector, or a matrix where each column is a right hand side vector.
    
    Parameters
    ----------
    L : array_like
        Lower triangular matrix, size = (n, n)
    b : array_like
        Right-hand side(s), size = (n ,) or (n, m)
        where m is the number of right-hand sides.
    Returns
    -------
    numpy.ndarray
        The vector or matrix of solutions x.
        This will have the same shape as b.

    """

        # state in documentation that we are assuming that the system is lower traingular; therefore
        # we do not need to check

    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)

    # check shape of coef matrix (2d nxn) and  b

    L_shape = L.shape
    if not len(L_shape) == 2:
            raise ValueError(f"Coefficient matrix L has shape {L_shape} and length is {len(L_shape)}, length must be 2")

    n = L_shape[0]
    if n != L_shape[1]:
        raise ValueError(f"Coefficient matrix has shape {L_shape}, shape must be square")

    b_shape = b.shape
    if not (len(b_shape) == 1 or len(b_shape) == 2):
        raise ValueError(f"Right-hand side vector b has shape {b_shape} and length is {len(b_shape)}, length must be 1 or 2")

    if n != b_shape[0]:
        raise ValueError(f"Coefficient matrix L has {n} rows and columns, right-hand side vector b has {b_shape[0]} rows, number of rows must match")

    if len(b_shape) == 1:
        b = np.reshape(b, (n, 1))

    # form the augmented matrix
    aug = np.hstack([L, b])

    for k, row in enumerate(L): # loop for every row in L matrix
        aug[k, n:] = (aug[k, n:] - L[k : k + 1, :k] @ aug[:k, n:]) / row[k]  
    


    return np.reshape(aug[::, n], shape=b_shape)