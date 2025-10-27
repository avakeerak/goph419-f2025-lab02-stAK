import numpy as np
from lab_02.linalg_interp import forward_substitution()

def main():
    test_forward_substitution()

def test_forward_substitution():
    print("Testing forward substitution...")
    # lower traiangular coefficient matrix
    A = np.array([
        [2135.0, 0.0, 0.0, 0.0,]
        [-2135.0, 5200.0, 0.0, 0.0,]
        [0.0, -5200.0, 5796.0, 0.0,]
        [0.0, 0.0, -5796.0, 7060.0,]])
    
    print("A=")
    print(A)
    # right-hand side vector

    b = np.array([500.0, 700.0, 1000.0, 500.0])
    print("b=")
    print(b)

    # solve  using numpy.linalg.solve()

    x_exp = np.linalg.solve(A, b)
    print(f"x expected = {x_exp}")
    print(f"expected shape: {x_exp.shape}")

    x_act = forward_substitution(A, b)
    print(f"x actual = {x_act}")
    print(f"actual shape: {x_act.shape}")

if __name__ == "__main__": #
    main()