import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix, bmat
import operators as ops

# Define a range of grid sizes to study
mx_values = [40, 50, 100, 200, 300]

def theta1(x, t):
    return np.exp(-((x-c*t)/0.2)**2)

def theta2(x, t):
    return -np.exp(-((x+c*t)/0.2)**2)

def analytic_N(x, t):
    return 0.5*theta1(x+2, t) - 0.5*theta2(x-2, t)

def analytic_D(x, t):
    return -0.5*theta1(x+2, t) + 0.5*theta2(x-2, t)

def f(x):
    return theta1(x, 0)

# Initialize a variable to store the previous error
error_prev = None

for mx in mx_values:
    c = 1.  # for simplicity
    alphal = 0.
    alphar = 0.
    betal = 1  # Dirichlet BC if it's the only one set to non-zero
    betar = 1  # Dirichlet BC if it's the only one set to non-zero
    gammal = 0  # Neumann BC if it's the only one set to non-zero
    gammar = 0  # Neumann BC if it's the only one set to non-zero

    T = 2  # end time
    xl = -1.
    xr = 1.

    hx = (xr - xl) / (mx - 1)
    xvec = np.linspace(xl, xr, mx)

    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    I2 = eye(2)
    I = eye(2 * mx)

    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    #H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, hx)
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(mx, hx)

    L = vstack([alphal * kron(e2, e_l) + betal * kron(e1, e_l) + gammal * kron(e1, d1_l),
                alphar * kron(e2, e_r) + betar * kron(e1, e_r) + gammar * kron(e1, d1_r)])

    HII = kron(I2, HI)
    P = I - HII @ L.T @ inv(L @ HII @ L.T) @ L

    A = bmat([[None, eye(mx)], [coo_matrix(c**2 * D2), None]])
    B = P @ A @ P

    V = np.zeros(2 * mx)
    V[:mx] = f(xvec)

    cfl_factor = 0.01
    k_try = cfl_factor * hx

    mt = int(np.ceil(T / k_try) * 20 + 1)
    tvec, ht = np.linspace(0, T, mt, retstep=True)
    t = 0

    for tidx in range(mt):
        k1 = ht * B @ V
        k2 = ht * B @ (V + 0.5 * k1)
        k3 = ht * B @ (V + 0.5 * k2)
        k4 = ht * B @ (V + k3)
        V = V + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + ht
        if t > T:
            break

    t = T
    t_final = T

    # Analytic solution at end-time T
    u_analytic = analytic_D(xvec, t_final)

    # Discrete L2 norm of the error
    error = np.sqrt(hx) * np.linalg.norm(u_analytic - V[:mx])

    # Convergence rate calculation
    if error_prev is not None:
        h_ratio = 2.0 / hx  # Assuming h2 = hx and h1 = 2 * hx
        convergence_rate = np.log10(error_prev / error) / np.log10(h_ratio)
        print(f"Convergence rate for mx = {mx}: {convergence_rate}")

    print(f"Error for mx = {mx}: {error}")

    # Store the current error for the next iteration
    error_prev = error
