import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import rungekutta4 as rk4
from math import ceil

# List of grid sizes to study
mx_values = [34, 50, 100, 200, 300]

# Order of accuracy
order = 6
# Left and right boundary
xl = -1.
xr = 1.

# Initialize a variable to store the previous error
error_prev = None

for mx in mx_values:
    # Number of steps
    hx = (xr - xl) / (mx - 1)
    xvec = np.linspace(xl, xr, mx)

    #end times 
    T = 1
    CFL = 0.01
    ht_try = CFL * hx
    mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)
    #time step 
    k = 0.01
    method = 1  # projection
    # method=0 # SAT
    #identity matrix 
    I = np.identity(mx)
    zero_matrix = np.zeros((mx, mx))
    # boundary conditions (based on assignment 1 with dirichlet )
    alpha_l = 0.
    alpha_r = 0.
    beta_l = 1  
    beta_r = 1  
    gamma_l = 0  
    gamma_r = 0  

    # Analytic solutions assuming the constant c=1
    def theta1(x, t):
        return np.exp(-((x - t) / 0.2)**2)

    def theta2(x, t):
        return -np.exp(-((x + t) / 0.2)**2)

    #H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_6th(mx, hx)
    #H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, hx)
    H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_4th(mx, hx)

    # if method==1:
    L = vstack([beta_l * dl_l,
                beta_l * dl_r])
    tau_l = 1
    tau_r = -1
    SAT_l = tau_l * HI @ (e_l.T @ dl_l)
    SAT_r = tau_r * HI @ (e_r.T @ dl_r)
    D = D2 + SAT_l + SAT_r

    eigD = np.linalg.eigvals(D.toarray())
    #print(D)
    W = np.block([
        [zero_matrix, I],
        [D.toarray(), zero_matrix]])

    def f(x):
        return theta1(x, 0)

    def rhs(x):
        return W @ x

    v = np.hstack((f(xvec), np.zeros(mx)))
    v = v.reshape(-1, 1)

    t = 0

    for tidx in range(mt - 1):

        k1 = ht * rhs(v)
        k2 = ht * rhs(v + 0.5 * k1)
        k3 = ht * rhs(v + 0.5 * k2)
        k4 = ht * rhs(v + k3)
        v = v + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + ht
        if t > T:
            break

    # Analytic solution at end-time T
    u_analytic = theta2(xvec, T)

    # Discrete L2 norm of the error
    error = np.sqrt(hx) * np.linalg.norm(u_analytic - v[0:mx].flatten())

    # Convergence rate calculation
    if error_prev is not None:
        h_ratio = 2.0 / hx  # Assuming h2 = hx and h1 = 2 * hx
        convergence_rate = np.log10(error_prev / error) / np.log10(h_ratio)
        print(f"Convergence rate for mx = {mx}: {convergence_rate}")

    print(f"Error for mx = {mx}: {error}")

    # Store the current error for the next iteration
    error_prev = error
