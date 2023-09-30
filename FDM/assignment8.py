import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack, bmat,coo_matrix

from math import sqrt, ceil
import numpy as np
from scipy.sparse.linalg import inv
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15.
mx = 101
n = 2*mx


# Model parameters
# We assume c=1
T = 11  # end times
xl = -1.
xr = 1.
yl = 1.
yr = -1.

# Initial data
def f(x, y):
    return np.exp(-100*(x**2 + y**2))


# Space discretization
h = (xr - xl) / (mx-1)
xvec = np.linspace(xl, xr, mx)
yvec = np.linspace(yl, yr, mx)
X, Y = np.meshgrid(xvec, yvec)
e1 = np.array([1, 0])
e2 = np.array([0, 1])

I2 = eye(2)
Im = eye(mx)
zero_matrix=np.zeros((mx**2, mx**2))


# The 4th order SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(mx, h)

# Construct the Boundary Operators #


# Dirichlet BC

# L = vstack([e_l, e_r])

# Neumann BC
L = vstack([d1_l, d1_r])

# Construct the Projection Operator

P = Im - HI@L.T@inv(coo_matrix(L@HI@L.T).tocsc())@L
# Solution Matrix
A = P@D2@P
D=kron(A, P)+kron(P, A)

print(np.eye(mx*mx).shape)
W =coo_matrix( np.block([
    [csc_matrix(zero_matrix).toarray(), np.eye(mx*mx)],
    [coo_matrix(D).toarray(), csc_matrix(zero_matrix).toarray()]]))

# W=coo_matrix(W)

# Numerical Solution
V = np.zeros((2*mx**2, 1))


# Initial conditons
for i in range(1, mx+1):
    V[mx*(i-1):mx*i] = f(xvec[i-1], yvec).reshape(mx,1)


# Time discretization
k = 0.01  
mt = int(np.ceil(T/k) + 1)  # round up so that (mt-1)*k = T
tvec, ht = np.linspace(0, T, mt, retstep=True)
# print(ht)

# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*W@V
    k2 = ht*W@(V + 0.5*k1)
    k3 = ht*W@(V + 0.5*k2)
    k4 = ht*W@(V + k3)

    V = V + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    if tidx % 50 == 0:
        V1 = V[0:mx**2].reshape(mx, mx)
        V2 = V[mx**2:2*mx**2].reshape(mx, mx)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a grid of X and Y values
        X, Y = np.meshgrid(np.linspace(0, 1, mx), np.linspace(0, 1, mx))

        # Create the 3D surface plot
        surf = ax.plot_surface(X, Y, V1, cmap='viridis', linewidth=0)

        # Set the z-axis limits
        ax.set_zlim(0, 1)

        # Add a text annotation
        ax.text2D(0.05, 0.95, "t = %f" % t, transform=ax.transAxes)

        # Show the plot
        plt.show()