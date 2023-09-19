#!/usr/bin/env python3

######################################################################################
##                                                                                  ##
##  Problemsolving 2 "Introduction to Finite Difference Methods", for course        ##
##  "Scientific computing for PDEs" at Uppsala University.                          ##
##                                                                                  ##
##  Author: Ken Mattsson                                                            ##
##  Date:   2023-09-15                                                              ##
##                                                                                  ##
##                                                                                  ##
##  Solves the 1D advection-diffusion equation with constant coefficients using    ##
##  summation-by-parts finite differences and the projection or SAT method to       ##
##  impose the boundary conditions.                                                 ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt

# Method parameters
# Number of grid points, integer > 15
mx = 101 

# Order of accuracy: 2, 3, 4, 5, 6, or 7. Odd orders are upwind operators
order = 4

# Type of method.
# 1 - Projection
# 0 - SAT

method = 1


# Model parameters
T = 1.8 # end time
a = 1
b = 0.1

# Domain boundaries
xl = -2
xr = 2


# Space discretization

hx = (xr - xl)/(mx-1)
xvec = np.linspace(xl,xr,mx)

if order == 2:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(mx,hx)
elif order == 4:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(mx,hx)
elif order == 6:
    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(mx,hx)
else:
    raise NotImplementedError('Order not implemented.')

I_m = eye(mx)


def norm(v):
    return np.sqrt(hx)*np.sqrt(np.sum(v**2))



# Construct RHS matrix using the projection method

if method==0:
    tau_l=a/2
    sigma_l=b
    tau_r=-b
    SAT_l=tau_l*HI@(e_l.T@e_l)+sigma_l*HI@(d1_l.T@e_l)
    SAT_r=tau_r*HI@(e_r.T@(a*e_r+2*b*d1_r))
    D = a*D1 + b*D2 + SAT_l + SAT_r
elif method == 1:
    L = vstack((e_l,a*e_r+2*b*d1_r),format="csc")
    P = I_m - HI@L.T@inv(L@HI@L.T)@L
    D = P@(a*D1 + b*D2)@P
else:
    raise NotImplementedError('This method is not implemented.')    

# Initial data
def gauss(x):
    rw = 6
    return np.exp(-(rw*x)**2)

v = np.hstack(gauss(xvec))

# Time discretization
CFL = 0.2
ht_try = CFL*hx**2
mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
tvec,ht = np.linspace(0,T,mt,retstep=True)

def rhs(v):
    return D@v

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
v0 = v[0:mx]
[line1] = ax.plot(xvec,v0,label='Solution')
plt.legend()
ax.set_xlim([xl,xr])
ax.set_ylim([0,1.5])
title = plt.title("t = " + "{:.2f}".format(0))
plt.draw()
plt.pause(0.5)

# Runge-Kutta 4
t = 0
for tidx in range(mt-1):
    k1 = ht*rhs(v)
    k2 = ht*rhs(v + 0.5*k1)
    k3 = ht*rhs(v + 0.5*k2)
    k4 = ht*rhs(v + k3)

    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    # Update plot every 50th time step
    if tidx % ceil(5) == 0 or tidx == mt-2:
        v0 = v[0:mx]
        line1.set_ydata(v0)
        title.set_text("t = " + "{:.2f}".format(tvec[tidx+1]))
        plt.draw()
        plt.pause(1e-3)


plt.show()
