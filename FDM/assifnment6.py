from cmath import log
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import matplotlib.pyplot as plt
import rungekutta4 as rk4
from math import sqrt, ceil

# Assuming c=1

def theta1(x,t):
    return np.exp(-((x-t)/0.2)**2)
def f(x):
    return theta1(x, 0)


#grid points 
mx=101
#Order of accuracy
order = 6
#left and right boundary 
xl = -1.
xr = 1.
#number of steps
hx = (xr - xl)/(mx-1)

xvec = np.linspace(xl,xr,mx)
#end times 
T=1.8
# identity matrix
I = np.identity(mx)
# zero matrix
zero_matrix=np.zeros((mx, mx))

k=0.01
mt = int(np.ceil(T/k) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)
#time step 

#assignment choice (boundary conditions)
assignment=4 #3,4,5

if order == 4:
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_4th(mx, hx)
elif order == 6:
    H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_6th(mx, hx)


if assignment==3:
    # boundary conditions based on results for stability 
    alpha_l = 0.
    alpha_r = 0.
    beta_l = 1  
    beta_r = 1  
    gamma_l = 0  
    gamma_r = 0  
    L=vstack([beta_l*e_l,
            beta_l*e_r])

    # projection operator
    P = I- HI@L.T@inv(L@HI@L.T)@L
    A=P@D2@P
    W = np.block([
        [zero_matrix, I],
        [A, zero_matrix]])
elif assignment==4:
    alpha_l = 0.
    alpha_r = 0.
    beta_l = 1  
    beta_r = 1  
    gamma_l = 0  
    gamma_r = 0
    L=vstack([beta_l*dl_l,
            beta_l*dl_r])
    tau_l=1
    tau_r=-1
    SAT_l=tau_l*HI@(e_l.T@dl_l)
    SAT_r=tau_r*HI@(e_r.T@dl_r)
    D = D2 + SAT_l + SAT_r
    W = np.block([
    [zero_matrix, I],
    [D.toarray(), zero_matrix]])
elif assignment==5:
    alpha_l = 0.
    alpha_r = 0.
    beta_l = 1  
    beta_r = 1  
    gamma_l = 0  
    gamma_r = 0
    L=vstack([beta_l*e_l,
            beta_l*e_r])
    tau_l=-1
    tau_r=-1
    SAT_l=-tau_l*HI@(e_l.T@dl_l)
    SAT_r=+tau_r*HI@(e_r.T@dl_r)
    D = D2 + SAT_l + SAT_r
    B = +tau_l*HI@(e_l.T@e_l) +tau_r*HI@(e_r.T@e_r)

    W = np.block([
    [zero_matrix, I],
    [D.toarray(), B.toarray()]])


v=np.hstack((f(xvec),np.zeros(mx)))
v=v.reshape(-1,1)
def rhs(x):
    return W@x



# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
v0 = v[0:mx]
[line1] = ax.plot(xvec,v0,label='Solution')
plt.legend()
ax.set_xlim([xl,xr])
ax.set_ylim([-1.5,1.5])
title = plt.title("t = " + "{:.2f}".format(0))
plt.draw()
plt.pause(0.5)

t = 0

for tidx in range(mt-1):

    v, t = rk4.step(rhs, v, t, ht)

    if tidx % ceil(5) == 0 or tidx == mt-2:
        v0 = v[0:mx]
        line1.set_ydata(v0)
        title.set_text("t = " + "{:.2f}".format(tvec[tidx+1]))
        plt.draw()
        plt.pause(1e-3)