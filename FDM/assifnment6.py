from cmath import log
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import matplotlib.pyplot as plt
import rungekutta4 as rk4

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
T=1
CFL = 0.5
ht_try = 0.01*hx
mt = int(np.ceil(T/ht_try) + 1)  # round up so that (mt-1)*ht = T
tvec, ht = np.linspace(0, T, mt, retstep=True)
#time step 
k=0.01
method=1 #projection
# method=0 # SAT
#identity matrix 
I = np.identity(mx)
zero_matrix=np.zeros((mx, mx))
# boundary conditions (based on assignment 1 with dirichlet )
alpha_l = 0.
alpha_r = 0.
beta_l = 1  
beta_r = 1  
gamma_l = 0  
gamma_r = 0  


#Analytic solutions assuming the constant c=1
def theta1(x,t):
    return np.exp(-((x-t)/0.2)**2)

def theta2(x, t):
    return -np.exp(-((x+t)/0.2)**2)

# e1 = np.array([1, 0])
# e2 = np.array([0, 1])

# if order==4:
#     H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_4th(mx, hx)

# if order==6:
H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_6th(mx, hx)

# if method==1:
L=vstack([beta_l*e_l,
            beta_l*e_r])
P = I- HI@L.T@inv(L@HI@L.T)@L
A=P@D2@P



W = np.block([
    [zero_matrix, I],
    [A, zero_matrix]])


def f(x):
    return theta1(x, 0)

v=np.hstack((f(xvec),np.zeros(mx)))
v=v.reshape(-1,1)
def rhs(x):
    return W@x
ht=0.1*hx

t = 0

for tidx in range(mt-1):

    k1 = ht*rhs(v)
    k2 = ht*rhs(v + 0.5*k1)
    k3 = ht*rhs(v + 0.5*k2)
    k4 = ht*rhs(v + k3)
    v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    t = t + ht

    if tidx in [50,100,150,200,250,300,400]:
        
        plt.plot(v[0:mx])
        plt.show()