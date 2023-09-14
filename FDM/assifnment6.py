from cmath import log
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import matplotlib.pyplot as plt
import rungekutta4 as rk4

#grid points 
m=101

#left and right boundary 
xl = -1.
xr = 1.

#number of steps
h = (xr - xl)/m

#end times 
T=[0.2,0.5,0.7,1.8]

#time step 
k=0.01


#Analytic solutions assuming the constant c=1
def theta1(x, t):
    return np.exp(-((x-t)/0.2)**2)

def theta2(x, t):
    return -np.exp(-((x+t)/0.2)**2)