from cmath import log
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import eye, vstack, kron, coo_matrix
import operators as ops
import matplotlib.pyplot as plt
import rungekutta4 as rk4
from math import sqrt, ceil

#left and right boundary 
xl = -1
xr = 1
#number of steps

#end times 
T=1


#time step 
k=0.01
method=1 #projection

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


def f(x):
    return theta1(x, 0)

def rhs(x):
    return W@x
        


if __name__ == "__main__":
    mvec = np.array([50, 100, 200, 300])
    order_vec = np.array([4, 6])
    errvec = np.zeros((mvec.size, order_vec.size))

    for order_idx, order in enumerate(order_vec):
        for m_idx, m in enumerate(mvec):
            # Modify grid points and order of accuracy
            mx = m
            hx = (xr - xl) / (mx - 1)
            xvec = np.linspace(xl, xr, mx)
            ht_try = 0.01 * hx
            mt = int(np.ceil(T / ht_try) + 1)  # round up so that (mt-1)*ht = T
            tvec, ht = np.linspace(0, T, mt, retstep=True)
            I = np.identity(mx)
            zero_matrix = np.zeros((mx, mx))
            # Copy the relevant part of your code for solving the problem
            if order == 4:
                H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_4th(mx, hx)
            elif order == 6:
                H, HI, D1, D2, e_l, e_r, dl_l, dl_r = ops.sbp_cent_6th(mx, hx)
            
            # Rest of your code to set up operators and solve the problem
            L = vstack([beta_l * e_l, beta_l * e_r])

            P = I - HI @ L.T @ inv(L @ HI @ L.T) @ L
            A = P @ D2 @ P

            W = np.block([
                [zero_matrix, I],
                [A, zero_matrix]])
            
            
            v = np.hstack((f(xvec), np.zeros(mx)))[:mx]  # Ensure v has 200 elements
            v = v.reshape(-1, 1)
            
            
            v_exact = np.vstack([theta1(xvec, T), theta2(xvec, T)])[:, :mx].T
            
            # Calculate the L2 norm error
            error = np.sqrt(hx) * np.sqrt(np.sum((v - v_exact) ** 2))

            errvec[m_idx, order_idx] = error

    q = np.zeros((mvec.size, order_vec.size))
    
    for order_idx, order in enumerate(order_vec):
        for m_idx, m in enumerate(mvec[1:]):
            q[m_idx+1, order_idx] = -np.log(errvec[m_idx, order_idx] / errvec[m_idx+1, order_idx]) / np.log(mvec[m_idx] / mvec[m_idx+1])

    for order_idx, order in enumerate(order_vec):
        print("--- Order: %d ---" % order)
        print("m\terr\t\tq")
        for idx in range(mvec.size):
            print("%d\t%.2f\t%.2f" % (mvec[idx], np.log10(errvec[idx, order_idx]), q[idx, order_idx]))
        print("")