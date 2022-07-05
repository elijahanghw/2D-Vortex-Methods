"""
Author: Elijah Ang Hao Wei
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

## Define airfoil properties
chord = 2           #Chord length
n_bound = 10        #Number of bound panels
n_wake = 500        #Number of wake panels
panel_normal = np.array([0, 1])
delta_c = chord/n_bound

## Define freestream properties
V_inf = 10          #Freestream velocity
AoA = 5             #Angle of attack (deg)
rho = 1             #Freestream density

## Define simulation parameters
num_steps = n_wake  #Number of time steps = number of wake panels to prevent truncation
dt = 0.01           #Timestep size
wake_vortex_factor = 0.3    #Location of wake panels (0 - 1)

def VOR2D(Gamma, x_c, z_c, x_v, z_v):
    r_2 = (x_c-x_v)**2 + (z_c - z_v)**2
    u = Gamma/(2*np.pi*r_2)*(z_c - z_v)
    w = -Gamma/(2*np.pi*r_2)*(x_c - x_v)
    return np.array([u, w])

def DISCRETISE(chord, n_bound, delta_c, n_wake, V_inf, dt, wake_vortex_factor):
    x_vortex = np.zeros(n_bound)
    z_vortex = np.zeros(n_bound)
    x_collocation = np.zeros(n_bound)
    z_collocation = np.zeros(n_bound)
    
    x_wake = np.zeros(n_wake)
    z_wake = np.zeros(n_wake)
    for i in range(n_bound):
        x_vortex[i] = (i+1/4)*delta_c
        z_vortex[i] = 0
        x_collocation[i] = (i+3/4)*delta_c
        z_collocation[i] = 0
    
    for j in range(n_wake):    
        x_wake[j] = chord + (j+wake_vortex_factor)*V_inf*dt
        z_wake[j] = 0
        
    return x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake

x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake = DISCRETISE(chord, n_bound, delta_c, n_wake, V_inf, dt, wake_vortex_factor)

## Define A Matrix
A_b = np.zeros([n_bound, n_bound])
A_w = np.zeros([n_bound, n_wake])
A_bw = np.zeros([n_wake, n_bound])
A_ww = np.eye(n_wake)

for i in range(n_bound):
    x_c = x_collocation[i]
    z_c = z_collocation[i]
    for j in range(n_bound):
        x_v = x_vortex[j]
        z_v = z_vortex[j]
        induced_velocity = VOR2D(1, x_c, z_c, x_v, z_v)
        A_b[i,j] = np.dot(induced_velocity, panel_normal)
        
    for k in range(n_wake):
        x_v = x_wake[k]
        z_v = z_wake[k]
        induced_velocity = VOR2D(1, x_c, z_c, x_v, z_v)
        A_w[i,k] = np.dot(induced_velocity, panel_normal)
        
    A_bw[0,i] = 1
        
A_top = np.concatenate((A_b, A_w), axis=1)
A_bot = np.concatenate((A_bw, A_ww), axis=1)
A = np.concatenate((A_top, A_bot), axis=0)

## Define B Matrix
B_b = np.zeros([n_bound, n_bound])
B_w = np.zeros([n_bound, n_wake])
B_bw = np.zeros([n_wake, n_bound])
B_ww = np.zeros([n_wake, n_wake])

for i in range(n_bound):
    B_bw[0,i] = 1
    
for i in range(1, n_wake):
        B_ww[i,i-1] = 1
        
B_top = np.concatenate((B_b, B_w), axis=1)
B_bot = np.concatenate((B_bw, B_ww), axis=1)
B = np.concatenate((B_top, B_bot), axis=0)

## Define downwash vector
W_b = np.zeros(n_bound)
W_w = np.zeros(n_wake)

for i in range(n_bound):
    W_b[i] = -V_inf * AoA * np.pi/180
    
W = np.concatenate((W_b, W_w), axis=0)

## Time stepping solution
Gamma_old = np.zeros((n_bound+n_wake))
time = []
Cl = []

for T in range(num_steps):
    time.append(T*dt)
    Lift = 0
    RHS = np.matmul(B, Gamma_old) + W
    Gamma_new = np.matmul(inv(A), RHS)
    Gamma_new_bound = Gamma_new[0:n_bound]
    Gamma_old_bound = Gamma_old[0:n_bound]
    
    for i in range(n_bound):
        delta_p = rho*(V_inf*Gamma_new_bound[i]/delta_c + (sum(Gamma_new_bound[0:i+1]) - sum(Gamma_old_bound[0:i+1]))/dt)
        Lift += delta_p*delta_c
        
    Cl.append(Lift/(0.5*rho*V_inf**2*chord))
    
    Gamma_old = Gamma_new
    
plt.plot(time, Cl, color='k')
plt.plot(time, 2*np.pi*AoA*np.pi/180*np.ones_like(time), color='r', linestyle="--")
plt.legend(["2D UVLM", "Kutta-Joukowski"])
plt.xlabel("time")
plt.ylabel("$C_l$")
plt.show()