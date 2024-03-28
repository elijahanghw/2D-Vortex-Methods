# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------------------------------
# Author    :   E.H.W Ang
# Date      :   6/7/2022
#--------------------------------------------------------------------------------------------------------------------------
"""Unsteady Lumped Vortex solver for flat plate airfoil undergoing sudden acceleration. Results compared with Kutta-Joukowski theorem."""
#--------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from lib.singularities_2D import VOR2D
from lib.geometry import NACA_thin

## Define airfoil properties
chord = 2           #Chord length
n_bound = 10        #Number of bound panels
n_wake = 500        #Number of wake panels

## Define freestream properties
V_inf = 10          #Freestream velocity
AoA = 5             #Angle of attack (deg)
rho = 1             #Freestream density

## Define simulation parameters
num_steps = n_wake  #Number of time steps = number of wake panels to prevent truncation
dt = 0.01           #Timestep size
singularity = "LumpedVortex"

airfoil = NACA_thin("0012", chord, n_bound, spacing="regular")
x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake = airfoil.discretise(n_wake, V_inf, dt, singularity)

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
        A_b[i,j] = np.dot(induced_velocity, airfoil.normal[i])
        
    for k in range(n_wake):
        x_v = x_wake[k]
        z_v = z_wake[k]
        induced_velocity = VOR2D(1, x_c, z_c, x_v, z_v)
        A_w[i,k] = np.dot(induced_velocity, airfoil.normal[i])
        
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
    Freestream_x = -V_inf * np.cos(AoA * np.pi/180) 
    Freestream_y = -V_inf * np.sin(AoA * np.pi/180)

    W_b[i] = np.dot(np.array([Freestream_x, Freestream_y]), airfoil.normal[i])
    
W = np.concatenate((W_b, W_w), axis=0)

## State-space form
A_sys = inv(A) @ B
W_sys = inv(A) @ W

## Time stepping solution
Gamma_old = np.zeros((n_bound+n_wake))
time = []
Cl = []

for T in range(num_steps):
    time.append(T*dt)
    Lift = 0
    Gamma_new = A_sys @ Gamma_old + W_sys
    Gamma_new_bound = Gamma_new[0:n_bound]
    Gamma_old_bound = Gamma_old[0:n_bound]
    
    for i in range(n_bound):
        delta_p = rho*(V_inf*Gamma_new_bound[i]/airfoil.delta_c[i] + (sum(Gamma_new_bound[0:i+1]) - sum(Gamma_old_bound[0:i+1]))/dt)
        Lift += delta_p*airfoil.delta_c[i]
        
    Cl.append(Lift/(0.5*rho*V_inf**2*chord))
    
    Gamma_old = Gamma_new

plt.figure()
plt.plot(time, Cl, color='k')
plt.plot(time, 2*np.pi*AoA*np.pi/180*np.ones_like(time), color='r', linestyle="--")
plt.legend(["2D Lumped Vortex", "Kutta-Joukowski"])
plt.xlabel("time")
plt.ylabel("$C_l$")
plt.grid()

plt.figure()
plt.plot(x_vortex/chord, Gamma_new_bound/airfoil.delta_c, "-x")
plt.xlabel("x/c")
plt.ylabel("$\gamma/\Delta c$")
plt.grid()
plt.show()