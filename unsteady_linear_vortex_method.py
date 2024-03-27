# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------------------------------
# Author    :   E.H.W Ang
# Date      :   6/7/2022
#--------------------------------------------------------------------------------------------------------------------------
"""Unsteady Linear Vortex solver for flat plate airfoil undergoing sudden acceleration. Results compared with Kutta-Joukowski theorem."""
#--------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from lib.two_dim.singularities import LinVOR2D
from lib.two_dim.singularities import VOR2D
from lib.geometry import NACA_thin

## Define airfoil properties
chord = 2           #Chord length
n_bound = 10      #Number of bound panels
n_wake = 500       #Number of wake panels

## Define freestream properties
V_inf = 10          #Freestream velocity
AoA = 5             #Angle of attack (deg)
rho = 1             #Freestream density

## Define simulation parameters
num_steps = n_wake  #Number of time steps = number of wake panels to prevent truncation
dt = 0.01           #Timestep size
singularity = "LinearVortex"

airfoil = NACA_thin("0012", chord, n_bound, spacing="regular")
x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake = airfoil.discretise(n_wake, V_inf, dt, singularity)

## Define A Matrix
A_b = np.zeros([n_bound+1, n_bound+1])
A_w = np.zeros([n_bound+1, n_wake])
A_bw = np.zeros([n_wake, n_bound+1])
A_ww = np.eye(n_wake)

for i in range(n_bound):
    x_c = x_collocation[i]
    z_c = z_collocation[i]
    for j in range(n_bound):
        x_v1 = x_vortex[j]
        z_v1 = z_vortex[j]
        x_v2 = x_vortex[j+1]
        z_v2 = z_vortex[j+1]
        induced_velocity = LinVOR2D(1, 1, x_c, z_c, x_v1, z_v1, x_v2, z_v2)
        induced_velocity1 = induced_velocity[0,:]
        induced_velocity2 = induced_velocity[1,:]
        A_b[i,j] = A_b[i,j] + np.dot(induced_velocity1, airfoil.normal[i])
        A_b[i,j+1] = A_b[i,j+1] + np.dot(induced_velocity2, airfoil.normal[i])
        
    for k in range(n_wake):
        x_v = x_wake[k]
        z_v = z_wake[k]
        induced_velocity = VOR2D(1, x_c, z_c, x_v, z_v)
        A_w[i,k] = np.dot(induced_velocity, airfoil.normal[i])
        
    A_bw[0,i] = A_bw[0,i] + 0.5*airfoil.delta_c[i]
    A_bw[0,i+1] = A_bw[0,i+1] + 0.5*airfoil.delta_c[i]

A_b[-1,-1] = 1

A_top = np.concatenate((A_b, A_w), axis=1)
A_bot = np.concatenate((A_bw, A_ww), axis=1)
A = np.concatenate((A_top, A_bot), axis=0)

## Define B Matrix
B_b = np.zeros([n_bound+1, n_bound+1])
B_w = np.zeros([n_bound+1, n_wake])
B_bw = np.zeros([n_wake, n_bound+1])
B_ww = np.zeros([n_wake, n_wake])

for i in range(n_bound):
    B_bw[0,i] = B_bw[0,i] + 0.5*airfoil.delta_c[i]
    B_bw[0,i+1] = B_bw[0,i+1] + 0.5*airfoil.delta_c[i]
    
for i in range(1, n_wake):
    B_ww[i,i-1] = 1

B_top = np.concatenate((B_b, B_w), axis=1)
B_bot = np.concatenate((B_bw, B_ww), axis=1)
B = np.concatenate((B_top, B_bot), axis=0)

## Define downwash vector
W_b = np.zeros(n_bound+1)
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
gamma_old = np.zeros((n_bound+1+n_wake))
Gamma_old = np.zeros(n_bound)
Gamma_new = np.zeros(n_bound)
time = []
Cl = []

for T in range(num_steps):
    time.append(T*dt)
    Lift = 0
    gamma_new = A_sys @ gamma_old + W_sys
    gamma_new_bound = gamma_new[0:n_bound+1]
    gamma_old_bound = gamma_old[0:n_bound+1]
    gamma_new_wake = gamma_new[n_bound+1:]

    for k in range(n_bound):
        Gamma_old[k] = (gamma_old_bound[k] + gamma_old_bound[k+1])/2 * airfoil.delta_c[k]
        Gamma_new[k] = (gamma_new_bound[k] + gamma_new_bound[k+1])/2 * airfoil.delta_c[k]
    
    for i in range(n_bound):
        delta_p = rho*(V_inf*Gamma_new[i]/airfoil.delta_c[i] + (sum(Gamma_new[0:i+1]) - sum(Gamma_old[0:i+1]))/dt)
        Lift += delta_p*airfoil.delta_c[i]
        
    Cl.append(Lift/(0.5*rho*V_inf**2*chord))
    
    gamma_old = gamma_new


plt.figure()
plt.plot(time, Cl, color='k')
plt.plot(time, 2*np.pi*AoA*np.pi/180*np.ones_like(time), color='r', linestyle="--")
plt.legend(["2D Linear Vortex", "Kutta-Joukowski"])
plt.xlabel("time")
plt.ylabel("$C_l$")
plt.grid()

plt.figure()
plt.plot(x_vortex/chord, gamma_new_bound, "-x")
plt.xlabel("x/c")
plt.ylabel("$\gamma$")
plt.grid()
plt.show()