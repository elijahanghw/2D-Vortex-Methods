# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------------------------------
# Author    :   E.H.W Ang
# Date      :   6/7/2022
#--------------------------------------------------------------------------------------------------------------------------
"""Unsteady Linear Vortex solver for flat plate airfoil undergoing heaving and pitching oscillations. Results compared with Theodorsen's aerodynamics."""
#--------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from cmath import phase

from lib.singularities_2D import LinVOR2D
from lib.singularities_2D import VOR2D
from lib.geometry import NACA_thin

## Define airfoil properties
chord = 2           #Chord length
n_bound = 20      #Number of bound panels
n_wake = 1000       #Number of wake panels

## Define freestream properties
V_inf = 10          #Freestream velocity
AoA = 0             #Angle of attack (deg)
rho = 1             #Freestream density

## Define simulation parameters
num_steps = n_wake  #Number of time steps = number of wake panels to prevent truncation
dt = 0.009           #Timestep size
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

## Define freestream downwash vector
W_b = np.zeros(n_bound+1)
W_w = np.zeros(n_wake)

for i in range(n_bound):
    Freestream_x = -V_inf * np.cos(AoA * np.pi/180) 
    Freestream_y = -V_inf * np.sin(AoA * np.pi/180)

    W_b[i] = np.dot(np.array([Freestream_x, Freestream_y]), airfoil.normal[i])
    
W = np.concatenate((W_b, W_w), axis=0)

## Define kinematic downwash vector
omega = 3      # Oscillation frequency
ea = 0.5        # Chordwise elastic axis wrt. LE
h0 = 0.02*chord # Heave amplitude
p0 = 0.5*np.pi/180 # Pitch amplitude

W_h = np.zeros((n_bound+1, num_steps))
W_p = np.zeros((n_bound+1, num_steps))

h = h0 * np.sin(omega*dt*np.arange(0, num_steps))
h_dot = omega * h0 * np.cos(omega*dt*np.arange(0, num_steps))
h_ddot = -omega**2 * h0 * np.sin(omega*dt*np.arange(0, num_steps))

p = p0 * np.sin(omega*dt*np.arange(0, num_steps))
p_dot = omega * p0 * np.cos(omega*dt*np.arange(0, num_steps))
p_ddot = -omega**2 * p0 * np.sin(omega*dt*np.arange(0, num_steps))

for i in range(n_bound):
    W_h[i] = h_dot - (x_collocation[i] - ea*chord) * p_dot
    W_p[i] = -V_inf * p

W_k = W_h + W_p

W_k = np.concatenate((W_k, np.zeros((n_wake, num_steps))), axis=0)

## State-space form
A_sys = inv(A) @ B
Wf_sys = inv(A) @ W
Wk_sys = inv(A) @ W_k

## Time stepping solution
gamma_old = np.zeros((n_bound+1+n_wake))
Gamma_old = np.zeros(n_bound)
Gamma_new = np.zeros(n_bound)
time = []
Cl = []

for T in range(num_steps):
    time.append(T*dt)
    Lift = 0
    gamma_new = A_sys @ gamma_old + Wf_sys + Wk_sys[:,T]
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

## Compare with Theodorsens
b = chord/2
a = (ea - 0.5) * chord
k = omega * b / V_inf

C_k = 1 - 0.165/(1- (0.0455/k)*1j) - 0.335/(1 - (0.3/k)*1j)
C_mag = abs(C_k)
C_arg = phase(C_k)

hC = h0 * np.sin(omega*dt*np.arange(0, num_steps) + C_arg)
hC_dot = omega * h0 * np.cos(omega*dt*np.arange(0, num_steps) + C_arg)

pC = p0 * np.sin(omega*dt*np.arange(0, num_steps) + C_arg)
pC_dot = omega * p0 * np.cos(omega*dt*np.arange(0, num_steps) + C_arg)

L_NC = rho * np.pi * b**2 * (V_inf*p_dot - a * p_ddot - h_ddot)
L_C = 2*np.pi*rho*b*V_inf*C_mag*(V_inf*pC - hC_dot - (a - b/2)*pC_dot)

L_tot = L_NC + L_C
Cl_theodorsen = L_tot/(0.5*rho*V_inf**2*chord)

plt.figure()
plt.plot(time, Cl, color='k')
plt.plot(time, Cl_theodorsen, color='b')

plt.xlabel("time")
plt.ylabel("$C_l$")
plt.grid()

plt.show()