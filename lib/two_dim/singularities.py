# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------------------------------
# Author    :   E.H.W Ang
# Date      :   20/3/2024
#--------------------------------------------------------------------------------------------------------------------------
"""Library of potential flow singularities"""
#--------------------------------------------------------------------------------------------------------------------------

import numpy as np

def VOR2D(Gamma, x_c, z_c, x_v, z_v):
    """This function returns a vector of the induced velocity on an arbitrary point due to a point vortex."""
    r = (x_c-x_v)**2 + (z_c - z_v)**2
    u = Gamma/(2*np.pi*r)*(z_c - z_v)
    w = -Gamma/(2*np.pi*r)*(x_c - x_v)
    return np.array([u, w])

def LinVOR2D(Gamma1, Gamma2, x_c, z_c, x_v1, z_v1, x_v2, z_v2):
    """This function returns a vector of the induced velocity on an arbitrary point due to a linear vortex panel."""
    

    beta = np.arctan2((z_v2 - z_v1),(x_v2 - x_v1))

    x_c_loc = np.cos(beta) * (x_c - x_v1) + np.sin(beta) * (z_c - z_v1)
    z_c_loc = -np.sin(beta) * (x_c - x_v1) + np.cos(beta) * (z_c - z_v1)

    x_v1_loc = np.cos(beta) * (x_v1 - x_v1) + np.sin(beta) * (z_v1 - z_v1)
    z_v1_loc = -np.sin(beta) * (x_v1 - x_v1) + np.cos(beta) * (z_v1 - z_v1)

    x_v2_loc = np.cos(beta) * (x_v2 - x_v1) + np.sin(beta) * (z_v2 - z_v1)
    z_v2_loc = -np.sin(beta) * (x_v2 - x_v1) + np.cos(beta) * (z_v2 - z_v1)

    r1 = np.sqrt((x_c_loc - x_v1_loc)**2 + (z_c_loc - z_v1_loc)**2)
    r2 = np.sqrt((x_c_loc - x_v2_loc)**2 + (z_c_loc - z_v2_loc)**2)

    theta1 = np.arctan2((z_c_loc - z_v1_loc),(x_c_loc - x_v1_loc))
    theta2 = np.arctan2((z_c_loc - z_v2_loc),(x_c_loc - x_v2_loc))

    u1 = z_c_loc/(2*np.pi) * -Gamma1/(x_v2_loc - x_v1_loc) * np.log(r2/r1) \
        + (Gamma1*(x_v2_loc-x_v1_loc) - Gamma1*(x_c_loc - x_v1_loc))/(2*np.pi*(x_v2_loc - x_v1_loc)) * (theta2 - theta1)

    u2 = z_c_loc/(2*np.pi) * Gamma2/(x_v2_loc - x_v1_loc) * np.log(r2/r1) \
        + (Gamma2*(x_c_loc - x_v1_loc))/(2*np.pi*(x_v2_loc - x_v1_loc)) * (theta2 - theta1)
    
    w1 = -(Gamma1*(x_v2_loc - x_v1_loc) - Gamma1*(x_c_loc - x_v1_loc))/(2*np.pi*(x_v2_loc - x_v1_loc)) * np.log(r1/r2) \
         +  1/(2*np.pi) * -Gamma1/(x_v2_loc - x_v1_loc) * ((x_v2_loc - x_v1_loc) + z_c_loc*(theta2 - theta1))

    w2 = - Gamma2*(x_c_loc - x_v1_loc)/(2*np.pi*(x_v2_loc - x_v1_loc)) * np.log(r1/r2) \
         + 1/(2*np.pi) * Gamma2/(x_v2_loc - x_v1_loc) * ((x_v2_loc - x_v1_loc) + z_c_loc*(theta2 - theta1))
    
    U1 = np.cos(beta) * u1 - np.sin(beta) * w1
    W1 = np.sin(beta) * u1 + np.cos(beta) * w1

    U2 = np.cos(beta) * u2 - np.sin(beta) * w2
    W2 = np.sin(beta) * u2 + np.cos(beta) * w2

    return np.array([[U1, W1],[U2, W2]])