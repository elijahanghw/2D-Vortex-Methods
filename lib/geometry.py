import numpy as np
import matplotlib.pyplot as plt

class NACA_thin:
    def __init__(self, NACA, chord, n_panels, spacing="regular"):

        self.m = int(NACA[0])/100
        self.p = int(NACA[1])/10
        self.chord = chord
        self.n_panels = n_panels
        self.spacing = spacing

        if self.spacing == "regular":
            self.X = np.linspace(0, 1, num=self.n_panels+1)

        elif self.spacing == "cosine":
            beta = np.linspace(0, np.pi, num=self.n_panels+1)
            self.X = 0.5*(1-np.cos(beta))

        else:
            raise Exception(f"{spacing} spacing not supported or invalid.")

        self.Z = np.zeros_like(self.X)
        for i,val in enumerate(self.X):
            if val > 0 and val <= self.p:
                self.Z[i] = self.m/self.p**2 * (2*self.p*val - val**2)
            elif val >= self.p and val <= 1:
                self.Z[i] = self.m/(1-self.p)**2 * ((1 - 2*self.p) + 2*self.p*val - val**2)

        self.X = self.chord * self.X
        self.Z = self.chord * self.Z

        self.delta_c = np.zeros(self.n_panels)
        self.normal = np.zeros((self.n_panels, 2))
        for i in range(self.n_panels):
            self.delta_c[i] = np.sqrt((self.X[i+1] - self.X[i])**2 + (self.Z[i+1] - self.Z[i])**2)
            normal = np.array([self.Z[i] - self.Z[i+1], self.X[i+1] - self.X[i]])
            self.normal[i,:] = normal / np.linalg.norm(normal)
        
    def discretise(self, n_wake, V_inf, dt, singularity, wake_vortex_factor=0.3):

        if singularity == "LumpedVortex":
            num_vortex = self.n_panels
            vor_loc = 1/4
            col_loc = 3/4

        elif singularity == "LinearVortex":
            num_vortex = self.n_panels+1
            vor_loc = 0
            col_loc = 1/2

        else:
            raise Exception(f"{singularity} singularity not supported or invalid.")

        x_vortex = np.zeros(num_vortex)
        z_vortex = np.zeros(num_vortex)
        x_collocation = np.zeros(self.n_panels)
        z_collocation = np.zeros(self.n_panels)

        x_wake = np.zeros(n_wake)
        z_wake = np.zeros(n_wake)

        for i in range(self.n_panels):
            x_vortex[i] = self.X[i] + (vor_loc)*(self.X[i+1] - self.X[i])
            z_vortex[i] = self.Z[i] + (vor_loc)*(self.Z[i+1] - self.Z[i])
            x_collocation[i] = self.X[i] + (col_loc)*(self.X[i+1] - self.X[i])
            z_collocation[i] = self.Z[i] + (col_loc)*(self.Z[i+1] - self.Z[i])

        if singularity == "LinearVortex":
            x_vortex[-1] = self.X[-1]
            z_vortex[-1] = self.Z[-1]
        
        for j in range(n_wake):    
            x_wake[j] = self.chord + (j+wake_vortex_factor)*V_inf*dt
            z_wake[j] = 0
        
        
        
        return x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake


# airfoil = NACA_thin("0012", 2, 20, spacing="cosine")
# x_vortex, z_vortex, x_collocation, z_collocation, x_wake, z_wake = airfoil.discretise(100, 20, 0.01, "LumpedVortex")

# fig, ax = plt.subplots()
# ax.plot(airfoil.X, airfoil.Z, "-x")
# ax.scatter(x_vortex, z_vortex, marker="o")
# ax.quiver(x_collocation, z_collocation, airfoil.normal[:,0], airfoil.normal[:,1], scale = 20)
# ax.set_aspect('equal')
# ax.set_ylim([-1, 1])
# plt.show()
