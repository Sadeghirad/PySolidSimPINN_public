import numpy as np


class ref_solution:
    """
    """

    def __init__(self, young, nu):
        self.E = young / (1. - nu ** 2)
        self.nu = nu / (1. - nu)
        self.L = 0.25
        self.h = 0.05
        self.M = 2./3. * 50. * self.h**2

    def exact_disp(self, X):
        """
        """
        x = X[0] - self.L
        y = X[1]
        u1 = -2. * (x+self.L) * y
        u2 = (x+2.*self.L) * x + self.nu * y**2 + self.L**2
        U = 3./4.*self.M/self.E/self.h**3 * np.array([u1, u2])
        return U

    def exact_strain(self, X):
        """
        """
        y = X[1]
        e11 = -y
        e22 = self.nu * y
        e12 = 0.
        E1 = 3./2.*self.M/self.E/self.h**3 * np.array([e11, e22, 2 * e12])
        return E1

    def exact_stress(self, X):
        """
        """
        y = X[1]
        s11 = -3./2.*self.M/self.h**3 * y
        s22 = 0.
        s12 = 0.
        S = np.array([s11, s22, s12])
        return S
