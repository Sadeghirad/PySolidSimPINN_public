import numpy as np


class ref_solution:
    """
    """

    def __init__(self, young, nu):
        self.E = young / (1. - nu ** 2)
        self.nu = nu / (1. - nu)
        self.r_out = 5.
        self.r_in = 3.
        self.p = 10.

    def exact_disp(self, X):
        """
        """
        x = X[0]
        y = X[1]
        r = np.hypot(x, y)
        t = np.arctan2(y, x)
        a = self.r_in
        b = self.r_out
        s = np.sin(t)
        c = np.cos(t)
        rb = 0.5 * (a + b)
        a2 = a**2
        b2 = b**2
        r2 = r**2
        r4 = r2**2
        rb2 = rb**2
        P = self.p / (a2 - b2 + (a2 + b2) * np.log(b / a))
        B = ( P * (3*self.nu*rb**4 - rb**4 + a2*b2 + a2*b2*self.nu + 2*a2*rb2*np.log(rb) + 2*b2*rb2*np.log(rb) - 2*a2*self.nu*rb2*np.log(rb) - 2*b2*self.nu*rb2*np.log(rb)) ) / (2*self.E*rb2)
        ur = B * c - ( 2*P*t*s*(a2 + b2) ) / self.E - ( P * c * (3*self.nu*r4 - r4 + a2*b2 + a2*b2*self.nu - 2*a2*r2*np.log(r)*(self.nu - 1)
                                                            - 2*b2*r2*np.log(r)*(self.nu - 1)) ) / (2*self.E*r2)
        ut = ( 2*P*(a2 + b2)*(s - t*c) ) / self.E - B*s - ( P*s*(a2*b2 - 5*r4 - self.nu*r4 + 2*a2*r2 + 2*b2*r2 + a2*b2*self.nu - 2*a2*self.nu*r2
                                                       - 2*b2*self.nu*r2 + 2*a2*r2*np.log(r)*(self.nu - 1) + 2*b2*r2*np.log(r)*(self.nu - 1)) ) / (2*self.E*r2)
        u1 = ur*c - ut*s
        u2 = ur*s + ut*c
        U = np.array([u1, u2])
        return U

    def exact_strain(self, X):
        """
        """
        x = X[0]
        y = X[1]
        r = np.hypot(x, y)
        t = np.arctan2(y, x)
        a = self.r_in
        b = self.r_out
        s = np.sin(t)
        c = np.cos(t)
        a2 = a ** 2
        b2 = b ** 2
        r2 = r ** 2
        r3 = r * r2
        P = self.p / (a2 - b2 + (a2 + b2) * np.log(b / a))
        err = ( P*c*(r - (a2 + b2)/r + (a2*b2)/r3) + P*self.nu*c*((a2 + b2)/r - 3*r + (a2*b2)/r3))/self.E
        ett = -(P*c*((a2 + b2)/r - 3*r + (a2*b2)/r3) + P*self.nu*c*(r - (a2 + b2)/r + (a2*b2)/r3))/self.E
        ert = (P*s*(a2 - r2)*(b2 - r2)*(self.nu + 1))/(self.E*r3)
        s = np.sin(-t)
        c = np.cos(-t)
        e11 = err * c**2 + ett * s**2 + 2 * ert * s * c
        e22 = err * s**2 + ett * c**2 - 2 * ert * s * c
        e12 = (ett - err) * s * c + ert * (c**2 - s**2)
        E1 = np.array([e11, e22, 2 * e12])
        return E1

    def exact_stress(self, X):
        """
        """
        x = X[0]
        y = X[1]
        r = np.hypot(x, y)
        t = np.arctan2(y, x)
        a = self.r_in
        b = self.r_out
        s = np.sin(t)
        c = np.cos(t)
        a2 = a ** 2
        b2 = b ** 2
        r2 = r ** 2
        r3 = r * r2
        P = self.p / (a2 - b2 + (a2 + b2) * np.log(b/a))
        srr = P*(r + a2 * b2/r3 - (a2 +b2)/r)*c
        stt = P*(3*r - a2 *b2/r3 - (a2 +b2)/r)*c
        srt = P*(r + a2*b2/r3 - (a2 +b2)/r)*s
        s = np.sin(-t)
        c = np.cos(-t)
        s11 = srr * c ** 2 + stt * s ** 2 + 2 * srt * s * c
        s22 = srr * s ** 2 + stt * c ** 2 - 2 * srt * s * c
        s12 = (stt - srr) * s * c + srt * (c ** 2 - s ** 2)
        S = np.array([s11, s22, s12])
        return S
