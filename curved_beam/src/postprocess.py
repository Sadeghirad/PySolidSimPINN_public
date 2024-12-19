import numpy as np
from src.visualizer import plot_loss


class postprocess:
    """
    """

    def __init__(self, Xp, Xp_errNorm, Vp_errNorm, refSol):
        self.Xp = Xp
        self.Xp_errNorm = Xp_errNorm
        self.Vp_errNorm = Vp_errNorm
        self.refSol = refSol

    def postprocess_err(self, U):
        Up, Sp, Ep = U[0], U[1], U[2]
        errNorm = self.calc_error_norm_err(self.Vp_errNorm, Up, Sp, Ep)
        return errNorm['Disp_ErrNormp  '], errNorm['Energy_ErrNormp']

    def postprocess(self, result, U):
        plot_loss(result)
        Up = U[0]
        Sp = U[1]
        print(f'\nThe simulation is done. Bye! ........ \n')

    def calc_error_norm_err(self, volp, Up, Sp, Ep):
        numpar = Up.shape[0]
        Upex = np.zeros((numpar, 2))
        Epex = np.zeros((numpar, 3))
        Spex = np.zeros((numpar, 3))
        for i, xp in enumerate(self.Xp_errNorm):
            Upex[i, :] = self.refSol.exact_disp(xp)
            Epex[i, :] = self.refSol.exact_strain(xp)
            Spex[i, :] = self.refSol.exact_stress(xp)
        errNorm = {}
        errNorm['Disp_ErrNormp  '], errNorm['Energy_ErrNormp'] = self.calc_normp_err(volp, Up, Upex, Sp, Spex, Ep, Epex)
        return errNorm

    def calc_normp_err(self, volp, Up, Upex, Sp, Spex, Ep, Epex):
        normDisp = np.sum((Up-Upex)**2 * volp)
        normDispEx = np.sum(Upex**2 * volp)
        normEner = np.sum((Sp - Spex) * (Ep - Epex) * volp)
        normEnerEx = np.sum(Spex * Epex * volp)
        return np.sqrt(normDisp/normDispEx), np.sqrt(normEner/normEnerEx)
