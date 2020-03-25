# refrence:
#  A. Szabo and N. S. Ostlund
#  "Modern Quantum Chemistry" sections 3.4, 3.5

import numpy as np
from scipy.special import erf
from scipy.linalg import eigh

def F_(x):
    return np.where(x==0, 1, erf(x)/x*np.pi**0.5/2)

class H2molUHF:
    def __init__(self, d, Z=1, alpha=None, coeff=None):
        """ Hydrogen molecule by Unrestricted Hartree Fock method
        d = distance between protons
        Z = effective charge of a proton
        alpha = list of gaussian exponents
        coeff = list of gaussian coefficients (initial guess)
        """
        if alpha is None: alpha = [0.109818, 0.405771, 2.22766]
        if coeff is None: coeff = [0.444635, 0.535328, 0.154329]

        self.d = d
        alpha = np.array(alpha) * Z**2
        L = len(alpha)
        beta = alpha.reshape(L,1) + alpha
        gamma = np.outer(alpha, alpha)/beta
        tau = d*np.sqrt(beta)
        rho = (4*gamma/beta)**0.75
        bb = beta * beta.reshape(L,L,1,1)
        sigma = d*np.sqrt(bb/(beta + beta.reshape(L,L,1,1)))
        lm = alpha*tau/beta
        mu = alpha.reshape(L,1,1)*sigma/beta.reshape(L,L,1,1)
        nu = sigma*np.abs(alpha.reshape(L,1,1,1) * alpha.reshape(L,1) -
                          alpha.reshape(L,1,1) * alpha)/bb

        pi2d = 2/np.pi**0.5/d
        rr = rho * rho.reshape(L,L,1,1)

        S = rho * np.exp(-gamma*d**2)
        H11 = rho*(3*gamma - pi2d*tau*(1 + F_(tau)))
        H12 = S*(gamma*(3 - 2*gamma*d**2) - 2*pi2d*tau*F_(lm))
        V1111 = pi2d * rr * sigma
        V1122 = pi2d * rr * sigma * F_(sigma)
        V1211 = pi2d * rho * S.reshape(L,L,1,1) * sigma * F_(mu)
        V1212 = pi2d * S   * S.reshape(L,L,1,1) * sigma * F_(nu)

        L2 = L+L;
        T = np.empty((L2,L2)) # overlap integral
        H = np.empty((L2,L2)) # core Hamiltonian
        V = np.empty((L2,L2,L2,L2)) # electron repulsion inegrals

        T[:L,:L] = rho
        T[L:,L:] = rho
        T[:L,L:] = S
        T[L:,:L] = S
        H[:L,:L] = H11
        H[L:,L:] = H11
        H[:L,L:] = (H12 + H12.T)/2
        H[L:,:L] = H[:L,L:]
        V[:L,:L,:L,:L] = V1111
        V[L:,L:,L:,L:] = V1111
        V[L:,:L,:L,:L] = np.swapaxes(V1211,0,1)
        V[:L,L:,L:,L:] = V[L:,:L,:L,:L]
        V[:L,L:,:L,:L] = V1211
        V[L:,:L,L:,L:] = V1211
        V[:L,:L,L:,:L] = np.transpose(V1211,[3,2,0,1])
        V[L:,L:,:L,L:] = V[:L,:L,L:,:L]
        V[:L,:L,:L,L:] = np.transpose(V1211,[2,3,0,1])
        V[L:,L:,L:,:L] = V[:L,:L,:L,L:]
        V[:L,:L,L:,L:] = V1122
        V[L:,L:,:L,:L] = V1122
        V[:L,L:,:L,L:] = V1212
        V[L:,:L,L:,:L] = V1212
        V[:L,L:,L:,:L] = np.swapaxes(V1212,0,1)
        V[L:,:L,:L,L:] = V[:L,L:,L:,:L]

        c = np.hstack((coeff, [0]*L))
        EPS = 1e-9

        while True: # Hartree Fock self-consistency
            c1 = np.roll(c,L)
            C = np.outer(c1,c1)
            F = H + np.einsum('ijkl,kl', V, C)
            e,c1 = eigh(F, T, eigvals=(0,0))
            if np.max(np.abs(c - c1.T[0])) < EPS: break
            c = c1.T[0]

        self.E = 1/d + e[0] + np.sum(C*H)
        self.c = c
        self.d = d
        self.a = alpha

    def WaveFunc(self, x, y):
        """ wave function of one electron
        evaulated at (x,y,0) on xy plane
        protons are at (0,0,0) and (d,0,0)
        """
        a = (2*self.a/np.pi)**0.75
        y2 = y**2
        x1 = x**2 + y2
        x2 = (x - self.d)**2 + y2
        x1 = a * np.exp(-self.a * x1[...,np.newaxis])
        x2 = a * np.exp(-self.a * x2[...,np.newaxis])
        L = len(self.a)
        psi = np.sum(self.c[:L] * x1, axis=-1)
        psi+= np.sum(self.c[L:] * x2, axis=-1)
        return psi
