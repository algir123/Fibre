import numpy as np
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import fsolve

def eq_transcendante(u, l, V):
    w = np.sqrt(V**2 - u**2)
    lhs = jv(l, u) / (u * jv(l-1, u))
    rhs = -kn(l, w) / (w * kn(l-1, w))
    return lhs - rhs


a = 4.6e-6
n1 = 1.462
n2 = 1.457
lamb = 1.5e-6
l = 0
R = 0.01149
L = np.pi*R

k0 = 2*np.pi/lamb
NA = np.sqrt(n1**2-n2**2)
V = 2*np.pi*a/lamb*NA

u = fsolve(eq_transcendante, 1.8, args=(0, V))[0]
w = np.sqrt(V**2-u**2)
beta = np.sqrt((n1 * k0)**2 - (u/a)**2)
K = (2*w**3)/(3*a**3*beta**2)
Ac = 1/2*np.sqrt(np.pi/(a*w**3))*(u/V)**2/kn(l-1, w)/kn(l+1, w)
alpha_c = Ac/np.sqrt(R)*np.exp(-K*R)

pertes = L*alpha_c
print(alpha_c)
pertes_dB = 10*np.log10(np.e)*pertes
print(pertes_dB)