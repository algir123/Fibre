import numpy as np
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import pandas as pd

a = 4.1e-6
lambda0 = 488e-9
NA = 0.14
k0 = 2*np.pi/lambda0
V = (2 * np.pi * a / lambda0) * NA

modes = ['LP01', 'LP11', 'LP21', 'LP02', 'LP31', 'LP12', 'LP41', 'LP22', 'LP03']

constantes = {
    "Si": {"A": [0.696166, 0.407942, 0.897479], "L": [0.068404, 0.116241, 9.896161]},
    "Ge": {"A": [0.806866, 0.718158, 0.854168], "L": [0.068972, 0.153966, 11.84193]}
}

def get_n(wavelength, x):
    n_sq_minus_1 = 0
    for i in range(3):
        r = constantes["Si"]["A"][i] + x * (constantes["Ge"]["A"][i] - constantes["Si"]["A"][i])
        b = constantes["Si"]["L"][i] + x * (constantes["Ge"]["L"][i] - constantes["Si"]["L"][i])
        n_sq_minus_1 += (r * wavelength**2) / (wavelength**2 - b**2)
    return np.sqrt(1 + n_sq_minus_1)

def eq_transcendante(u, l, V):
    w = np.sqrt(V**2 - u**2)
    lhs = jv(l, u) / (u * jv(l-1, u))
    rhs = -kn(l, w) / (w * kn(l-1, w))
    return lhs - rhs

def psi_l(w, l):
    if w < 1e-10:
        if l == 1: return 0
        return 1 - 1/abs(l-1)
    return (kn(l, w)**2) / (kn(l-1, w) * kn(l+1, w))

def eq_diff(V, u, l):
    w_sq = max(V**2 - u**2, 1e-12)
    w = np.sqrt(w_sq)
    
    if w < 1e-6:
        psi = 0 if l == 1 else (1 - 1/max(abs(l-1), 1e-10))
    else:
        psi = (kn(l, w)**2) / (kn(l-1, w) * kn(l+1, w))
    
    dudV = (u / V) * (1 - psi)
    return dudV

def calc_neff(u):
    if u > V:
        w = 0
    else:
        w = np.sqrt(V**2-u**2)
    gamma = w/a
    beta = np.sqrt(gamma**2+(k0*n2)**2)
    return beta/k0

n2 = get_n(0.488,0)

u_exact = []
u_myagi = []
u_eqdiff = []
neff_exact = []
neff_myagi = []
neff_eqdiff = []
Gamma = []

for mode in modes:
    l = int(mode[2])
    m = int(mode[3])
    zeros = jn_zeros(l, m)
    zeros_minus_1 = jn_zeros(l-1, m)
    cutoff = zeros[m-1]
    
    guess = cutoff * 0.85
    u = fsolve(eq_transcendante, guess, args=(l, V))[0]
    u_exact.append(float(u))
    
    w = np.sqrt(V**2-u**2)
    psi = kn(l, w)**2 / (kn(l + 1, w) * kn(l - 1, w))
    frac_puiss = 1-(u/V)**2*(1-psi)
    Gamma.append(frac_puiss)

    if l == 0 :
        if m == 1:
            u_val = 1e-6
        else:
            u_val = zeros_minus_1[m-2]
    else:
        u_val = zeros_minus_1[m-1]
    V_start = u_val-1e-8
    V_end = V
    
    sol = solve_ivp(lambda V, u_eq: eq_diff(V, u_eq, l), t_span=(V_start, V_end), y0=[V_start], rtol=1e-8)
    u_val = sol.y[0][-1]
    u_eqdiff.append(u_val)
    
    myagi = float(zeros[m-1]*(V/(V+1))*(1-zeros[m-1]**2/(6*(V+1)**3)-zeros[m-1]**4/(20*(V+1)**5)))
    u_myagi.append(myagi)
    
    neff_exact.append(calc_neff(u))
    neff_myagi.append(calc_neff(myagi))
    neff_eqdiff.append(calc_neff(u_val))


df_modes = pd.DataFrame({
    "Mode": modes,
    "u_exact": u_exact,
    "u_myagi": u_myagi,
    "u_eqdiff": u_eqdiff,
    "n_eff_exact": neff_exact,
    "n_eff_myagi": neff_myagi,
    "n_eff_eqdiff": neff_eqdiff,
    "Fraction puissance": Gamma
})

df_modes = df_modes.round({
    "u_exact": 5,
    "u_myagi": 5,
    "u_eqdiff": 5,
    "n_eff_exact": 5,
    "n_eff_myagi": 5,
    "n_eff_eqdiff": 5,
    "Fraction puissance": 5
})

print(df_modes.to_string(index=False))