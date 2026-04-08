import numpy as np
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import fsolve

constantes = {
    "Si": {"A": [0.696166, 0.407942, 0.897479], "L": [0.068404, 0.116241, 9.896161]},
    "Ge": {"A": [0.806866, 0.718158, 0.854168], "L": [0.068972, 0.153966, 11.84193]}
}

def get_n(wavelength, x):
    n_sq_minus_1 = 0
    for i in range(3):
        # Interpolation linéaire des coefficients de Sellmeier selon la fraction molaire x
        a = constantes["Si"]["A"][i] + x * (constantes["Ge"]["A"][i] - constantes["Si"]["A"][i])
        # Attention : b dans Sellmeier est souvent L^2. Vérifiez si vos constantes L sont déjà au carré.
        b = constantes["Si"]["L"][i] + x * (constantes["Ge"]["L"][i] - constantes["Si"]["L"][i])
        n_sq_minus_1 += (a * wavelength**2) / (wavelength**2 - b**2)
    return np.sqrt(1 + n_sq_minus_1)

def eq_transcendante(u, l, V):
    w = np.sqrt(V**2 - u**2)
    lhs = jv(l, u) / (u * jv(l-1, u))
    rhs = -kn(l, w) / (w * kn(l-1, w))
    return lhs - rhs

def calc_neff(u):
    w = np.sqrt(V**2-u**2)
    gamma = w/a
    beta = np.sqrt(gamma**2+(k0*n_gaine)**2)
    return beta/k0

# Paramètres
period = 0.540 # en µm (1080nm / 2)
x_geo2 = 0.08
lambda_bragg = 1.55 # Valeur initiale estimée en µm
delta_n = 0.0001
L = 0.015
# Algorithme de convergence
tolerance = 1e-9
a = 2.5
diff = 1
iteration = 0
visibilite = 1
l = 0
m = 1

################# A ##################
while diff > tolerance:
    n_core = get_n(lambda_bragg, x_geo2)
    n_gaine = get_n(lambda_bragg, 0)
    NA = np.sqrt(n_core**2-n_gaine**2)
    V =  2*np.pi/lambda_bragg*a*NA
    k0 = 2*np.pi/lambda_bragg
    
    zeros = jn_zeros(l, m)
    cutoff = zeros[m-1]
    guess = 1
    
    u = fsolve(eq_transcendante, guess, args=(l, V))[0]
    neff = calc_neff(u)
    
    lambda_new = 2 * neff * period
    
    diff = abs(lambda_new - lambda_bragg)
    lambda_bragg = lambda_new
    iteration += 1
    #print(f"Iteration {iteration}: λ = {lambda_bragg:.6f} µm, n = {neff:.6f}")
    
print('neff', neff)
print(f"Longueur d'onde de Bragg convergée : {lambda_bragg:.4f} µm\n")

################# B ##################
kappa = np.pi/lambda_bragg*delta_n*visibilite*1e6
print(kappa)
reflectivite = np.tanh(kappa*L)**2

print(f'Reflectivité Max: {(reflectivite*100):.2f} %')

transmission = 1-reflectivite
loss = -10*np.log10(transmission)

print(f'Loss dB {loss:.2f} dB')

################# C ##################
diff = 1
iteration = 0

while diff > tolerance:
    n_core = get_n(lambda_bragg, x_geo2)
    n_gaine = get_n(lambda_bragg, 0)
    NA = np.sqrt(n_core**2-n_gaine**2)
    V =  2*np.pi/lambda_bragg*a*NA
    k0 = 2*np.pi/lambda_bragg
    zeros = jn_zeros(l, m)
    cutoff = zeros[m-1]
    guess = 1
    
    u = fsolve(eq_transcendante, guess, args=(l, V))[0]
    neff = calc_neff(u)
    
    lambda_new = neff * period
    
    diff = abs(lambda_new - lambda_bragg)
    lambda_bragg = lambda_new
    iteration += 1
    #print(f"Iteration {iteration}: λ = {lambda_bragg:.6f} µm, n = {neff:.6f}")

print('neff', neff)
print(f"Longueur d'onde de Bragg convergée : {lambda_bragg:.4f} µm\n")
#LP11
l = 1
m = 1
diff = 1
iteration = 0
while diff > tolerance:
    n_core = get_n(lambda_bragg, x_geo2)
    n_gaine = get_n(lambda_bragg, 0)
    NA = np.sqrt(n_core**2-n_gaine**2)
    V =  2*np.pi/lambda_bragg*a*NA
    k0 = 2*np.pi/lambda_bragg
    zeros = jn_zeros(l, m)
    cutoff = zeros[m-1]
    guess = 3
    
    u = fsolve(eq_transcendante, guess, args=(l, V))[0]
    neff = calc_neff(u)
    
    lambda_new = neff * period
    
    diff = abs(lambda_new - lambda_bragg)
    lambda_bragg = lambda_new
    iteration += 1
    # print(f"LP11 Iteration {iteration}: λ = {lambda_bragg:.6f} µm, n = {neff:.6f}")
print('neff', neff)
print(f"Longueur d'onde de Bragg convergée : {lambda_bragg:.4f} µm\n")