import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.optimize import brentq

c = 3*10**8 # m/s
r_coeur = 5.28 # um

# Données du tableau de constantes
constantes = {
    "Si": {"A": [0.696166, 0.407942, 0.897479], "L": [0.068404, 0.116241, 9.896161]},
    "Ge": {"A": [0.806866, 0.718158, 0.854168], "L": [0.068972, 0.153966, 11.84193]}
}

def get_n(wavelength, x):
    """
    Calcule l'indice de réfraction n selon l'équation de Sellmeier.
    wavelength: float ou array en µm
    x: fraction molaire de GeO2
    """

    n_sq_minus_1 = 0
    for i in range(3):
        a = constantes["Si"]["A"][i] + x * (constantes["Ge"]["A"][i] - constantes["Si"]["A"][i])
        b = constantes["Si"]["L"][i] + x * (constantes["Ge"]["L"][i] - constantes["Si"]["L"][i])
        n_sq_minus_1 += (a * wavelength**2) / (wavelength**2 - b**2)
        
    return np.sqrt(1 + n_sq_minus_1)

def calculate_material_dispersion(wavelength, x, step=0.001):
    # Calcul de la dérivée seconde d2n/dlambda2 par différence finie centrale
    n_plus = get_n(wavelength + step, x)
    n_mid  = get_n(wavelength, x)
    n_minus = get_n(wavelength - step, x)
    
    d2n_dlam2 = (n_plus - 2*n_mid + n_minus) / (step**2)
    
    # Conversion : x 10^12 pour passer de s/(µm-m) à ps/(nm-km)
    dm = -(wavelength / c) * d2n_dlam2 * 1e12
    return dm



x_core = 0.02    #x: fraction molaire de GeO2
lambdas = np.linspace(1.0, 1.5, 1000)

dm = [calculate_material_dispersion(l, x_core) for l in lambdas]


def get_n_group(wavelength, x, step=0.001):
    n = get_n(wavelength, x)
    n_plus = get_n(wavelength + step, x)
    n_minus = get_n(wavelength - step, x)
    dn_dlam = (n_plus - n_minus) / (2 * step)
    return n - wavelength * dn_dlam
### on est rendu à D_w
x_cladding = 0
n_core = get_n(lambdas, x_core)
n_cladding = get_n(lambdas, x_cladding)
n2g = get_n_group(lambdas, x_cladding)

delta = (n_core**2 - n_cladding**2) / (2*n_core**2) # delta = (n_core^2 - n_cladding^2) / (2*n_core^2)

NA = np.sqrt(n_core**2 - n_cladding**2)
V = 2 * np.pi * r_coeur / lambdas * NA
u_infty = 2.405  # Valeur de u pour le mode fondamental LP01
#### Faut tu changer u si on  a v > 2.405??????????
u = u_infty*(V/(V+1))*(1-u_infty**2/(6*(V+1)**3)-u_infty**4/(20*(V+1)**5)) # Approximation de u pour V > 2.405
w = np.sqrt(V**2 - u**2)
l = 0
psi = kn(l, w)**2 / (kn(l + 1, w) * kn(l - 1, w))

Vb_prime = 1-(u/V)**2*(1-2*psi)
V_Vb_primeprime = 2*(u/V)**2*(psi*(1-2*psi)+2/w *(w**2+u**2*psi)*np.sqrt(psi)*(psi+np.sqrt(psi)/w-1))


Dw = delta * (dm*Vb_prime-V_Vb_primeprime/(c*lambdas*1e-12)*n2g**2/n_core)
D_total = dm + Dw

f_dispersion = interp1d(lambdas, D_total, kind='cubic')
zdw = brentq(f_dispersion, lambdas[0], lambdas[-1])

print(f"La longueur d'onde de dispersion nulle (ZDW) est : {zdw:.4f} µm")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(lambdas, dm, 'k-.', label="$D_M$", lw=2)
ax.plot(lambdas, Dw, 'k:', label="$D_w$", lw=2)
ax.plot(lambdas, D_total, label="$D$", color='black', lw=2)

ax.set_xlabel("Longueur d'onde $\lambda$ (µm)", fontsize=14)
ax.xaxis.set_label_coords(0.5,0.05)
ax.set_ylabel("Dispersion [ps/(nm$\cdot$km)]", fontsize=14)

ax.set_xticks(np.arange(1, 1.51, 0.1))
ax.set_yticks(np.arange(-40, 40, 20))

ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)

ax.tick_params(axis='both', direction='in')

ax.plot(zdw, 0, 'k^', markersize=10)
ax.annotate(f'ZDW = {zdw:.3f} µm', xy=(zdw, 0), xytext=(zdw-0.04, 7))
plt.grid(which='both', alpha=1)
ax.legend()

L = 20000
T0 = 10e-12
D_15 = f_dispersion(1.5)

beta2 = (1.5e-6)**2*D_15/(-2*np.pi*c)
Ld = T0**2/np.abs(beta2)
T1 = T0*np.sqrt(1+(L/Ld)**2)
print(D_15)

















plt.show()