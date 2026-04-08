import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize
from scipy.special import jv, kn, jn_zeros

c = 3*10**8 # m/s
r_coeur = 5.28 # um

# Données du tableau de constantes
constantes = {
    "Si": {"A": [0.696166, 0.407942, 0.897479], "L": [0.068404, 0.116241, 9.896161]},
    "Ge": {"A": [0.806866, 0.718158, 0.854168], "L": [0.068972, 0.153966, 11.84193]}
}

def eq_transcendante(u, l, V):
    w = np.sqrt(V**2 - u**2)
    lhs = jv(l, u) / (u * jv(l-1, u))
    rhs = -kn(l, w) / (w * kn(l-1, w))
    return lhs - rhs

def get_n(wavelength, x):
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

def get_n_group(wavelength, x, step=0.001):
    n = get_n(wavelength, x)
    n_plus = get_n(wavelength + step, x)
    n_minus = get_n(wavelength - step, x)
    dn_dlam = (n_plus - n_minus) / (2 * step)
    return n - wavelength * dn_dlam

x_core = 0.02    #x: fraction molaire de GeO2
lambdas = np.linspace(1.0, 1.5, 1000)
k0 = 2*np.pi/lambdas

def compute_D(x_core, r_coeur, lambdas, x_cladding=0):
    k0 = 2*np.pi/lambdas
    n_core = get_n(lambdas, x_core)
    n_cladding = get_n(lambdas, x_cladding)
    n2g = get_n_group(lambdas, x_cladding)

    delta = (n_core**2 - n_cladding**2) / (2*n_core**2)
    NA = np.sqrt(n_core**2 - n_cladding**2)
    V = 2*np.pi*r_coeur/lambdas * NA
    u_infty = 2.405
    u = u_infty*(V/(V+1))*(1-u_infty**2/(6*(V+1)**3)-u_infty**4/(20*(V+1)**5))
    w = np.sqrt(np.maximum(1e-20, V**2 - u**2))
    gamma = w/r_coeur
    beta = np.sqrt(gamma**2 + (k0*n_cladding)**2)
    neff = beta/k0

    dn_dlam = np.gradient(neff, lambdas)
    d2n_dlam2 = np.gradient(np.gradient(neff, lambdas), lambdas)
    D = -lambdas/c * d2n_dlam2 * 1e12
    Ng = neff - lambdas * dn_dlam

    l = 0
    psi = kn(l, w)**2 / (kn(l+1, w) * kn(l-1, w))
    Vb_prime = 1-(u/V)**2*(1-2*psi)
    V_Vb_primeprime = 2*(u/V)**2*(psi*(1-2*psi)+2/w*(w**2+u**2*psi)*np.sqrt(psi)*(psi+np.sqrt(psi)/w-1))
    
    dm = np.array([calculate_material_dispersion(l, x_core) for l in lambdas])
    Dw = delta * (dm*Vb_prime - V_Vb_primeprime/(c*lambdas*1e-12)*n2g**2/n_core)
    D_total = dm + Dw

    return D_total, dm, Dw, Ng, V

D_total, Dm, Dw, Ng, V = compute_D(x_core, r_coeur, lambdas)

f_dispersion = interp1d(lambdas, D_total, kind='cubic')
zdw = brentq(f_dispersion, lambdas[0], lambdas[-1])
print(f"La longueur d'onde de dispersion nulle (ZDW) est : {zdw:.4f} µm")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, Dm, 'k-.', label="$D_M$", lw=2)
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
#plt.show()

################## B #################
L = 20000
T0 = 10e-12
D_15 = D_total[-1]*1e-6
Ng_15 = Ng[-1]
Tg = Ng_15/c*20000
beta2 = (1.5e-6)**2*D_15/(-2*np.pi*c)

Ld = T0**2/np.abs(beta2)
T1 = T0*np.sqrt(1+(L/Ld)**2)

################# C ##################
V_init = V[-1]
lambdas = np.linspace(1.0, 1.5, 1000)
def D_at_15(x, r_coeur,x_cladding):
    D_total, _, _, _, V = compute_D(x, r_coeur, lambdas, x_cladding)
    obj = abs(D_total[-1]) + abs(V[-1]-V_init)*10
    print(obj, "optimisation en cours...")
    return obj

res = minimize(lambda x: D_at_15(x[0], x[1], x[2]), np.array([0.05, 1, 0.01]), bounds=((0, 0.2), (1, 5), (0, 0.1)))
# res = minimize(lambda x: D_at_15(x[0], x[1], x[2]), np.array([0.10409373, 2.30174476, 0.0]), bounds=((0, 0.2), (1, 5), (0, 0.1)))
print(res.x)
D_total, Dm, Dw, Ng, V = compute_D(res.x[0], res.x[1], lambdas, res.x[2])
print(V_init, V[-1], D_total[-1])
#print(Dw)

D_15 = D_total[-1]*1e-6
Ng_15 = Ng[-1]
Tg = Ng_15/c*20000
beta2 = (1.5e-6)**2*D_15/(-2*np.pi*c)

Ld = T0**2/np.abs(beta2)
T1 = T0*np.sqrt(1+(L/Ld)**2)
print(Tg, T1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, Dm, 'k-.', label="$D_M$", lw=2)
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
plt.grid(which='both', alpha=1)
ax.legend()
plt.show()