# =========================================================
# code block 
# =========================================================
#import sys
#print ("Codium running Python " + sys.version)

import time
import numpy as np
import matplotlib.pyplot as plt



startTime = time.time()

def J(omega, prefactor=0.005,D=1):
    "Spectral function of choice"
    return prefactor*np.sqrt(1.-(omega/D)**2)

n = 300 # an arbitrary choice for now, not too large
N  = n+1

H = np.zeros((N,N),dtype=np.complex128)
epsilon = 0.0 # our scale of energies

D = 1.0 # choice of the bandwidth
omegas = np.linspace(-D, D,n) #discretisation of the bath
Js = J(omegas,D=D)

# first column
# scaled for correct thermodynamic limit, since the rates depend on j_i^2
H[1:,0] = np.sqrt(Js)/np.sqrt(n)
# first row
H[0,1:] = np.sqrt(Js)/np.sqrt(n)
# diagonal
np.fill_diagonal(H, [epsilon]+list(omegas))
#print(H)

# plot the spectral function and its discretisation
"""_x = np.linspace(omegas.min(),omegas.max(),1000)
plt.plot(_x, J(_x))
plt.plot(omegas, Js, 'o')
plt.gca().set(xlabel="$omega$", ylabel="$J(omega)$");
""" # TODO: Uncomment plotting as necessary
# =========================================================
# code block 
# =========================================================

C0 = np.zeros_like(H)
# choosing the temperature
beta = 1.0
mu =0
fermi = lambda x:1/(1+np.exp(beta*x-beta*mu))

density_0 = 0
np.fill_diagonal(C0,[0]+[fermi(o) for o in omegas])

C0

# =========================================================
# code block 
# =========================================================

from scipy.linalg import eigh
e_vals, U = eigh(H)  # H = U @ diag(e_vals) @ U.conj().T
# Step 1: Diagonalize H
e_vals, U = eigh(H)  # H = U @ diag(e_vals) @ U.conj().T

# Step 2: Transform C0 into the energy eigenbasis
C_energy_basis = U.conj().T @ C0 @ U

# Step 3: Check diagonality
off_diagonal_norm = np.linalg.norm(C_energy_basis - np.diag(np.diagonal(C_energy_basis)))
#print("Off-diagonal norm of C in energy basis:", off_diagonal_norm)

# =========================================================
# code block 
# =========================================================
"""
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(1,2,figsize=(12,5))
im0 = ax[0].matshow(np.log10(np.abs(C0) + 1e-6), cmap="inferno")
im1 = ax[1].matshow(np.log10(np.abs(C_energy_basis) + 1e-6), cmap="inferno")
ax[0].set_title("C in the original basis")
ax[1].set_title("C in the energy basis")
fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])
""" # TODO: Uncomment plotting as necessary

# =========================================================
# code block 
# =========================================================

import scipy
def unitary(t):
    return np.matrix(scipy.linalg.expm(-1j*H*t))

# =========================================================
# code block 
# =========================================================

def evolve(C,t):
    U = unitary(t)
    return U@C@U.H

# =========================================================
# code block 
# =========================================================

dt = 0.1
tmax = 1000.
times = np.arange(0,tmax, dt)

# =========================================================
# code block 
# =========================================================

# supposedly faster iterative calculation
Udt = unitary(dt)
UdtT = Udt.H
Css ={0:C0}
key = 0
for t in times[1:]:
    Css[t] = Udt@Css[key]@UdtT
    key = t

# =========================================================
# code block  - pytorch alternative to above
# =========================================================
"""
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU acceleration.")
else:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU.")

Udt = unitary(dt)
UdtT = torch.from_numpy(Udt.H)
Udt = torch.from_numpy(unitary(dt))
Css ={0:torch.from_numpy(C0)}
key = 0
for t in times[1:]:
    Css[t] = Udt@Css[key]@UdtT
    key = t
"""
# =========================================================
# code block 
# =========================================================

density = np.array([ val [0,0] for key, val in Css.items()])

# =========================================================
# code block 
# =========================================================
"""
plt.plot(times, density.real)
plt.gca().set(
    xlabel = "Time",
    ylabel = "Occupation on the system"
)""" # TODO: Uncomment plotting as necessary

# =========================================================
# code block 
# =========================================================
"""
for key,val in Css.items():
    plt.plot(omegas,np.diagonal(val[1:,1:]), color= plt.cm.coolwarm(key/tmax))

u = np.linspace(-1.2, 1.2, 50)
plt.plot(u,fermi(u), '.',color='k',ms=4)
# plt.plot(omegas,fermi(omegas), '-',color='white', lw=2)
plt.gca().set(
    xlabel = "$omega$",
    ylabel = "Occupation"
)
""" # TODO: Uncomment plotting as necessary
endTime= time.time()
timeDiff = endTime - startTime
print("All done! Time elapsed: ", str(timeDiff))