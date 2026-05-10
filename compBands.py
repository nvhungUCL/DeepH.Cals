import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from numpy.linalg import inv, norm
from scipy.linalg import eigvals

# --- Initial Parameters ---
KPS = 200
EF = 4.785239202
outNAME = 'pBands_MoS2bilayer.mat'

# Path Pk: Γ -> M -> K -> Γ
Pk = np.array([
    [0,   0,   0,   1/2, 0,   0],  # Γ -> M
    [1/2, 0,   0,   1/3, 1/3, 0],  # M -> K
    [1/3, 1/3, 0,   0,   0,   0]   # K -> Γ
])

# Define Lattice Vectors (v123)
v1 = np.array([1, 0, 0])
v2 = np.array([-1/2, np.sqrt(3)/2, 0])
v123 = np.vstack([v1, v2]) * 3.14753974
v123 = np.vstack([v123, [0, 0, 50]]) # Add z-direction

# Hermitian coefficients logic
HM_val = 1
HMco = np.array([1, 0]) - np.array([1, -1]) * HM_val / 2

# --- Reciprocal Space Setup ---
u123 = inv(v123) * 2 * np.pi
v123 = v123.T
u123 = u123.T

# Calculate total path length in reciprocal space to scale KPS
lkp = sum(norm((Pk[k, 3:6] - Pk[k, 0:3]) @ u123) for k in range(Pk.shape[0]))
KPS_scaled = KPS / lkp

# --- Generate k-path (kv) ---
kv = []
for k in range(Pk.shape[0]):
    start_pt = Pk[k, 0:3] @ u123
    end_pt = Pk[k, 3:6] @ u123
    diff_vec = end_pt - start_pt

    # Calculate number of points for this segment
    nk_seg = int(round(norm(diff_vec) * KPS_scaled))

    # Logic to avoid duplicating points at segment junctions
    # and ensuring the final point of the whole path is included
    is_last_segment = (k == Pk.shape[0] - 1)
    num_pts = nk_seg + (1 if is_last_segment else 0)

    dk = diff_vec / nk_seg
    segment = start_pt + np.arange(num_pts)[:, None] * dk
    kv.append(segment)

kv = np.vstack(kv)
nk = kv.shape[0]

# --- Load Hamiltonians ---
## Note: Ensure Hamiltonians.mat is in your working directory
#data = loadmat('Hamiltonians.mat')
#ijce = data['ijce'].astype(float)
#H0n = data['H0n'].squeeze() # squeeze handles MATLAB cell-to-array conversion
#S0n = data['S0n'].squeeze()
data = loadmat('Assembled_Hamiltonians.mat')
H0n  = data['chunks']
data = loadmat('Assembled_Overlaps.mat')
S0n  = data['chunks']
ijce = data['ijcell'].astype(float)

Eb = []

# --- Calculation Loop ---
print(f"Computing {nk} k-points...")
for i in range(nk):
    kt = kv[i, :] @ v123

    Hk = 0j
    Sk = 0j

    # Sum over R vectors (ip, iq, ik)
    for idx in range(ijce.shape[0]):
        R_vec = ijce[idx, 0:3]
        phase = np.exp(1j * np.dot(R_vec, kt))

        # Convert to dense if they are sparse
        #h_matrix = H0n[idx]
        #s_matrix = S0n[idx]
        h_matrix = H0n[:,:,idx]
        s_matrix = S0n[:,:,idx]

        #if hasattr(h_matrix, "toarray"): h_matrix = h_matrix.toarray()
        #if hasattr(s_matrix, "toarray"): s_matrix = s_matrix.toarray()

        Hk += h_matrix * phase
        Sk += s_matrix * phase

    # Symmetrization
    Hk = HMco[0] * Hk + HMco[1] * Hk.conj().T
    Sk = HMco[0] * Sk + HMco[1] * Sk.conj().T

    # Now eigvals will accept them!
    Ek = eigvals(Hk, Sk)
    Eb.append(np.sort(Ek.real) - EF)

Eb = np.array(Eb).T # Shape: [bands, k-points]

# --- Save and Plot ---
savemat(outNAME, {'Eb': Eb})

x = np.linspace(0, 1, Eb.shape[1])
plt.figure(figsize=(10, 8))
plt.plot(x, Eb.T, color='red', linewidth=1.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Fermi Level
plt.grid(True, linestyle=':')
plt.xlim([0, 1])
plt.ylim([-8, 8])
plt.title("Band Structure: $MoS_2$ Bilayer", fontsize=20)
plt.ylabel("$E - E_F$ (eV)", fontsize=18)
#plt.xticks([]) # Hide x-ticks as they represent the path
plt.show()
