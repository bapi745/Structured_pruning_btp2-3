"""
2D Discrete Cosine Transform (DCT) on a 2x2 matrix.

Demonstrates both manual computation using the DCT-II formula
and verification using scipy.fft.dctn.
"""

import numpy as np
from scipy.fft import dctn

# ── Input 2×2 matrix ────────────────────────────────────────────
matrix = np.array([
    [52, 55],
    [61, 59]
], dtype=float)

print("Input 2x2 Matrix:")
print(matrix)
print()

# ── Manual 2D DCT-II computation ────────────────────────────────
# DCT-II formula for an N×N matrix:
#   X(u,v) = alpha(u) * alpha(v) * sum over x,y of
#            f(x,y) * cos[pi*(2x+1)*u / (2N)] * cos[pi*(2y+1)*v / (2N)]
#
# where alpha(0) = sqrt(1/N), alpha(k) = sqrt(2/N) for k > 0

N = 2

def alpha(k):
    """Normalization factor for DCT-II."""
    return np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)

dct_manual = np.zeros((N, N))

for u in range(N):
    for v in range(N):
        total = 0.0
        for x in range(N):
            for y in range(N):
                total += (
                    matrix[x, y]
                    * np.cos(np.pi * (2 * x + 1) * u / (2 * N))
                    * np.cos(np.pi * (2 * y + 1) * v / (2 * N))
                )
        dct_manual[u, v] = alpha(u) * alpha(v) * total

print("2D DCT (Manual Computation):")
print(np.round(dct_manual, 4))
print()

# ── Verification using scipy ────────────────────────────────────
dct_scipy = dctn(matrix, type=2, norm='ortho')

print("2D DCT (scipy.fft.dctn verification):")
print(np.round(dct_scipy, 4))
print()

# ── Interpretation ──────────────────────────────────────────────
print("Coefficient Interpretation:")
print(f"  DC component  X(0,0) = {dct_manual[0,0]:.4f}  (average intensity × 2)")
print(f"  Horizontal     X(0,1) = {dct_manual[0,1]:.4f}  (horizontal frequency)")
print(f"  Vertical       X(1,0) = {dct_manual[1,0]:.4f}  (vertical frequency)")
print(f"  Diagonal       X(1,1) = {dct_manual[1,1]:.4f}  (diagonal frequency)")
