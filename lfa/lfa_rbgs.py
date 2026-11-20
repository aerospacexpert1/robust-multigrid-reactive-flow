"""Local Fourier analysis of the red-black Gauss-Seidel smoother (thesis Appendix A).

Computes the smoothing factor of RBGS for the five-point Laplacian with coupling ratio
c = a_N / a_E (c = 1/9 for cells with dy = 3 dx), coarsening ratios qx, qy and s sweeps,
and prints the values of the smoothing-factor table in Chapter 3 of the thesis (s = 1, 4, 6).

Usage: python3 lfa_rbgs.py        (requires NumPy)
"""
import numpy as np

def smoothing_factor(c, qx, qy, s=1, n=361):
    """Smoothing factor of RBGS. c = a_N/a_E; qx, qy = coarsening ratios;
    s = number of sweeps; n = number of frequencies per direction."""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
    tx, ty = np.meshgrid(th, th, indexing="ij")
    A = (np.cos(tx) + c * np.cos(ty)) / (1.0 + c)      # Jacobi symbol
    def high(t, q):                     # mode not represented on the next grid
        return np.abs(t) >= np.pi / q if q > 1 else np.zeros(t.shape, bool)
    shift = lambda t: (t + 2.0 * np.pi) % (2.0 * np.pi) - np.pi    # theta + pi
    q1 = high(tx, qx) | high(ty, qy)
    q2 = high(shift(tx), qx) | high(shift(ty), qy)
    mu = 0.0
    for a, h1, h2 in zip(A.ravel(), q1.ravel(), q2.ravel()):
        SR = 0.5 * np.array([[a + 1, -a - 1], [a - 1, 1 - a]])   # red
        SB = 0.5 * np.array([[a + 1, a + 1], [1 - a, 1 - a]])    # black
        Q = np.diag([float(h1), float(h2)])
        M = Q @ np.linalg.matrix_power(SB @ SR, s) @ Q
        mu = max(mu, np.max(np.abs(np.linalg.eigvals(M))) ** (1.0 / s))
    return mu

rows = [("square, q = 2", 1.0, 2, 2), ("square, q = 3", 1.0, 3, 3),
        ("dy = 3 dx, q = 2", 1 / 9, 2, 2), ("dy = 3 dx, q = 3", 1 / 9, 3, 3),
        ("dy = 3 dx, x only, q = 3", 1 / 9, 3, 1)]
for name, c, qx, qy in rows:
    print(name, ["%.2f" % smoothing_factor(c, qx, qy, s) for s in (1, 4, 6)])
