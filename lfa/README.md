# Local Fourier analysis of the red–black Gauss–Seidel smoother

`lfa_rbgs.py` computes the smoothing factors of pointwise red–black Gauss–Seidel (RBGS) given in
Chapter 3 of the thesis (table "Smoothing factors of pointwise RBGS on the present mesh").

```
python3 lfa_rbgs.py        # requires NumPy
```

## Method

The pressure operator is the five-point Laplacian. With the coupling ratio $c = a_N/a_E$
($c = 1$ for square cells, $c = 1/9$ for cells with $\Delta y = 3\Delta x$), the Fourier symbol of the
Jacobi iteration is

$$
A(\boldsymbol\theta) = \frac{\cos\theta_x + c\,\cos\theta_y}{1 + c}.
$$

A red–black sweep couples the mode $\boldsymbol\theta$ with the mode $\boldsymbol\theta + (\pi,\pi)$,
whose Jacobi symbol is $-A$. On this pair of modes, the red and the black half-sweep are the matrices

$$
\hat S_{\mathrm R} = \frac12 \begin{pmatrix} A+1 & -(A+1) \\ A-1 & 1-A \end{pmatrix}, \qquad
\hat S_{\mathrm B} = \frac12 \begin{pmatrix} A+1 & A+1 \\ 1-A & 1-A \end{pmatrix},
$$

and one sweep is $\hat S_h = \hat S_{\mathrm B}\,\hat S_{\mathrm R}$.

A mode is a high-frequency mode if the next coarser grid cannot represent it. With coarsening
ratios $q_x$ and $q_y$, this is the case when $|\theta_x| \ge \pi/q_x$ or $|\theta_y| \ge \pi/q_y$; a
direction that is not coarsened ($q = 1$) adds no high-frequency modes. The projection $\hat Q$ is a
diagonal matrix with entry one for a high-frequency mode and zero otherwise. For $s$ sweeps, the
smoothing factor per sweep is

$$
\mu_{\mathrm s}^{(s)} = \Big( \max_{\boldsymbol\theta}\ \rho\big( \hat Q\, \hat S_h^{\,s}\, \hat Q \big) \Big)^{1/s},
$$

where $\rho$ is the spectral radius. The script evaluates it on $361 \times 361$ frequencies.

## Output

```
square, q = 2 ['0.25', '0.40', '0.50']
square, q = 3 ['0.56', '0.56', '0.56']
dy = 3 dx, q = 2 ['0.81', '0.81', '0.81']
dy = 3 dx, q = 3 ['0.90', '0.90', '0.90']
dy = 3 dx, x only, q = 3 ['0.30', '0.40', '0.50']
```

The three columns are $s = 1$, 4 and 6. For $q = 2$, the script reproduces the published values
0.25 for square cells and 0.81 for $\Delta y = 3\Delta x$.
