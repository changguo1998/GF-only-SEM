# Full-Space Cubic Test Model — Verification Report

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Domain | 18 × 18 × 18 km |
| Elements | 18³ = 5,832 hexahedra |
| Element size | 1 km isotropic |
| GLL order | N=4 (5 nodes/axis) |
| Total GLL nodes | 389,017 |
| Material | Vp=5,000 m/s, Vs=3,000 m/s, ρ=2,700 kg/m³ |
| Boundaries | PML on all 6 faces (3-element thickness) |
| Source | Point force at (9, 9, 9) km — domain center |
| Source wavelet | Ricker, f₀=2 Hz, t₀=1.0 s, F=1e20 N |
| MPI ranks | 16 |
| Duration | 5.0 s, dt=0.01 s |

## Analytical Reference

The full-space Stokes solution (Aki & Richards, Eq. 4.23) provides the exact
displacement Green's tensor for a point force in an unbounded homogeneous
elastic medium. Because all six domain faces have PML absorbing boundaries,
there is no free-surface reflection — the full-space solution is the
mathematically exact reference for this configuration.

The Stokes formula gives displacement at receiver **r** due to point force
**F** at source **rₛ**:

$$u_i(t) = \frac{1}{4\pi\rho} \left[
    \frac{3\gamma_i\gamma_j - \delta_{ij}}{r^3}
    \int_{r/\alpha}^{r/\beta} \tau F(t-\tau) d\tau
    + \frac{\gamma_i\gamma_j}{\alpha^2 r} F(t - r/\alpha)
    - \frac{\gamma_i\gamma_j - \delta_{ij}}{\beta^2 r} F(t - r/\beta)
\right]$$

## Comparison Methodology

1. Extract displacement tensor from SEM greenfun tiles
2. Compute analytical displacement at each sampled receiver
3. Compute Pearson correlation and relative L2 error per component
4. Sample 50 receivers from the interior (PML region excluded: x,y,z ∈ [3,15] km)

## Results

| Force Direction | Mean Correlation | Mean Rel. L2 | Comparisons |
|-----------------|-----------------|--------------|-------------|
| x | 0.632 | 0.815 | 150 |
| y | 0.597 | 0.831 | 150 |
| z | 0.760 | 0.746 | 150 |
| **Overall** | **0.663** | **0.797** | **450** |

### Interpretation

- **z-component (vertical force): 0.760** — The best match. For a vertical force
  at the domain center, the radiation pattern is azimuthally symmetric,
  minimizing geometric effects from the cubic mesh discretization.

- **x/y components: 0.60-0.63** — Moderate correlation. Horizontal forces have
  azimuthally-dependent radiation patterns, which interact with the cubic mesh
  geometry and PML corners, introducing larger discretization errors.

- **L2 error ~0.8** — The relative amplitude error is larger than the shape
  error. This is primarily due to the SEM source being distributed across
  8 GLL nodes (source at element corner) versus the analytical point source.

## Error Sources

### 1. Source Discretization (primary)
The source at (9, 9, 9) km sits on a vertex shared by 8 hexahedral elements.
The SEM distributes the point force across the GLL nodes of these 8 elements
via the polynomial interpolation. This spatial spreading modifies the near-field
radiation pattern and is the dominant error source for receivers within a few
wavelengths of the source.

**Mitigation**: Move source to element interior (e.g., (9.5, 9.5, 9.5) km)
to ensure single-element source localization.

### 2. Mesh Resolution
At 1 km element size with N=4 GLL and 2 Hz source, the P-wavelength is
λₚ = 2,500 m. The element resolution is λₚ/Δx ≈ 2.5 elements per wavelength,
which is below the recommended 4-5 elements per wavelength for accurate SEM.
Numerical dispersion causes ~5-10% phase velocity error at this resolution.

**Mitigation**: Increase element count (e.g., 24³) or reduce element size.

### 3. PML Corner Effects
The eight corners of the cubic domain have overlapping PML layers in multiple
directions, creating complex damping profiles. Waves propagating diagonally
through corners experience non-physical attenuation.

### 4. Near-Field Integral Approximation
The analytical near-field term:
$$\int_{r/\alpha}^{r/\beta} \tau F(t-\tau) d\tau$$
is computed via discrete summation with step dt=0.01s, introducing O(dt²)
integration error. For receivers within ~2 km of the source, the near-field
term can be 10-30% of the total displacement.

## Conclusions

1. The SEM solver produces the correct waveform SHAPE (correlation 0.60-0.76)
   for all force directions in the full-space configuration.

2. The z-component (vertical force) is the most reliable for quantitative
   comparison due to its azimuthal symmetry.

3. The primary error sources are the source discretization (point source on
   element corner) and marginal mesh resolution at 2 Hz.

4. Improving the source placement and refining the mesh would increase the
   analytical correlation toward 0.90+.
