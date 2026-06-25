# Mathematical Formulation — gf-calculation

CG-SEM forward solver for elastic wave propagation + Green's function extraction.

---

## 1. Governing Equation

Elastic wave equation (second-order hyperbolic) in domain Ω ⊂ ℝ³:

<center>ρ ∂²u/∂t² = ∇·σ + f</center>

- **ρ**: density (kg/m³)
- **u** = (u_x, u_y, u_z): displacement field (m)
- **σ**: Cauchy stress tensor (Pa)
- **f**: body force / source term (N/m³)

Isotropic constitutive law (Hooke's law):

<center>σ = λ·tr(ε)·I + 2μ·ε</center>

- **λ** = ρ(v_p² − 2v_s²): first Lamé parameter (Pa)
- **μ** = ρ·v_s²: shear modulus (Pa)
- **ε** = ½(∇u + ∇uᵀ): infinitesimal strain tensor
- **v_p**: P-wave velocity (m/s)
- **v_s**: S-wave velocity (m/s)

Boundary conditions:
- **Free surface**: z = z_min, σ·n̂ = 0 (natural boundary, no explicit enforcement needed in weak form)
- **Absorbing (PML)**: other domain faces, convolutional PML

---

## 2. Weak Form & SEM Discretization

### 2.1 Weak Form

Multiplying the governing equation by a test function **w** and integrating by parts (ignoring boundary terms from the free surface; PML handled separately):

<center>∫_Ω ρ·w·∂²u/∂t² dΩ + ∫_Ω ∇w : σ dΩ = ∫_Ω w·f dΩ</center>

### 2.2 Spatial Discretization

Domain Ω partitioned into N_e non-overlapping hexahedral elements Ω_e. Each element mapped from reference cube [-1,1]³ via:

<center>x(ξ,η,ζ) = Σ_a N_a(ξ,η,ζ)·x_a</center>

where N_a are trilinear shape functions and x_a are the 8 physical corner coordinates.

### 2.3 GLL Quadrature

SEM uses Gauss-Lobatto-Legendre (GLL) quadrature with N+1 points per axis. Nodes coincide with quadrature points (collocation).

| Quantity | Symbol | Definition |
|----------|--------|-----------|
| Polynomial order | N | User-specified (testing: N=3, production: N=5) |
| GLL points per axis | NGLL = N+1 | 4 (test) / 6 (prod) |
| GLL nodes per element | NGLL³ | 64 (test) / 216 (prod) |
| Total GLL points | ξ_i, i=0..N | Roots of P'_N(ξ); endpoints −1, +1 |
| Quadrature weights | w_i | 2/(N(N+1)[P_N(ξ_i)]²) |
| Derivative matrix | D_ij = ℓ'_j(ξ_i) | Off-diagonal: P_N(ξ_i)/[P_N(ξ_j)(ξ_i−ξ_j)]; Diagonal endpoints: ∓N(N+1)/4; Diagonal interior: 0 |

P_N(x) computed via Bonnet's recurrence:

<center>P₀ = 1, P₁ = x, (k+1)P_{k+1} = (2k+1)xP_k − kP_{k-1}</center>

Lagrange basis on GLL nodes:

<center>ℓ_i(ξ) = Π_{k≠i} (ξ − ξ_k)/(ξ_i − ξ_k)</center>

Satisfying ℓ_i(ξ_j) = δ_ij (collocation property).

### 2.4 Discrete Displacement Field

<center>u(x) ≈ Σ_e Σ_{ijk} u^{e}_{ijk}·ℓ_i(ξ)·ℓ_j(η)·ℓ_k(ζ)·J⁻¹_e(ξ,η,ζ)</center>

After mapping, the global discrete system:

<center>M·ü + K·u = f</center>

- **M**: global mass matrix (diagonally lumped)
- **K**: global stiffness matrix (never assembled explicitly — matrix-free)
- **f**: source force vector

---

## 3. Element-Level Computation (Matrix-Free)

### 3.1 Per-Element Geometry

For each GLL node (i,j,k) of element e, precomputed and stored in partition_{r}.h5:

#### Jacobian Matrix

Physical coordinates via 3D tensor-product Lagrange interpolation:

<center>x(ξ_i,η_j,ζ_k) = Σ_p Σ_q Σ_r x^{e}_{pqr}·ℓ_p(ξ_i)·ℓ_q(η_j)·ℓ_r(ζ_k)</center>

Jacobian J = ∂x/∂ξ (3×3):

<center>J_{mn} = Σ_{pqr} x^{e}_{pqr,m}·ℓ'_p(ξ_i)ℓ_q(η_j)ℓ_r(ζ_k)·δ_{n,1}\,+ ...</center>

Determinant: det(J) > 0 for well-formed elements.

#### Inverse Jacobian (dξ/dx)

<center>dξ_i/dx_j = (J⁻¹)_{ij}</center>

Stored as 9-component flat array per GLL node: dξ/dx, dη/dx, dζ/dx, dξ/dy, dη/dy, dζ/dy, dξ/dz, dη/dz, dζ/dz.

### 3.2 Displacement Gradient in Reference Space

Using the collocation derivative:

<center>∂u/∂ξ|_i = Σ_s D_is·u_{s,j,k}</center>
<center>∂u/∂η|_j = Σ_s D_js·u_{i,s,k}</center>
<center>∂u/∂ζ|_k = Σ_s D_ks·u_{i,j,s}</center>

### 3.3 Physical Gradient (Chain Rule)

<center>∂u_l/∂x_m = (∂u_l/∂ξ)·(∂ξ/∂x_m) + (∂u_l/∂η)·(∂η/∂x_m) + (∂u_l/∂ζ)·(∂ζ/∂x_m)</center>

### 3.4 Strain Tensor

<center>ε_lm = ½(∂u_l/∂x_m + ∂u_m/∂x_l)</center>

Voigt ordering (6 components): ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz.

### 3.5 Stress Tensor (Isotropic)

<center>σ_lm = λ·δ_lm·ε_kk + 2μ·ε_lm</center>

where ε_kk = ε_xx + ε_yy + ε_zz is the trace.

### 3.6 Internal Force (Element Residual)

At each quadrature node (i,j,k), the contribution of stress σ to the residual at all nodes (s,t,u) of the element:

<center>r^{e}_{stu,m} += σ_{mn}·∂N_{stu}/∂x_n · det(J) · w_i·w_j·w_k</center>

The gradient of basis function N_{stu} at node (i,j,k) is:

<center>∂N_{stu}/∂x_n = D^{ξ}_{si}·ℓ_t(η_j)·ℓ_u(ζ_k)·∂ξ/∂x_n + ℓ_s(ξ_i)·D^{η}_{tj}·ℓ_u(ζ_k)·∂η/∂x_n + ℓ_s(ξ_i)·ℓ_t(η_j)·D^{ζ}_{uk}·∂ζ/∂x_n</center>

Sign: **negative** accumulation (r = −K·u), residual = −internal force.

The summation over all elements assembles the global residual:

<center>r_global = Σ_e r^{e}  (additive assembly at shared GLL nodes)</center>

---

## 4. Lumped Mass Matrix

Diagonal mass matrix (GLL spectral element):

<center>M^{e}_{ijk} = ρ_{ijk}·det(J_{ijk})·w_i·w_j·w_k</center>

One scalar per GLL node — all 3 displacement components at that node share the same lumped mass. No mass matrix inversion needed beyond scalar division.

---

## 5. Time Integration: Newmark Explicit (β=0, γ=½)

### 5.1 Predictor Step

Given (u_n, v_n, a_n) at time t_n:

<center>ũ = u_n + Δt·v_n + ½Δt²·a_n</center>
<center>ṽ = v_n + ½Δt·a_n</center>

### 5.2 Residual Evaluation

<center>r(ũ) = −K·ũ + f(t_{n+1})</center>

(negative stiffness term + source force)

### 5.3 Corrector Step

<center>a_{n+1} = M⁻¹·r(ũ)</center>
<center>v_{n+1} = ṽ + ½Δt·a_{n+1}</center>
<center>u_{n+1} = ũ  (no displacement correction when β=0)</center>

### 5.4 Stability

Explicit central difference scheme. Conditionally stable — CFL constraint:

<center>Δt ≤ h_min / v_p_max</center>

where h_min is the minimum Euclidean distance between adjacent GLL nodes within any element.

---

## 6. C-PML (Convolutional Perfectly Matched Layer)

### 6.1 Damping Profile

Simplified ramp profile (full C-PML with K, α, convolution coefficients deferred):

For each PML element and face direction, damping increases linearly from 0 at the PML-entry interface to 1.0 at the physical domain boundary:

<center>d(x_axis) = clamp(|x_axis − pml_start| / pml_width, 0, 1)</center>

### 6.2 Application to Velocity

<center>v_i ← v_i − d(node)·v_i</center>

All 3 DOF components at a node share the same damping coefficient. Non-PML elements: d = 0 everywhere (no effect).

Note: full C-PML implementation with stretched-coordinate memory variables (21 arrays = 39 scalars per GLL node per PML element) is deferred. When implemented, it will follow the second-order recursive convolution formulation of Wang et al. (2006, eq. 21, θ=1/8), matching SPECFEM3D.

---

## 7. Source Injection

### 7.1 Point Force

Single force source at physical position (x_s, y_s, z_s). Source located via Newton iteration over natural coordinates:

<center>ξ^{(k+1)} = ξ^{(k)} − J⁻¹(ξ^{(k)})·(x(ξ^{(k)}) − x_s)</center>

where J is interpolated from GLL nodes via Lagrange basis.

### 7.2 Source Time Function (STF)

User-defined Python callable `stf_func(t_s)`, pre-evaluated at solver_dt intervals:

<center>STF[n] = stf_func(n·Δt)</center>

Stored as array in config.h5 — no runtime evaluation.

### 7.3 Force Distribution

Force distributed to GLL nodes via Lagrange interpolation weights:

<center>F_{ijk} = STF[n] · w_{ijk} · ê_dir</center>

where w_{ijk} = ℓ_i(ξ_s)·ℓ_j(η_s)·ℓ_k(ζ_s) are the Lagrange basis values at the source natural coordinates. Weights normalized so Σ w_{ijk} = 1 across all sharing surface elements.

### 7.4 Force Direction

Three orthogonal force directions (x, y, z) run independently. Each produces a 3×3 strain Green's tensor column. Direction passed via CLI `--direction {x,y,z}`, not embedded in config file.

---

## 8. MPI Halo Exchange

CG-SEM assembly across MPI ranks at shared GLL interface nodes:

1. Precomputed exchange patterns per neighbor: `send_dof_indices` and `recv_dof_indices`
2. Pack: copy local DOF values to contiguous send buffer
3. Non-blocking MPI_Isend / MPI_Irecv per neighbor
4. MPI_Waitall
5. **Accumulate**: add received values to local field at recv_dof_indices

<center>field[recv_idx[j]] += recv_buf[j]  (additive, not overwrite)</center>

This ensures contributions from neighbor ranks to shared GLL nodes are summed, implementing the global assembly without constructing the global system matrix.

---

## 9. Strain Computation

After each Newmark correct step, element-wise strain computed from the corrected displacement field:

<center>ε_lm = ½(∂u_{new,l}/∂x_m + ∂u_{new,m}/∂x_l)</center>

Voigt storage order (6 components per GLL node): ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz.

L2 strain smoothing (matches SPECFEM3D convention):
<center>ε_smooth = M⁻¹ · Σ_e ∫ N·ε_e dΩ</center>

Smoothed strain written to HDF5 snapshots at stride intervals.

---

## 10. Strain Green's Function Extraction

### 10.1 Green's Tensor Assembly

Three forward runs (direction x, y, z) produce strain field ε^{(d)} for d ∈ {x,y,z} at all GLL nodes and time steps:

<center>G_{ij}(t, x) = ε^{(j)}_i(t, x) / F_j(ω)</center>

where i = 1..6 (strain component in Voigt order) and j = x,y,z (force direction).

Full strain Green's tensor at each GLL node: 3 force directions × 6 strain components = 18 entries.

### 10.2 Spatial Tiling

Output tiled by element range (lat/lon bounding boxes). Each `greenfun/tile_{i}.h5` stores the full Green's tensor for a contiguous block of physical-domain elements (PML elements excluded).

### 10.3 No Receivers

Green's functions extracted at all GLL nodes. No receiver positions, no receiver search, no position interpolation in postprocess. This is a design constraint — all strain response data is preserved.

---

## 11. Timestep Derivation

From user-specified `output_dt_s` and material-dependent CFL limit:

<center>cfl_dt = cfl_safety × h_min / v_p_max</center>

Search for smallest integer stride such that:

<center>solver_dt = output_dt_s / stride ≤ cfl_dt</center>

<center>snapshot_stride = output_dt_s / solver_dt ∈ ℤ</center>

<center>nsteps = ⌈total_duration_s / solver_dt⌉</center>

Snapshot written when `step % snapshot_stride == 0`.

---

## 12. Validation Constraints

All checked at preprocess time:

| Check | Condition |
|-------|-----------|
| Mesh | det(J) > 0 at all GLL nodes |
| Material | v_p > 0, v_s ≥ 0, ρ > 0 at all GLL nodes |
| CFL | solver_dt ≤ cfl_dt |
| Source | x, y within domain; STF finite & non-NaN |
| Storage | estimated disk usage ≤ storage_limit_gb |
| PML | ≥ 2 elements per absorbing face (warn if thinner) |

---

## 13. Summary of Key Equations

| # | Equation | Meaning |
|---|----------|---------|
| 1 | ρ·ü = ∇·σ + f | Elastic wave equation |
| 2 | σ = λ·tr(ε)·I + 2μ·ε | Isotropic stress-strain |
| 3 | ε = ½(∇u + ∇uᵀ) | Infinitesimal strain |
| 4 | Du/Dξ = D·u | Reference gradient via GLL derivative matrix |
| 5 | ∂u/∂x = (∂u/∂ξ)·(∂ξ/∂x) | Physical gradient via chain rule |
| 6 | r = −∫ ∇N·σ dΩ | Matrix-free internal force |
| 7 | M_{ijk} = ρ·det(J)·w_i·w_j·w_k | Lumped mass |
| 8 | ũ = u_n + Δt·v_n + ½Δt²·a_n | Newmark predictor |
| 9 | a_{n+1} = M⁻¹·r(ũ) | Newmark corrector (acceleration) |
| 10 | v_{n+1} = v_n + ½Δt·a_n + ½Δt·a_{n+1} | Newmark corrector (velocity) |
| 11 | Δt ≤ h_min / v_p_max | CFL stability |
| 12 | G_{ij} = ε^{(j)}_i | Strain Green's tensor |

---

*Generated from source code: forward/include/gf/*.hpp, forward/src/*.cpp, preprocess/*.py.*