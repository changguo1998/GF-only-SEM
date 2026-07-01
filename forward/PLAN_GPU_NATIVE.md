# GPU-Native Single-GPU Mode Plan

## Goal
Keep all state vectors on GPU permanently during time loop. No H2D/D2H copy per step.
Only copy to host for I/O (snapshots, restart, progress).

## Current bottleneck (2 cudaMemcpy/step)

```
HOST: newmark_predict (O(n) vec op)
      fill(residual) (O(n) vec op)
                            → copy_u_to_device (H2D)
                            → element_residual_kernel (GPU)
                            → copy_r_to_host (D2H)
      apply_pml_damping (O(n) vec op)
      source_injection (O(n_src) scatter)
      newmark_correct (O(n) vec op)
      strain_compute (O(n_rec) gradient)
```

## Target (0 cudaMemcpy/step for compute)

```
GPU: newmark_predict_kernel
     cudaMemset(d_r, 0)
     element_residual_kernel (already GPU)
     pml_damping_kernel
     source_injection_kernel
     newmark_correct_kernel
     strain_kernel → copy D2H only at snapshot steps
```

## Files to create/modify

### NEW: `forward/src/cuda_step.cu`
CUDA kernels + host wrappers for:
- `cuda_newmark_predict(d_disp, d_vel, d_acc, d_disp_tilde, dt, beta, n_dof)`
- `cuda_pml_damping(d_pml, d_vel, n_dof, n_nodes)`
- `cuda_source_injection(d_residual, d_src_weights, stf_val, d_src_elem_offsets, dir, n_src, n_node)`
- `cuda_newmark_correct(d_mass, d_disp, d_vel, d_acc, d_residual, dt, gamma, n_dof)`
- `cuda_compute_recorded_strain(d_disp, d_dxi_dx, d_D, NGLL, d_rec_elem, d_rec_corner, d_strain, n_vertices, n_node)`

### NEW: `forward/include/gf/cuda_step.hpp`
Host-callable function declarations + `CudaDeviceState` struct for device-side state management.

### MODIFY: `forward/src/solver.cpp`
Add `#ifdef GF_WITH_CUDA` path in time loop:
```
// Allocate device state (once)
// Per step: launch GPU kernels for all compute
// For I/O only: copy state back to host
```

### MODIFY: `forward/CMakeLists.txt`
Add `src/cuda_step.cu` to `libgf_cuda_nompi` and `libgf_cuda`.