# Physics-Informed Neural Networks for Fluid Dynamics Using MATLAB

A MATLAB-based implementation of **Physics-Informed Neural Networks (PINNs)** for two-dimensional, steady, incompressible lid-driven cavity flow at Reynolds number **Re = 100**.

This project combines finite-difference Computational Fluid Dynamics (CFD), deep learning, automatic differentiation, and analytical boundary-condition enforcement to investigate neural-network approximations of the Navier–Stokes equations.

## Key Highlights

- Developed an independent finite-difference CFD reference solver.
- Implemented boundary-conforming streamfunction PINNs.
- Analytically enforced incompressibility and velocity boundary conditions.
- Investigated fourth-order automatic differentiation and gradient conflicts.
- Compared analytical boundary extensions with exponents **m = 2, 6, 8**.
- Achieved **0.093532 combined velocity RMSE** on the held-out CFD spatial test split.
- Reduced fourth-order transport residual RMSE by approximately **59.05%** through boundary-extension redesign under matched training conditions.

## Mathematical Formulation

The flow is modeled in a unit-square cavity:

$$
(x,y)\in[0,1]\times[0,1]
$$

For two-dimensional incompressible flow, continuity requires

$$
\frac{\partial u}{\partial x}+
\frac{\partial v}{\partial y}=0
$$

The streamfunction formulation defines velocity as

$$
u=\frac{\partial\psi}{\partial y},
\qquad
v=-\frac{\partial\psi}{\partial x}
$$

Vorticity is derived directly from the streamfunction:

$$
\omega=-\nabla^2\psi
$$

The remaining steady governing equation is

$$
u\frac{\partial\omega}{\partial x}
+
v\frac{\partial\omega}{\partial y}
-
\frac{1}{Re}\nabla^2\omega=0
$$

At **Re = 100**, the dimensionless viscosity is

$$
\nu=\frac{1}{100}=0.01
$$

## Boundary-Conforming PINN

The final neural-network architecture is:

**2 → 64 → 64 → 64 → 64 → 1**

It uses four hidden layers with 64 neurons each, tanh activations, and one scalar output.

The analytical streamfunction transformation is

$$
\psi(x,y)=y^m(y-1)g(x)
+256x^2(1-x)^2y^2(1-y)^2N_\theta(x,y)
$$

The smoothed moving-lid velocity is

$$
g(x)=
\left(1-e^{-(x/\delta)^2}\right)
\left(1-e^{-((1-x)/\delta)^2}\right),
\qquad \delta=0.075
$$

The selected architecture uses **m = 8**.

The transformation enforces the prescribed velocity boundary conditions by construction, including no-slip stationary walls and the smoothed moving lid. It also enforces continuity and the streamfunction–vorticity Poisson identity analytically.

## CFD Reference

An independent finite-difference streamfunction–vorticity solver was implemented using sparse matrix operations, pseudo-time iteration, upwind advection and implicit diffusion.

| Parameter | Value |
|---|---|
| Reynolds number | 100 |
| Final CFD grid | 129 × 129 |
| Lid smoothing | 0.075 |
| Solver iterations | 8,887 |
| Minimum streamfunction | −0.103099 |
| Primary vortex center | (0.617188, 0.742188) |

A refinement comparison was conducted using 65 × 65 and 129 × 129 grids.

## Boundary-Extension Comparison

Three streamfunction lifting exponents were investigated under matched CFD-supervised training conditions.

| Boundary exponent | Validation velocity RMSE | Transport residual RMSE |
|---|---:|---:|
| m = 2 | 0.158517 | 87.879 |
| m = 6 | **0.094774** | 41.325 |
| m = 8 | 0.095174 | **35.983** |

The m = 8 formulation reduced transport residual RMSE by approximately **59.05%** relative to m = 2 on the same 41 × 41 evaluation grid.

## Final Model Results

The physics-refined m = 8 model was selected.

| Metric | Result |
|---|---:|
| Validation combined velocity RMSE | 0.091464 |
| Test horizontal velocity RMSE | 0.074230 |
| Test vertical velocity RMSE | 0.109480 |
| Test combined velocity RMSE | **0.093532** |
| Test streamfunction RMSE | 0.013120 |
| Fourth-order transport RMSE | 35.375 |
| Sidewall transport RMSE | 64.927 |
| Interior transport RMSE | 3.245 |
| Maximum absolute transport residual | 250.776 |

### Primary Vortex Comparison

| Quantity | CFD reference | Final PINN |
|---|---|---|
| Vortex center | (0.617188, 0.742188) | (0.484375, 0.703125) |
| Minimum streamfunction | −0.103099 | −0.098381 |

The vortex-center displacement is approximately **0.13844 cavity-length units**.

## Software

- MATLAB R2026a Update 4
- Deep Learning Toolbox
- MATLAB sparse matrix and numerical computing functions

## Implementation Workflow

The project includes the following major stages:

1. Finite-difference CFD reference solution.
2. CFD dataset generation and partitioning.
3. Boundary-conforming PINN development.
4. CFD-assisted neural-network training.
5. Analytical boundary-extension comparison.
6. Fourth-order physics-informed refinement.
7. Final comparative evaluation.

The original MATLAB scripts use local Windows paths. These must be configured for the user's environment before running the complete workflow.

Training from scratch requires the appropriate initialization and dataset-preparation scripts, as well as the main training and evaluation programs. Saved model checkpoints can be distributed separately when needed.

## Limitations

The model is a research prototype and is **not a fully converged Navier–Stokes solution**.

Significant fourth-order transport residuals remain near the sidewalls. The predicted primary vortex center also differs from the independent CFD reference.

The final experiments focus on Reynolds number 100 with a smoothed-lid boundary condition. Test samples originate from the same numerical CFD flow field and therefore evaluate spatial interpolation rather than generalization to unseen flow conditions.

## Conclusion

This project demonstrates how analytical boundary-extension design, CFD-assisted training and physics-informed optimization influence the accuracy of neural fluid-flow models.

The redesigned streamfunction PINN improves CFD velocity reconstruction and substantially reduces fourth-order governing-equation residuals compared with the original architecture, while highlighting the challenges of achieving complete physical consistency.

## Author

**Priyankar Das**

Electronics and Telecommunication Engineering  
Army Institute of Technology, Pune

[GitHub Profile](https://github.com/priyankardas2007)
