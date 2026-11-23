# Physics-Informed Neural Networks for 2D Lid-Driven Cavity Flow Using MATLAB

A MATLAB-based research implementation of **Physics-Informed Neural Networks (PINNs)** for two-dimensional, steady, incompressible lid-driven cavity flow at Reynolds number 100.

The project investigates how analytical boundary-condition enforcement, CFD-assisted learning, and boundary-extension design influence neural-network predictions and the satisfaction of governing fluid equations.

## Project Overview

Traditional computational fluid dynamics (CFD) methods solve fluid-flow governing equations through numerical discretization. Physics-Informed Neural Networks offer an alternative approach by incorporating governing physical equations into neural-network training.

This project develops and evaluates a boundary-conforming streamfunction PINN, using an independently implemented finite-difference CFD solver as a numerical reference.

The investigation includes:

- Finite-difference simulation of a two-dimensional lid-driven cavity.
- PINN modeling of velocity and vorticity fields.
- Streamfunction-based enforcement of incompressibility.
- Analytical enforcement of no-slip and moving-lid boundary conditions.
- CFD-supervised learning and physics-informed refinement.
- Analysis of fourth-order derivatives and transport residuals.
- Comparison of alternative boundary-extension functions.

## Mathematical Formulation

For a steady, incompressible, two-dimensional fluid, the Navier–Stokes equations are

\[
u\frac{\partial u}{\partial x}
+v\frac{\partial u}{\partial y}
=
-\frac{\partial p}{\partial x}
+\frac{1}{Re}\nabla^2u
\]

\[
u\frac{\partial v}{\partial x}
+v\frac{\partial v}{\partial y}
=
-\frac{\partial p}{\partial y}
+\frac{1}{Re}\nabla^2v
\]

subject to the continuity equation

\[
\frac{\partial u}{\partial x}
+\frac{\partial v}{\partial y}=0.
\]

The final neural model uses the streamfunction–vorticity formulation:

\[
u=\frac{\partial\psi}{\partial y},
\qquad
v=-\frac{\partial\psi}{\partial x}
\]

\[
\omega=-\nabla^2\psi.
\]

The remaining physics equation is the steady vorticity-transport equation:

\[
u\frac{\partial\omega}{\partial x}
+v\frac{\partial\omega}{\partial y}
-\frac{1}{Re}\nabla^2\omega=0.
\]

The Reynolds number is fixed at

\[
Re=100,\qquad \nu=0.01.
\]

### Boundary Conditions

The computational domain is a unit-square cavity:

\[
(x,y)\in[0,1]\times[0,1].
\]

The stationary walls satisfy

\[
u=v=0.
\]

The moving upper wall satisfies

\[
u(x,1)=g(x),\qquad v(x,1)=0
\]

where

\[
g(x)=
\left(1-e^{-(x/\delta)^2}\right)
\left(1-e^{-((1-x)/\delta)^2}\right),
\qquad \delta=0.075.
\]

This smoothed-lid boundary condition is used consistently in the PINN and the CFD reference. It differs from the classical discontinuous, uniformly moving lid problem.

## PINN Architecture

The selected neural network uses four hidden layers with 64 neurons per layer and hyperbolic tangent activation functions.

**Architecture:**

`2 → 64 → 64 → 64 → 64 → 1`

Inputs: spatial coordinates \(x,y\).

Output: a scalar neural correction \(N_\theta(x,y)\).

The boundary-conforming streamfunction is

\[
\psi(x,y)=
y^m(y-1)g(x)
+
256x^2(1-x)^2y^2(1-y)^2N_\theta(x,y).
\]

The final selected boundary exponent is

\[
m=8.
\]

This construction enforces the prescribed velocity boundary conditions analytically. Incompressibility and the streamfunction–vorticity Poisson relationship are also satisfied by construction, up to numerical differentiation error.

## CFD Reference

A finite-difference streamfunction–vorticity solver was independently implemented in MATLAB.

Its numerical approach uses:

- A structured Cartesian grid.
- Finite-difference spatial operators.
- Sparse linear-system solvers.
- Pseudo-time iteration toward steady state.
- Upwind treatment of vorticity advection.
- Implicit treatment of viscous diffusion.

Two grid resolutions were examined: 65 × 65 and 129 × 129.

The finer-grid reference produced:

| Quantity | Result |
|---|---:|
| Reynolds number | 100 |
| Grid resolution | 129 × 129 |
| Lid smoothing parameter | 0.075 |
| Minimum streamfunction | −0.103099 |
| Primary vortex center | (0.617188, 0.742188) |
| Solver iterations | 8,887 |

The 65 × 65 versus 129 × 129 grid comparison gave velocity RMSE differences of approximately 0.002634 for \(u\) and 0.002089 for \(v\).

This provides evidence of grid refinement consistency, although a formal grid-convergence-order study was not performed.

## Training Methodology

Development proceeded through successive implementations:

1. Initial velocity–pressure PINN.
2. Streamfunction–vorticity PINN.
3. Boundary-conforming streamfunction architecture.
4. Gradient-conflict and loss-balancing investigations.
5. CFD-assisted supervised learning.
6. Fourth-order streamfunction-only physics formulation.
7. Analytical boundary-extension comparison.
8. Physics-informed refinement of the selected architecture.

The final model uses CFD-supervised velocity and streamfunction losses together with a fourth-order vorticity-transport residual.

Gradient projection and controlled physics-gradient scaling were employed during the final refinement stage to reduce conflicts between supervised and physics objectives.

## Boundary-Extension Comparison

The analytical lifting function was generalized to

\[
f_m(y)=y^m(y-1).
\]

The exponents \(m=2,6,8\) were trained and compared.

### CFD Validation Accuracy

| Boundary exponent | Combined velocity RMSE |
|---|---:|
| \(m=2\) | 0.158517 |
| \(m=6\) | 0.094774 |
| \(m=8\) | 0.095174 |

### Fourth-Order Physics Accuracy

| Boundary exponent | Transport residual RMSE |
|---|---:|
| \(m=2\) | 87.879 |
| \(m=6\) | 41.325 |
| \(m=8\) | 35.983 |

All physics residual values in this comparison were evaluated using the same 41 × 41 interior grid.

The redesigned \(m=8\) model reduced transport residual RMSE by approximately **59.05%** relative to the matched \(m=2\) model.

This was an improvement in residual magnitude, not a claim of full convergence to a Navier–Stokes solution.

## Final Evaluation Results

The selected model is the physics-refined \(m=8\) streamfunction PINN.

| Evaluation metric | Final value |
|---|---:|
| Validation combined velocity RMSE | 0.091464 |
| Test horizontal velocity RMSE | 0.074230 |
| Test vertical velocity RMSE | 0.109480 |
| Test combined velocity RMSE | 0.093532 |
| Test streamfunction RMSE | 0.013120 |
| Fourth-order transport residual RMSE | 35.375 |
| Sidewall transport residual RMSE | 64.927 |
| Interior transport residual RMSE | 3.245 |
| Maximum absolute transport residual | 250.776 |

Relative to the unrefined \(m=8\) model, physics refinement improved validation velocity RMSE by approximately 3.90% and transport residual RMSE by approximately 1.69%, measured on the same respective evaluation sets.

### Primary Vortex Comparison

| Quantity | CFD reference | Final PINN |
|---|---|---|
| Vortex center | (0.617188, 0.742188) | (0.484375, 0.703125) |
| Minimum streamfunction | −0.103099 | −0.098381 |

The vortex-center displacement is approximately **0.13844 cavity-length units**.

## Software Requirements

- MATLAB R2026a (development environment).
- Deep Learning Toolbox.
- MATLAB functionality for sparse matrices, numerical differentiation and visualization.

The original experiments were performed in MATLAB R2026a Update 4. Other MATLAB versions have not been verified.

## Reproducing the Results

The project scripts use a local project root. Update the root path in the scripts before execution.

The principal workflow is:

1. Run `solve_cavity_fdm.m` to generate the finite-difference CFD reference.
2. Prepare the CFD-supervised dataset using the project's dataset-preparation script.
3. Initialize or load the boundary-conforming PINN checkpoint.
4. Run `train_boundary_variants.m` to compare the boundary exponents.
5. Run `compare_boundary_variant_physics.m` for the common-grid physics comparison.
6. Run `train_final_physics_m8.m` for refinement.
7. Run `final_model_evaluation.m` to reproduce the final evaluation tables and figures.

Reproduction from scratch requires the preceding initialization and dataset-preparation stages. Saved checkpoints can instead be used to evaluate already trained models when distributed separately.

## Limitations

The project does not establish a fully converged PINN solution of the Navier–Stokes equations.

Important limitations include:

- Significant fourth-order transport residuals remain, particularly near the cavity sidewalls.
- The predicted primary vortex center differs from the independent CFD reference.
- The CFD baseline uses a smoothed-lid boundary condition rather than the classical discontinuous moving lid.
- Only Reynolds number 100 was investigated in the final configuration.
- The CFD grid comparison used two resolutions rather than a formal three-grid convergence analysis.
- The spatially partitioned CFD dataset originates from one numerical flow field; the test split is not an independent flow configuration.
- Results on the reused CFD test partition should be interpreted as spatial interpolation evaluation, not generalization to new Reynolds numbers or geometries.

## Conclusion

The project demonstrates that the design of an analytical boundary extension can substantially influence the training behavior and governing-equation residuals of a streamfunction-based PINN.

The final \(m=8\) architecture provides improved CFD-assisted velocity predictions and a substantially lower fourth-order transport residual than the original \(m=2\) architecture.

The model remains a research prototype with documented physical-consistency limitations rather than a replacement for a converged CFD solver.

## Project Status

Research implementation and final comparative evaluation completed.

Further research may investigate improved physics convergence, more accurate vortex-center prediction, and extension to additional Reynolds numbers and geometries.

## License

See the `LICENSE` file for the applicable repository license.
