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

## Reproducing the Final Results

This repository includes the MATLAB implementation, trained model checkpoints, CFD reference data, and evaluation scripts used in the final comparison.

### Requirements

- MATLAB R2026a (developed using Update 4)
- Deep Learning Toolbox
- MATLAB support for automatic differentiation and numerical computing

### Step 1: Clone the Repository

```bash
git clone https://github.com/priyankardas2007/PINN-Fluid-Dynamics-MATLAB.git
cd PINN-Fluid-Dynamics-MATLAB
```

### Step 2: Configure MATLAB

Open MATLAB and set the project root to the location where you cloned the repository:

```matlab
repoRoot = 'YOUR_LOCAL_REPOSITORY_PATH';

addpath(genpath(fullfile(repoRoot,'MATLAB')));
cd(repoRoot);
```

Replace `YOUR_LOCAL_REPOSITORY_PATH` with the actual folder path.

### Step 3: Verify the Required Files

The final evaluation uses these four files:

```text
data/training/cfd_supervised_dataset.mat

results/cfd/cavity_fdm_Re100_N129.mat

results/streamfunction_only/boundary_variants/
matched_training/variant_m8_best.mat

results/streamfunction_only/final_physics_m8/
projected_best.mat
```

All four files must be available at the specified relative paths.

### Step 4: Run the Final Evaluation

In the MATLAB Command Window:

```matlab
run(fullfile(repoRoot,'MATLAB', ...
    'evaluation','final_model_evaluation.m'));
```

This script compares the CFD-supervised m = 8 model against the physics-refined m = 8 model.

The evaluation includes:

- Validation and test velocity RMSE
- Streamfunction RMSE
- Fourth-order vorticity-transport residual
- Sidewall and interior residual errors
- Primary vortex-center location

No additional neural-network training is required.

### Step 5: Inspect the Results

The script generates its outputs under:

```text
results/streamfunction_only/final_evaluation/
```

Expected generated files include:

```text
final_model_metrics.csv
final_model_comparison.png
final_evaluation_results.mat
final_selected_pinn.mat
```

### Reference Results

The selected physics-refined m = 8 PINN achieved:

| Metric | Reported value |
|---|---:|
| Validation velocity RMSE | 0.091464 |
| Test velocity RMSE | 0.093532 |
| Test streamfunction RMSE | 0.013120 |
| Transport residual RMSE | 35.375 |
| Sidewall transport RMSE | 64.927 |
| Interior transport RMSE | 3.245 |
| Primary vortex displacement | 0.13844 |

The transport residual was evaluated on a fixed 41 × 41 interior grid.

Small numerical differences may occur depending on the MATLAB execution environment.

### Reproducibility Scope

This procedure reproduces the **final comparison using existing trained checkpoints**, without retraining.

The complete historical training sequence includes additional intermediate experiments and checkpoints that are not all included in this release.

The CFD test data consists of spatial samples from the same reference flow configuration. It should not be interpreted as an independent Reynolds-number or geometry generalization benchmark.
## Limitations

The model is a research prototype and is **not a fully converged Navier–Stokes solution**.

Significant fourth-order transport residuals remain near the sidewalls. The predicted primary vortex center also differs from the independent CFD reference.

The final experiments focus on Reynolds number 100 with a smoothed-lid boundary condition. Test samples originate from the same numerical CFD flow field and therefore evaluate spatial interpolation rather than generalization to unseen flow conditions.

## Conclusion

This project demonstrates how analytical boundary-extension design, CFD-assisted training and physics-informed optimization influence the accuracy of neural fluid-flow models.

The redesigned streamfunction PINN improves CFD velocity reconstruction and substantially reduces fourth-order governing-equation residuals compared with the original architecture, while highlighting the challenges of achieving complete physical consistency.

## Acknowledgment of AI Assistance

I developed this project using MATLAB, implementing and evaluating a Physics-Informed Neural Network (PINN) for two-dimensional fluid flow simulation. Throughout the development process, I used OpenAI's ChatGPT as a supporting tool for understanding mathematical concepts, debugging MATLAB code, exploring optimization strategies, and improving technical documentation.

I worked on the model development, executed the simulations and training experiments, analyzed the numerical results, and refined the implementation based on the observed performance. ChatGPT assisted with code suggestions, troubleshooting, and interpreting experimental outcomes.

The final implementation, experimental findings, and conclusions reflect my work in developing, testing, and evaluating the project using MATLAB.
## Author

**Priyankar Das**

Electronics and Telecommunication Engineering  
Army Institute of Technology, Pune

[GitHub Profile](https://github.com/priyankardas2007)
