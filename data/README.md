

\# Data and Model Checkpoints



This repository contains MATLAB source code and

published evaluation metrics.



The following generated MAT-files are required

to reproduce the final evaluation without retraining:



\## CFD Reference



results/cfd/cavity\_fdm\_Re100\_N129.mat



\## CFD-Supervised Dataset



data/training/cfd\_supervised\_dataset.mat



\## Matched m=8 Checkpoint



results/streamfunction\_only/boundary\_variants/

matched\_training/variant\_m8\_best.mat



\## Physics-Refined m=8 Checkpoint



results/streamfunction\_only/final\_physics\_m8/

projected\_best.mat



The checkpoints were generated during the

MATLAB R2026a training experiments.



If these files are not distributed with the

repository, final\_model\_evaluation.m cannot

be executed directly.



The published results are available in

results/metrics/final\_model\_metrics.csv.



