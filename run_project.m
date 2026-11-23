
%% ============================================================
% PROJECT 252 - PINN FLUID DYNAMICS
% MATLAB and Simulink Challenge Projects
%
% Main entry point for final model evaluation
%
% Author: Priyankar Das
%
% MATLAB: R2026a
% Required Toolbox: Deep Learning Toolbox
%
% Run:
%   run_project
%
% No neural network training is performed.
%% ============================================================

clc;
close all;

fprintf('\n==========================================\n');
fprintf('PINN FLUID DYNAMICS - PROJECT 252\n');
fprintf('==========================================\n');

fprintf('2D Lid-Driven Cavity Flow\n');
fprintf('Reynolds Number: 100\n');
fprintf('Selected Model: Streamfunction PINN (m=8)\n');

%% 1. Locate repository automatically

repoRoot = fileparts(mfilename('fullpath'));

fprintf('\nRepository:\n%s\n',repoRoot);

%% 2. Configure MATLAB paths

matlabDir = fullfile(repoRoot,'MATLAB');

if ~isfolder(matlabDir)
    error('MATLAB source directory not found.');
end

addpath(genpath(matlabDir));

fprintf('\nMATLAB paths configured.\n');

%% 3. Locate required evaluation inputs

requiredFiles = {
    fullfile(repoRoot,'data','training', ...
        'cfd_supervised_dataset.mat')

    fullfile(repoRoot,'results','cfd', ...
        'cavity_fdm_Re100_N129.mat')

    fullfile(repoRoot,'results','streamfunction_only', ...
        'boundary_variants','matched_training', ...
        'variant_m8_best.mat')

    fullfile(repoRoot,'results','streamfunction_only', ...
        'final_physics_m8','projected_best.mat')
};

for k = 1:numel(requiredFiles)

    if ~isfile(requiredFiles{k})
        error('Required file missing:\n%s',requiredFiles{k});
    end

end

fprintf('All required evaluation inputs found.\n');

%% 4. Locate final evaluation script

evaluationScript = fullfile( ...
    repoRoot,'MATLAB','evaluation', ...
    'final_model_evaluation.m');

if ~isfile(evaluationScript)
    error('Final evaluation script not found.');
end

%% 5. Execute final evaluation

fprintf('\n==========================================\n');
fprintf('RUNNING FINAL MODEL EVALUATION\n');
fprintf('==========================================\n');

fprintf('\nComparing:\n');
fprintf('1. CFD-supervised PINN (m=8)\n');
fprintf('2. Physics-refined PINN (m=8)\n\n');

% The final evaluation script loads the existing
% checkpoints and performs inference and analysis.
% It does not retrain the neural networks.

run(evaluationScript);
