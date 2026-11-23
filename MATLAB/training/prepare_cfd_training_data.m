
%% ============================================================
% PROJECT 252 - STAGE 9M.1
% PREPARE CFD-ASSISTED PINN TRAINING DATA
%
% Source:
% 129 x 129 finite-difference cavity-flow solution
%
% Creates:
% 70% training samples
% 15% validation samples
% 15% held-out test samples
%
% MATLAB R2026a
%% ============================================================

clc;
clear;
close all;

%% 1. PROJECT PATHS

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

cfdFile = fullfile(rootDir,'results','cfd', ...
    'cavity_fdm_Re100_N129.mat');

outputDir = fullfile(rootDir,'data','training');

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

assert(isfile(cfdFile), ...
    'CFD reference file not found.');

%% 2. LOAD CFD REFERENCE

S = load(cfdFile);

required = {'X','Y','u','v','psi', ...
    'Re','delta','N','converged'};

for k = 1:numel(required)

    assert(isfield(S,required{k}), ...
        'Missing CFD variable: %s',required{k});

end

assert(S.converged, ...
    'CFD reference did not converge.');

assert(S.N==129, ...
    'Expected a 129 x 129 CFD grid.');

assert(S.Re==100, ...
    'Expected Reynolds number 100.');

assert(abs(S.delta-0.075)<1e-12, ...
    'Lid smoothing parameter does not match.');

assert(isequal(size(S.X),size(S.Y), ...
    size(S.u),size(S.v),size(S.psi)), ...
    'CFD field dimensions do not match.');

assert(all(isfinite([S.u(:);S.v(:);S.psi(:)])), ...
    'CFD fields contain non-finite values.');

fprintf('\n=====================================\n');
fprintf('STAGE 9M.1 - CFD DATA PREPARATION\n');
fprintf('=====================================\n');

fprintf('Reference grid: %d x %d\n',S.N,S.N);
fprintf('Reynolds number: %d\n',S.Re);
fprintf('Lid delta: %.4f\n',S.delta);

%% 3. SELECT INTERIOR GRID POINTS

% Exclude boundary nodes because the boundary-conforming
% network already enforces wall velocities analytically.
%
% CFD fields are sampled only from interior grid nodes.

Xint = S.X(2:end-1,2:end-1);
Yint = S.Y(2:end-1,2:end-1);

Uint = S.u(2:end-1,2:end-1);
Vint = S.v(2:end-1,2:end-1);

PsiInt = S.psi(2:end-1,2:end-1);

%% 4. CONVERT FIELDS TO SAMPLE VECTORS

xAll = Xint(:)';
yAll = Yint(:)';

uAll = Uint(:)';
vAll = Vint(:)';

psiAll = PsiInt(:)';

numSamples = numel(xAll);

fprintf('\nInterior CFD samples: %d\n',numSamples);

assert(numel(uAll)==numSamples);
assert(numel(vAll)==numSamples);
assert(numel(psiAll)==numSamples);

%% 5. DATA SPLIT CONFIGURATION

rngSeed = 42;
rng(rngSeed,'twister');

trainFraction = 0.70;
valFraction = 0.15;
testFraction = 0.15;

assert(abs(trainFraction+valFraction+ ...
    testFraction-1)<1e-12);

indices = randperm(numSamples);

nTrain = floor(trainFraction*numSamples);
nVal = floor(valFraction*numSamples);

nTest = numSamples-nTrain-nVal;

trainIdx = indices(1:nTrain);

valIdx = indices(nTrain+1:nTrain+nVal);

testIdx = indices(nTrain+nVal+1:end);

%% 6. TRAINING DATA

xTrain = xAll(trainIdx);
yTrain = yAll(trainIdx);

uTrain = uAll(trainIdx);
vTrain = vAll(trainIdx);

psiTrain = psiAll(trainIdx);

%% 7. VALIDATION DATA

xVal = xAll(valIdx);
yVal = yAll(valIdx);

uVal = uAll(valIdx);
vVal = vAll(valIdx);

psiVal = psiAll(valIdx);

%% 8. HELD-OUT TEST DATA

xTest = xAll(testIdx);
yTest = yAll(testIdx);

uTest = uAll(testIdx);
vTest = vAll(testIdx);

psiTest = psiAll(testIdx);

%% 9. VERIFY SPLIT INTEGRITY

assert(numel(unique(indices))==numSamples, ...
    'Duplicate sample indices detected.');

assert(isempty(intersect(trainIdx,valIdx)));
assert(isempty(intersect(trainIdx,testIdx)));
assert(isempty(intersect(valIdx,testIdx)));

fprintf('\n===== DATASET SPLIT =====\n');

fprintf('Training samples: %d\n',nTrain);
fprintf('Validation samples: %d\n',nVal);
fprintf('Test samples: %d\n',nTest);

fprintf('Total samples: %d\n', ...
    nTrain+nVal+nTest);

fprintf('\nPASS: Dataset partitions do not overlap.\n');

%% 10. VELOCITY STATISTICS

fprintf('\n===== CFD FIELD STATISTICS =====\n');

fprintf('u range: [%.6f, %.6f]\n', ...
    min(uAll),max(uAll));

fprintf('v range: [%.6f, %.6f]\n', ...
    min(vAll),max(vAll));

fprintf('psi range: [%.6f, %.6f]\n', ...
    min(psiAll),max(psiAll));

fprintf('Training u STD: %.6f\n',std(uTrain));
fprintf('Training v STD: %.6f\n',std(vTrain));

%% 11. SAVE DATASET

outputFile = fullfile(outputDir, ...
    'cfd_supervised_dataset.mat');

Re = S.Re;
delta = S.delta;
gridN = S.N;

save(outputFile, ...
    'xTrain','yTrain','uTrain','vTrain','psiTrain', ...
    'xVal','yVal','uVal','vVal','psiVal', ...
    'xTest','yTest','uTest','vTest','psiTest', ...
    'trainIdx','valIdx','testIdx', ...
    'Re','delta','gridN','rngSeed', ...
    'trainFraction','valFraction','testFraction', ...
    'nTrain','nVal','nTest');

fprintf('\nDataset saved successfully:\n%s\n', ...
    outputFile);

%% 12. VISUALIZE DATASET PARTITIONS

fig = figure('Color','w', ...
    'Position',[100 100 1300 420]);

tiledlayout(1,3, ...
    'TileSpacing','compact');

nexttile;

scatter(xTrain,yTrain,3,uTrain,'filled');

axis equal tight;
colorbar;
xlabel('x');
ylabel('y');
title(sprintf('Training: %d samples',nTrain));

nexttile;

scatter(xVal,yVal,6,uVal,'filled');

axis equal tight;
colorbar;
xlabel('x');
ylabel('y');
title(sprintf('Validation: %d samples',nVal));

nexttile;

scatter(xTest,yTest,6,uTest,'filled');

axis equal tight;
colorbar;
xlabel('x');
ylabel('y');
title(sprintf('Test: %d samples',nTest));

sgtitle('CFD-Assisted PINN Dataset Partitions');

figureFile = fullfile(outputDir, ...
    'cfd_dataset_partitions.png');

exportgraphics(fig,figureFile, ...
    'Resolution',300);

%% 13. FINAL CONFIRMATION

fprintf('\n===== STAGE 9M.1 COMPLETED =====\n');

fprintf('Training: %d\n',nTrain);
fprintf('Validation: %d\n',nVal);
fprintf('Test: %d\n',nTest);

fprintf('Random seed: %d\n',rngSeed);

fprintf('\nAll CFD data preparation operations completed.\n');

