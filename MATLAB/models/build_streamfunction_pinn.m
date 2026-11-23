
%% ============================================================
% PROJECT 252 - STAGE 7A
% BUILD STREAMFUNCTION-VORTICITY PINN
%
% Problem: 2D Lid-Driven Cavity Flow
% Reynolds Number: 100
%
% Inputs:  x, y
% Outputs: Streamfunction psi, Vorticity omega
%
% Architecture:
% 2 -> 64 -> 64 -> 64 -> 64 -> 2
%% ============================================================

clc;
clear;
close all;

%% STEP 1: Project configuration

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

outputDir = fullfile(rootDir, 'MATLAB');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

Re = 100;
nu = 1/Re;

rng(42);

fprintf('\n===== STREAMFUNCTION PINN CONFIGURATION =====\n');
fprintf('Reynolds number: %d\n', Re);
fprintf('Kinematic viscosity: %.5f\n', nu);

%% STEP 2: Neural network architecture

layers = [

    featureInputLayer(2, ...
        Normalization="none", ...
        Name="input")

    fullyConnectedLayer(64, Name="fc1")
    tanhLayer(Name="tanh1")

    fullyConnectedLayer(64, Name="fc2")
    tanhLayer(Name="tanh2")

    fullyConnectedLayer(64, Name="fc3")
    tanhLayer(Name="tanh3")

    fullyConnectedLayer(64, Name="fc4")
    tanhLayer(Name="tanh4")

    fullyConnectedLayer(2, Name="output")
];

%% STEP 3: Construct network

net = dlnetwork(layers);

fprintf('\n===== NETWORK CREATED =====\n');

disp(net);

%% STEP 4: Test network predictions

numTestPoints = 100;

xTest = rand(1, numTestPoints);
yTest = rand(1, numTestPoints);

XTest = dlarray([xTest; yTest], 'CB');

YTest = forward(net, XTest);

fprintf('\n===== NETWORK OUTPUT TEST =====\n');

fprintf('Input dimensions: %d x %d\n', size(XTest));
fprintf('Output dimensions: %d x %d\n', size(YTest));

assert(size(YTest,1) == 2, ...
    'Network must output exactly 2 quantities.');

assert(size(YTest,2) == numTestPoints, ...
    'Incorrect number of output samples.');

Yvalues = gather(extractdata(YTest));

assert(all(isfinite(Yvalues(:))), ...
    'Network produced non-finite outputs.');

fprintf('Output 1: Streamfunction psi\n');
fprintf('Output 2: Vorticity omega\n');

disp('Network output test successful.');

%% STEP 5: Save network

saveFile = fullfile(outputDir, ...
    'streamfunction_pinn_initial.mat');

save(saveFile, 'net', 'Re', 'nu');

fprintf('\n===== NETWORK SAVED =====\n');
fprintf('Location: %s\n', saveFile);

disp('Streamfunction PINN successfully initialized.');

