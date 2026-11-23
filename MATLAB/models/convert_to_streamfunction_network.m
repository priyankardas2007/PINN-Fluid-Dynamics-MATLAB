
%% ============================================================
% PROJECT 252 - STAGE 9O.2C
% TWO-OUTPUT TO SINGLE-OUTPUT NETWORK CONVERSION
%
% Source: Hybrid 500 (Npsi, omega)
% Target: Streamfunction-only network (Npsi)
%
% Preserve all hidden weights and the first output neuron.
% Verify numerical equivalence before saving.
%% ============================================================

clc;
clear;
close all;

%% 1. Paths
rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

sourceFile = fullfile(rootDir,'results','hybrid', ...
    'hybrid_best_validation.mat');

outputDir = fullfile(rootDir,'results', ...
    'streamfunction_only');

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

assert(isfile(sourceFile), ...
    'Hybrid 500 checkpoint not found.');

%% 2. Load original network

S = load(sourceFile);
oldNet = S.netCheckpoint;

assert(isa(oldNet,'dlnetwork'));

fprintf('\n====================================\n');
fprintf('STAGE 9O.2C - NETWORK CONVERSION\n');
fprintf('====================================\n');

%% 3. Build a new one-output network

layers = oldNet.Layers;

assert(numel(layers)==10, ...
    'Unexpected number of layers.');

assert(strcmp(layers(end).Name,'output'), ...
    'Unexpected output-layer name.');

% The old network ends with 2 outputs.
% The new one ends with 1 output.

layers(end) = fullyConnectedLayer(1, ...
    'Name','output');

newNet = dlnetwork(layers);

%% 4. Transfer learned parameters

oldLearnables = oldNet.Learnables;
newLearnables = newNet.Learnables;

for k = 1:height(newLearnables)

    layerName = string(newLearnables.Layer(k));
    paramName = string(newLearnables.Parameter(k));

    oldIdx = find( ...
        string(oldLearnables.Layer)==layerName & ...
        string(oldLearnables.Parameter)==paramName);

    assert(numel(oldIdx)==1, ...
        'Could not locate parameter %s/%s', ...
        layerName,paramName);

    oldValue = oldLearnables.Value{oldIdx};

    if layerName=="output"

        if paramName=="Weights"
            newValue = oldValue(1,:);
        elseif paramName=="Bias"
            newValue = oldValue(1,:);
        else
            error('Unexpected output parameter.');
        end

    else

        % Copy all hidden layer parameters unchanged.
        newValue = oldValue;

    end

    expectedSize = size(newLearnables.Value{k});

    assert(isequal(size(newValue),expectedSize), ...
        'Parameter size mismatch: %s/%s', ...
        layerName,paramName);

    newNet.Learnables.Value{k} = newValue;

end

%% 5. Verify output dimensions

rng(2042,'twister');

X = dlarray(rand(2,128),'CB');

oldY = forward(oldNet,X);
newY = forward(newNet,X);

fprintf('\n===== OUTPUT DIMENSIONS =====\n');

fprintf('Original: %s\n',mat2str(size(oldY)));
fprintf('Converted: %s\n',mat2str(size(newY)));

assert(size(oldY,1)==2);
assert(size(newY,1)==1);

%% 6. Compare first-output predictions

oldNpsi = double(gather( ...
    extractdata(oldY(1,:))));

newNpsi = double(gather( ...
    extractdata(newY(1,:))));

maxDifference = max(abs(oldNpsi(:)-newNpsi(:)));

rmseDifference = sqrt(mean( ...
    (oldNpsi(:)-newNpsi(:)).^2));

fprintf('\n===== WEIGHT TRANSFER CHECK =====\n');

fprintf('Maximum output difference: %.8e\n', ...
    maxDifference);

fprintf('Output RMSE difference: %.8e\n', ...
    rmseDifference);

tolerance = 1e-5;

assert(maxDifference<tolerance, ...
    'Converted network does not match original.');

fprintf('\nPASS: Streamfunction output preserved.\n');

%% 7. Save converted network

netCheckpoint = newNet;

Re = 100;
delta = 0.075;

save(fullfile(outputDir, ...
    'streamfunction_only_initial.mat'), ...
    'netCheckpoint','Re','delta', ...
    'maxDifference','rmseDifference');

%% 8. Summary

fprintf('\n===== CONVERTED NETWORK =====\n');

disp(newNet);

fprintf('\nSTAGE 9O.2C COMPLETED.\n');

fprintf('Original Hybrid 500 checkpoint unchanged.\n');
fprintf('Single-output network saved successfully.\n');

