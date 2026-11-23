
%% ============================================================
% PROJECT 252 - STAGE 9O.3A
% SINGLE-OUTPUT STREAMFUNCTION PINN: CFD WARM-UP
%
% Source network: Stage 9O.2C
% Re = 100, delta = 0.075
%
% Warm-up objectives:
% - Horizontal velocity CFD agreement
% - Vertical velocity CFD agreement
% - Streamfunction CFD agreement
%
% No fourth-order PDE residual during this stage.
% No test-set access.
%% ============================================================

clc;
clear;
close all;

%% 1. Project paths

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

modelFile = fullfile(rootDir,'results', ...
    'streamfunction_only', ...
    'streamfunction_only_initial.mat');

dataFile = fullfile(rootDir,'data','training', ...
    'cfd_supervised_dataset.mat');

outputDir = fullfile(rootDir,'results', ...
    'streamfunction_only','warmup');

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

assert(isfile(modelFile),'Converted model missing.');
assert(isfile(dataFile),'CFD dataset missing.');

%% 2. Load network and data

S = load(modelFile);
net = S.netCheckpoint;

D = load(dataFile);

assert(isa(net,'dlnetwork'));
assert(D.Re == 100);
assert(abs(D.delta-0.075)<1e-12);

delta = D.delta;

fprintf('\n====================================\n');
fprintf('STAGE 9O.3A - STREAMFUNCTION WARM-UP\n');
fprintf('====================================\n');

fprintf('Single-output network loaded.\n');

%% 3. Training configuration

maxIterations = 300;
batchSize = 256;

learningRate = 5e-6;
clipThreshold = 10;

lambdaU = 5;
lambdaV = 3;
lambdaPsi = 2;

validationInterval = 50;

rng(2043,'twister');

avgGrad = [];
avgSqGrad = [];

history = zeros(maxIterations,6);

%% 4. Baseline validation

[baseU,baseV,basePsi] = ...
    validateModel(net,D,delta);

baseCombined = hypot(baseU,baseV)/sqrt(2);

bestNet = net;
bestIteration = 0;
bestCombined = baseCombined;

validationHistory = ...
    [0,baseU,baseV,basePsi,baseCombined];

fprintf('\n===== BASELINE =====\n');

fprintf('u RMSE: %.8f\n',baseU);
fprintf('v RMSE: %.8f\n',baseV);
fprintf('psi RMSE: %.8f\n',basePsi);
fprintf('Combined RMSE: %.8f\n',baseCombined);

%% 5. Training loop

fprintf('\n===== WARM-UP STARTED =====\n');

tic;

for iteration = 1:maxIterations

    %% Random CFD training batch

    idx = randi(numel(D.xTrain),1,batchSize);

    X = dlarray([
        reshape(D.xTrain(idx),1,[]);
        reshape(D.yTrain(idx),1,[])
    ],'CB');

    Utarget = dlarray( ...
        reshape(D.uTrain(idx),1,[]),'CB');

    Vtarget = dlarray( ...
        reshape(D.vTrain(idx),1,[]),'CB');

    PsiTarget = dlarray( ...
        reshape(D.psiTrain(idx),1,[]),'CB');

    %% Compute supervised losses

    [total,lossU,lossV,lossPsi,grads] = ...
        dlfeval(@warmupLoss, ...
        net,X,Utarget,Vtarget,PsiTarget, ...
        delta,lambdaU,lambdaV,lambdaPsi);

    values = [
        scalarValue(total)
        scalarValue(lossU)
        scalarValue(lossV)
        scalarValue(lossPsi)
    ];

    assert(all(isfinite(values)), ...
        'Non-finite loss at iteration %d',iteration);

    %% Global gradient norm

    normSquared = 0;

    for k = 1:height(grads)

        g = grads.Value{k};

        if isempty(g)
            continue;
        end

        gData = double(gather(extractdata(g)));

        assert(all(isfinite(gData(:))), ...
            'Non-finite gradient detected.');

        normSquared = normSquared + ...
            sum(gData(:).^2);

    end

    gradNorm = sqrt(normSquared);

    assert(isfinite(gradNorm) && gradNorm>0, ...
        'Invalid gradient norm.');

    %% Global norm clipping

    if gradNorm>clipThreshold

        scale = clipThreshold/gradNorm;

        grads.Value = cellfun( ...
            @(g) g.*scale, ...
            grads.Value, ...
            'UniformOutput',false);

    end

    %% Adam update

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grads,avgGrad,avgSqGrad, ...
        iteration,learningRate);

    history(iteration,:) = ...
        [iteration,values(:)',gradNorm];

    %% Training log

    if iteration==1 || mod(iteration,25)==0

        fprintf(['Iter %3d | Total %.5e | ' ...
            'U %.5e | V %.5e | ' ...
            'Psi %.5e | G %.3e\n'], ...
            iteration,values(1),values(2), ...
            values(3),values(4),gradNorm);

    end

    %% Validation and checkpoint

    if mod(iteration,validationInterval)==0

        [valU,valV,valPsi] = ...
            validateModel(net,D,delta);

        valCombined = hypot(valU,valV)/sqrt(2);

        validationHistory(end+1,:) = ...
            [iteration,valU,valV,valPsi,valCombined];

        fprintf('\n===== VALIDATION %d =====\n', ...
            iteration);

        fprintf('u RMSE: %.8f\n',valU);
        fprintf('v RMSE: %.8f\n',valV);
        fprintf('psi RMSE: %.8f\n',valPsi);

        fprintf('Combined RMSE: %.8f\n', ...
            valCombined);

        netCheckpoint = net;

        save(fullfile(outputDir, ...
            sprintf('warmup_checkpoint_%04d.mat', ...
            iteration)), ...
            'netCheckpoint','iteration', ...
            'valU','valV','valPsi','valCombined', ...
            'delta');

        if valCombined < bestCombined

            bestCombined = valCombined;
            bestIteration = iteration;
            bestNet = net;

            fprintf('New best validation checkpoint.\n');

        end

    end

end

elapsedTime = toc;

%% 6. Save best model

netCheckpoint = bestNet;

save(fullfile(outputDir, ...
    'streamfunction_warmup_best.mat'), ...
    'netCheckpoint','bestIteration', ...
    'bestCombined','baseCombined','delta', ...
    'lambdaU','lambdaV','lambdaPsi');

%% 7. Export training history

H = array2table(history, ...
    'VariableNames', ...
    {'Iteration','TotalLoss','ULoss', ...
     'VLoss','PsiLoss','GradientNorm'});

% History has 6 columns:
% iteration, total, u, v, psi, gradient norm.

writetable(H,fullfile(outputDir, ...
    'warmup_training_history.csv'));

V = array2table(validationHistory, ...
    'VariableNames', ...
    {'Iteration','uRMSE','vRMSE', ...
     'psiRMSE','CombinedRMSE'});

writetable(V,fullfile(outputDir, ...
    'warmup_validation_history.csv'));

%% 8. Validation graph

fig = figure('Color','w', ...
    'Position',[100 100 1000 450]);

tiledlayout(1,2);

nexttile;

plot(validationHistory(:,1), ...
    validationHistory(:,2),'-o', ...
    'LineWidth',1.5);

hold on;

plot(validationHistory(:,1), ...
    validationHistory(:,3),'-s', ...
    'LineWidth',1.5);

xlabel('Iteration');
ylabel('RMSE');
title('CFD Velocity Validation');

legend('u RMSE','v RMSE','Location','best');
grid on;

nexttile;

plot(validationHistory(:,1), ...
    validationHistory(:,5),'-o', ...
    'LineWidth',1.5);

yline(baseCombined,'--','Initial baseline');

xlabel('Iteration');
ylabel('Combined RMSE');
title('Warm-Up Validation Progress');
grid on;

exportgraphics(fig,fullfile(outputDir, ...
    'warmup_validation.png'), ...
    'Resolution',300);

%% 9. Summary

fprintf('\n====================================\n');
fprintf('STAGE 9O.3A COMPLETED\n');
fprintf('====================================\n');

fprintf('Iterations: %d\n',maxIterations);

fprintf('Elapsed time: %.2f seconds\n', ...
    elapsedTime);

fprintf('Initial combined RMSE: %.8f\n', ...
    baseCombined);

fprintf('Best combined RMSE: %.8f\n', ...
    bestCombined);

fprintf('Best iteration: %d\n', ...
    bestIteration);

fprintf('\nOriginal converted checkpoint unchanged.\n');

%% ============================================================
% LOCAL FUNCTION 1: WARM-UP LOSS
%% ============================================================

function [total,lossU,lossV,lossPsi,grads] = ...
    warmupLoss(net,X,Ut,Vt,Pt,delta, ...
    lambdaU,lambdaV,lambdaPsi)

    [psi,u,v] = streamfunctionFields( ...
        net,X,delta);

    lossU = mean((u-Ut).^2,'all');

    lossV = mean((v-Vt).^2,'all');

    lossPsi = mean((psi-Pt).^2,'all');

    total = lambdaU*lossU ...
        +lambdaV*lossV ...
        +lambdaPsi*lossPsi;

    grads = dlgradient(total,net.Learnables);

end

%% ============================================================
% LOCAL FUNCTION 2: STREAMFUNCTION FIELDS
%% ============================================================

function [psi,u,v] = ...
    streamfunctionFields(net,X,delta)

    Y = forward(net,X);

    % True single-output network
    Ntheta = Y(1,:);

    x = X(1,:);
    y = X(2,:);

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

    psiBase = y.^2.*(y-1).*g;

    A = 256.*x.^2.*(1-x).^2.* ...
        y.^2.*(1-y).^2;

    psi = psiBase+A.*Ntheta;

    dPsi = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    u = dPsi(2,:);
    v = -dPsi(1,:);

end

%% ============================================================
% LOCAL FUNCTION 3: VALIDATION
%% ============================================================

function [rmseU,rmseV,rmsePsi] = ...
    validateModel(net,D,delta)

    n = numel(D.xVal);

    batchSize = 128;

    sumU = 0;
    sumV = 0;
    sumPsi = 0;

    for first = 1:batchSize:n

        last = min(first+batchSize-1,n);

        idx = first:last;

        X = dlarray([
            reshape(D.xVal(idx),1,[]);
            reshape(D.yVal(idx),1,[])
        ],'CB');

        [psi,u,v] = dlfeval( ...
            @streamfunctionFields,net,X,delta);

        u = double(gather(extractdata(u)));
        v = double(gather(extractdata(v)));

        psi = double(gather(extractdata(psi)));

        tu = D.uVal(idx);
        tv = D.vVal(idx);
        tp = D.psiVal(idx);

        du = u(:)-tu(:);
        dv = v(:)-tv(:);
        dp = psi(:)-tp(:);

        sumU = sumU+sum(du.^2);
        sumV = sumV+sum(dv.^2);
        sumPsi = sumPsi+sum(dp.^2);

    end

    rmseU = sqrt(sumU/n);
    rmseV = sqrt(sumV/n);
    rmsePsi = sqrt(sumPsi/n);

end

%% ============================================================
% LOCAL FUNCTION 4: SCALAR EXTRACTION
%% ============================================================

function value = scalarValue(x)

    value = double(gather(extractdata(x)));

end

