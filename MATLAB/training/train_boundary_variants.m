
%% ============================================================
% PROJECT 252 - STAGE 9R.2
% MATCHED CFD-SUPERVISED BOUNDARY VARIANT TRAINING
%
% Compare m = 2, 6, 8
% Same initial neural weights and data-batch sequence
% No fourth-order physics loss in this experiment
% No test-set access
%% ============================================================

clc;
clear;
close all;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

sourceFile = fullfile(root,'results', ...
    'streamfunction_only','warmup', ...
    'streamfunction_warmup_best.mat');

dataFile = fullfile(root,'data','training', ...
    'cfd_supervised_dataset.mat');

outDir = fullfile(root,'results', ...
    'streamfunction_only','boundary_variants', ...
    'matched_training');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

S = load(sourceFile);
D = load(dataFile);

initialNet = S.netCheckpoint;

assert(isa(initialNet,'dlnetwork'));
assert(D.Re==100);
assert(abs(D.delta-0.075)<1e-12);

delta = D.delta;
Re = D.Re;

exponents = [2 6 8];

maxIterations = 500;
batchSize = 256;
learningRate = 2e-5;
clipThreshold = 10;
validationInterval = 50;

lambdaU = 5;
lambdaV = 3;
lambdaPsi = 2;

nModels = numel(exponents);

initialRMSE = zeros(nModels,1);
bestRMSE = zeros(nModels,1);
bestIteration = zeros(nModels,1);
finalRMSE = zeros(nModels,1);

allValidation = zeros(maxIterations/validationInterval+1, ...
    nModels);

fprintf('\n====================================\n');
fprintf('STAGE 9R.2 - MATCHED MODEL TRAINING\n');
fprintf('====================================\n');

%% Loop over exponent variants

for modelIdx = 1:nModels

    m = exponents(modelIdx);

    % Identical learned initialization
    net = initialNet;

    % Reset random seed to reuse the same batch sequence
    rng(2050,'twister');

    avgGrad = [];
    avgSqGrad = [];

    history = zeros(maxIterations,6);

    validationIterations = ...
        (0:validationInterval:maxIterations)';

    validationRMSE = zeros( ...
        numel(validationIterations),1);

    %% Initial validation

    [u0,v0,psi0] = scoreModel(net,D,delta,m);

    initialCombined = hypot(u0,v0)/sqrt(2);

    initialRMSE(modelIdx) = initialCombined;

    bestValue = initialCombined;
    bestIter = 0;
    bestNet = net;

    validationRMSE(1) = initialCombined;

    fprintf('\n===== MODEL m = %d =====\n',m);
    fprintf('Initial combined RMSE: %.8f\n', ...
        initialCombined);

    tic;

    for iteration = 1:maxIterations

        %% Identically seeded CFD batches
        idx = randi(numel(D.xTrain),1,batchSize);

        X = dlarray([
            reshape(D.xTrain(idx),1,[]);
            reshape(D.yTrain(idx),1,[])
        ],'CB');

        Ut = dlarray( ...
            reshape(D.uTrain(idx),1,[]),'CB');

        Vt = dlarray( ...
            reshape(D.vTrain(idx),1,[]),'CB');

        Pt = dlarray( ...
            reshape(D.psiTrain(idx),1,[]),'CB');

        %% Calculate loss and gradients

        [total,Lu,Lv,Lpsi,grads] = dlfeval( ...
            @supervisedLoss,net,X,Ut,Vt,Pt, ...
            delta,m,lambdaU,lambdaV,lambdaPsi);

        values = double(gather(extractdata( ...
            [total,Lu,Lv,Lpsi])));

        assert(all(isfinite(values)));

        %% Gradient norm

        normSq = 0;

        for k = 1:height(grads)
            g = double(gather( ...
                extractdata(grads.Value{k})));

            assert(all(isfinite(g(:))));
            normSq = normSq+sum(g(:).^2);
        end

        gradNorm = sqrt(normSq);

        %% Clip gradient

        if gradNorm>clipThreshold
            scale = clipThreshold/gradNorm;

            grads.Value = cellfun( ...
                @(g) g.*scale,grads.Value, ...
                'UniformOutput',false);
        end

        %% Adam

        [net,avgGrad,avgSqGrad] = adamupdate( ...
            net,grads,avgGrad,avgSqGrad, ...
            iteration,learningRate);

        history(iteration,:) = ...
            [iteration,values(:)',gradNorm];

        %% Periodic validation

        if mod(iteration,validationInterval)==0

            [uRMSE,vRMSE,psiRMSE] = ...
                scoreModel(net,D,delta,m);

            combined = hypot(uRMSE,vRMSE)/sqrt(2);

            valIndex = iteration/validationInterval+1;
            validationRMSE(valIndex) = combined;

            fprintf(['m=%d | Iter %3d | ' ...
                'u=%.6f | v=%.6f | ' ...
                'psi=%.6f | Combined=%.6f\n'], ...
                m,iteration,uRMSE,vRMSE, ...
                psiRMSE,combined);

            if combined<bestValue
                bestValue = combined;
                bestIter = iteration;
                bestNet = net;
            end
        end

    end

    elapsedTime = toc;

    %% Save best checkpoint

    netCheckpoint = bestNet;

    bestRMSE(modelIdx) = bestValue;
    bestIteration(modelIdx) = bestIter;
    finalRMSE(modelIdx) = validationRMSE(end);

    save(fullfile(outDir, ...
        sprintf('variant_m%d_best.mat',m)), ...
        'netCheckpoint','m','Re','delta', ...
        'bestValue','bestIter', ...
        'initialCombined','elapsedTime');

    %% Save training and validation histories

    H = array2table(history, ...
        'VariableNames', ...
        {'Iteration','TotalLoss','ULoss', ...
        'VLoss','PsiLoss','GradientNorm'});

    % History includes gradient norm as a sixth column.
    % Correct the variable list below before exporting:
    H.Properties.VariableNames = ...
        {'Iteration','TotalLoss','ULoss', ...
         'VLoss','PsiLoss','GradientNorm'};

    V = table(validationIterations,validationRMSE, ...
        'VariableNames', ...
        {'Iteration','CombinedRMSE'});

    writetable(H,fullfile(outDir, ...
        sprintf('variant_m%d_training.csv',m)));

    writetable(V,fullfile(outDir, ...
        sprintf('variant_m%d_validation.csv',m)));

    allValidation(:,modelIdx) = validationRMSE;

    fprintf('m=%d complete | Best %.8f at iter %d\n', ...
        m,bestValue,bestIter);

end

%% Final comparison

T = table(exponents(:),initialRMSE,bestRMSE, ...
    bestIteration,finalRMSE, ...
    'VariableNames', ...
    {'Exponent','InitialRMSE','BestRMSE', ...
     'BestIteration','FinalRMSE'});

fprintf('\n===== MATCHED TRAINING RESULTS =====\n');
disp(T);

writetable(T,fullfile(outDir, ...
    'matched_training_comparison.csv'));

%% Plot validation curves

figure('Color','w');

plot(0:validationInterval:maxIterations, ...
    allValidation,'LineWidth',1.6);

grid on;

xlabel('Iteration');
ylabel('Combined validation velocity RMSE');

legend('m=2','m=6','m=8','Location','best');
title('Stage 9R.2: Matched CFD Training');

exportgraphics(gcf,fullfile(outDir, ...
    'matched_training_validation.png'), ...
    'Resolution',300);

fprintf('\nStage 9R.2 completed.\n');

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function [total,Lu,Lv,Lpsi,grads] = ...
    supervisedLoss(net,X,Ut,Vt,Pt,delta,m,wu,wv,wp)

    [psi,u,v] = fields(net,X,delta,m);

    Lu = mean((u-Ut).^2,'all');
    Lv = mean((v-Vt).^2,'all');
    Lpsi = mean((psi-Pt).^2,'all');

    total = wu*Lu+wv*Lv+wp*Lpsi;

    grads = dlgradient(total,net.Learnables);
end

function [psi,u,v] = fields(net,X,delta,m)

    Ntheta = forward(net,X);

    x = X(1,:);
    y = X(2,:);

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

    psiBase = y.^m.*(y-1).*g;

    A = 256.*x.^2.*(1-x).^2.* ...
        y.^2.*(1-y).^2;

    psi = psiBase+A.*Ntheta;

    dpsi = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    u = dpsi(2,:);
    v = -dpsi(1,:);
end

function [rmseU,rmseV,rmsePsi] = ...
    scoreModel(net,D,delta,m)

    n = numel(D.xVal);

    sU = 0;
    sV = 0;
    sPsi = 0;

    for first = 1:128:n

        idx = first:min(first+127,n);

        X = dlarray([
            reshape(D.xVal(idx),1,[]);
            reshape(D.yVal(idx),1,[])
        ],'CB');

        [psi,u,v] = dlfeval( ...
            @fields,net,X,delta,m);

        u = double(gather(extractdata(u)));
        v = double(gather(extractdata(v)));
        psi = double(gather(extractdata(psi)));

        tu = D.uVal(idx);
        tv = D.vVal(idx);
        tp = D.psiVal(idx);

        sU = sU+sum((u(:)-tu(:)).^2);
        sV = sV+sum((v(:)-tv(:)).^2);
        sPsi = sPsi+sum((psi(:)-tp(:)).^2);
    end

    rmseU = sqrt(sU/n);
    rmseV = sqrt(sV/n);
    rmsePsi = sqrt(sPsi/n);
end

