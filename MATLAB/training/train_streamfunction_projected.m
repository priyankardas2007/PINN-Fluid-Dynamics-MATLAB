
%% ============================================================
% PROJECT 252 - STAGE 9O.3B-2
% CONFLICT-AWARE FOURTH-ORDER PHYSICS TRAINING
%
% Start: Warm-up 300
% Physics: Derived vorticity, fourth-order transport
% Optimizer: Adam with projected gradient
%
% No test-set access.
%% ============================================================

clc;
clear;
close all;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

modelFile = fullfile(root,'results', ...
    'streamfunction_only','warmup', ...
    'streamfunction_warmup_best.mat');

dataFile = fullfile(root,'data','training', ...
    'cfd_supervised_dataset.mat');

outDir = fullfile(root,'results', ...
    'streamfunction_only','projected');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% Load checkpoint and data

S = load(modelFile);
D = load(dataFile);

net = S.netCheckpoint;

assert(isa(net,'dlnetwork'));
assert(D.Re==100);
assert(abs(D.delta-0.075)<1e-12);

nu = 1/D.Re;
delta = D.delta;

%% Training settings

maxIterations = 100;
batchData = 128;
batchPhysics = 32;

learningRate = 2e-6;
clipThreshold = 5;

lambdaU = 5;
lambdaV = 3;
lambdaPsi = 2;

% Relative transport gradient strength
transportRatio = 0.15;

% Never exceed the following coefficient
maxTransportWeight = 1e-3;

validationInterval = 25;
velocityTolerance = 0.02;

rng(2045,'twister');

%% Fixed physics validation grid

gridCoords = linspace(0.025,0.975,12);

[XF,YF] = meshgrid(gridCoords,gridCoords);

physicsXY = [XF(:)';YF(:)'];

%% Baseline diagnostics

[baseU,baseV] = ...
    validateVelocity(net,D,delta);

baseCombined = hypot(baseU,baseV)/sqrt(2);

baseTransport = validateTransport( ...
    net,physicsXY,nu,delta);

velocityLimit = ...
    baseCombined*(1+velocityTolerance);

fprintf('\n===== STAGE 9O.3B-2 =====\n');

fprintf('Baseline combined RMSE: %.8f\n', ...
    baseCombined);

fprintf('Baseline transport RMSE: %.8f\n', ...
    baseTransport);

fprintf('Velocity acceptance limit: %.8f\n', ...
    velocityLimit);

%% Best model

bestNet = net;
bestIteration = 0;
bestTransport = baseTransport;
bestVelocity = baseCombined;

avgGrad = [];
avgSqGrad = [];

trainingHistory = zeros(maxIterations,7);

validationHistory = ...
    [0,baseCombined,baseTransport];

%% Training

tic;

for iteration = 1:maxIterations

    %% CFD supervised data

    idx = randi(numel(D.xTrain),1,batchData);

    Xd = dlarray([
        reshape(D.xTrain(idx),1,[]);
        reshape(D.yTrain(idx),1,[])
    ],'CB');

    Ut = dlarray( ...
        reshape(D.uTrain(idx),1,[]),'CB');

    Vt = dlarray( ...
        reshape(D.vTrain(idx),1,[]),'CB');

    Pt = dlarray( ...
        reshape(D.psiTrain(idx),1,[]),'CB');

    %% Interior physics points

    Xf = dlarray( ...
        0.01+0.98*rand(2,batchPhysics),'CB');

    %% Independent gradients

    [Ls,Lt,gS,gT] = dlfeval( ...
        @separateGradients, ...
        net,Xd,Ut,Vt,Pt,Xf,nu,delta, ...
        lambdaU,lambdaV,lambdaPsi);

    supervisedLoss = numeric(Ls);
    transportLoss = numeric(Lt);

    assert(isfinite(supervisedLoss) && ...
        isfinite(transportLoss), ...
        'Nonfinite loss.');

    %% Global norms and dot product

    ns2 = 0;
    nt2 = 0;
    dotST = 0;

    for k = 1:height(gS)

        a = double(gather( ...
            extractdata(gS.Value{k})));

        b = double(gather( ...
            extractdata(gT.Value{k})));

        assert(all(isfinite(a(:))) && ...
            all(isfinite(b(:))), ...
            'Nonfinite gradients.');

        ns2 = ns2+sum(a(:).^2);
        nt2 = nt2+sum(b(:).^2);
        dotST = dotST+sum(a(:).*b(:));

    end

    ns = sqrt(ns2);
    nt = sqrt(nt2);

    assert(ns>0 && nt>0);

    cosine = dotST/(ns*nt);

    %% Remove conflicting transport component

    projectedT = gT;

    if dotST<0

        coefficient = dotST/ns2;

        projectedT.Value = cellfun( ...
            @(gt,gs) gt-coefficient.*gs, ...
            gT.Value,gS.Value, ...
            'UniformOutput',false);

    end

    %% Norm of projected transport gradient

    np2 = 0;

    for k = 1:height(projectedT)

        q = double(gather(extractdata( ...
            projectedT.Value{k})));

        np2 = np2+sum(q(:).^2);

    end

    np = sqrt(np2);

    %% Conservative transport scaling

    lambdaT = min( ...
        maxTransportWeight, ...
        transportRatio*ns/max(np,1e-12));

    %% Compose update

    grads = gS;

    grads.Value = cellfun( ...
        @(gs,gt) gs+lambdaT.*gt, ...
        gS.Value,projectedT.Value, ...
        'UniformOutput',false);

    %% Clip combined gradient

    ng2 = 0;

    for k = 1:height(grads)

        g = double(gather(extractdata( ...
            grads.Value{k})));

        ng2 = ng2+sum(g(:).^2);

    end

    ng = sqrt(ng2);

    if ng>clipThreshold

        scale = clipThreshold/ng;

        grads.Value = cellfun( ...
            @(g) g.*scale,grads.Value, ...
            'UniformOutput',false);

    end

    %% Optimizer update

    [net,avgGrad,avgSqGrad] = adamupdate( ...
        net,grads,avgGrad,avgSqGrad, ...
        iteration,learningRate);

    trainingHistory(iteration,:) = ...
        [iteration,supervisedLoss, ...
         transportLoss,ns,nt,cosine,lambdaT];

    if iteration==1 || mod(iteration,10)==0

        fprintf(['Iter %3d | S %.4e | T %.4e | ' ...
            'cos %.3f | lambda %.3e\n'], ...
            iteration,supervisedLoss, ...
            transportLoss,cosine,lambdaT);

    end

    %% Fixed validation

    if mod(iteration,validationInterval)==0

        [valU,valV] = ...
            validateVelocity(net,D,delta);

        valCombined = ...
            hypot(valU,valV)/sqrt(2);

        valTransport = validateTransport( ...
            net,physicsXY,nu,delta);

        validationHistory(end+1,:) = ...
            [iteration,valCombined,valTransport];

        fprintf('\nVALIDATION %d\n',iteration);
        fprintf('Combined velocity: %.8f\n',valCombined);
        fprintf('Transport RMSE: %.8f\n',valTransport);

        netCheckpoint = net;

        save(fullfile(outDir,sprintf( ...
            'projected_checkpoint_%04d.mat',iteration)), ...
            'netCheckpoint','iteration', ...
            'valCombined','valTransport');

        %% Constrained checkpoint selection

        if valCombined<=velocityLimit && ...
                valTransport<bestTransport

            bestNet = net;
            bestIteration = iteration;
            bestTransport = valTransport;
            bestVelocity = valCombined;

            fprintf('New best feasible checkpoint.\n');

        end

    end
end

elapsedTime = toc;

%% Save best checkpoint

netCheckpoint = bestNet;

save(fullfile(outDir,'projected_best.mat'), ...
    'netCheckpoint','bestIteration', ...
    'bestTransport','bestVelocity', ...
    'baseTransport','baseCombined', ...
    'velocityLimit','elapsedTime');

%% Save histories

H = array2table(trainingHistory, ...
    'VariableNames', ...
    {'Iteration','SupervisedLoss','TransportLoss', ...
     'SupervisedGradNorm','TransportGradNorm', ...
     'GradientCosine','TransportWeight'});

V = array2table(validationHistory, ...
    'VariableNames', ...
    {'Iteration','CombinedRMSE','TransportRMSE'});

writetable(H,fullfile(outDir, ...
    'projected_training_history.csv'));

writetable(V,fullfile(outDir, ...
    'projected_validation_history.csv'));

%% Plot validation

fig = figure('Color','w');

tiledlayout(1,2);

nexttile;

plot(V.Iteration,V.CombinedRMSE,'-o', ...
    'LineWidth',1.5);

hold on;
yline(velocityLimit,'r--');

xlabel('Iteration');
ylabel('Combined velocity RMSE');
grid on;
title('Velocity accuracy');

nexttile;

plot(V.Iteration,V.TransportRMSE,'-o', ...
    'LineWidth',1.5);

xlabel('Iteration');
ylabel('Transport RMSE');
grid on;
title('Fourth-order physics');

exportgraphics(fig,fullfile(outDir, ...
    'projected_validation.png'), ...
    'Resolution',300);

fprintf('\n===== STAGE 9O.3B-2 COMPLETE =====\n');

fprintf('Baseline velocity RMSE: %.8f\n', ...
    baseCombined);

fprintf('Selected velocity RMSE: %.8f\n', ...
    bestVelocity);

fprintf('Baseline transport RMSE: %.8f\n', ...
    baseTransport);

fprintf('Selected transport RMSE: %.8f\n', ...
    bestTransport);

fprintf('Best iteration: %d\n',bestIteration);

fprintf('Elapsed time: %.2f seconds\n',elapsedTime);

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function [Ls,Lt,gS,gT] = separateGradients( ...
    net,Xd,Ut,Vt,Pt,Xf,nu,delta, ...
    lambdaU,lambdaV,lambdaPsi)

    [psiD,uD,vD] = fields(net,Xd,delta);

    Ls = lambdaU*mean((uD-Ut).^2,'all') ...
       + lambdaV*mean((vD-Vt).^2,'all') ...
       + lambdaPsi*mean((psiD-Pt).^2,'all');

    rT = transportResidual(net,Xf,nu,delta);

    Lt = mean(rT.^2,'all');

    gS = dlgradient(Ls,net.Learnables, ...
        RetainData=true);

    gT = dlgradient(Lt,net.Learnables);
end

function [psi,u,v] = fields(net,X,delta)

    Ntheta = forward(net,X);

    x = X(1,:);
    y = X(2,:);

    g = (1-exp(-(x/delta).^2)).* ...
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

function rT = transportResidual(net,X,nu,delta)

    [psi,u,v] = fields(net,X,delta);

    dPsi = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dpx = dlgradient(sum(dPsi(1,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dpy = dlgradient(sum(dPsi(2,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    omega = -(dpx(1,:)+dpy(2,:));

    dw = dlgradient(sum(omega,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dwx = dlgradient(sum(dw(1,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dwy = dlgradient(sum(dw(2,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    rT = u.*dw(1,:)+v.*dw(2,:) ...
        -nu.*(dwx(1,:)+dwy(2,:));
end

function [rmseU,rmseV] = ...
    validateVelocity(net,D,delta)

    n = numel(D.xVal);
    sumU = 0;
    sumV = 0;

    for first = 1:128:n

        idx = first:min(first+127,n);

        X = dlarray([ ...
            reshape(D.xVal(idx),1,[]); ...
            reshape(D.yVal(idx),1,[])],'CB');

        [~,u,v] = dlfeval(@fields,net,X,delta);

        u = double(gather(extractdata(u)));
        v = double(gather(extractdata(v)));

        tu = D.uVal(idx);
        tv = D.vVal(idx);

        sumU = sumU+sum((u(:)-tu(:)).^2);
        sumV = sumV+sum((v(:)-tv(:)).^2);
    end

    rmseU = sqrt(sumU/n);
    rmseV = sqrt(sumV/n);
end

function rmseT = validateTransport( ...
    net,XY,nu,delta)

    n = size(XY,2);
    sumT = 0;

    for first = 1:24:n

        idx = first:min(first+23,n);

        X = dlarray(XY(:,idx),'CB');

        rT = dlfeval(@transportResidual, ...
            net,X,nu,delta);

        r = double(gather(extractdata(rT)));

        sumT = sumT+sum(r(:).^2);
    end

    rmseT = sqrt(sumT/n);
end

function v = numeric(x)
    v = double(gather(extractdata(x)));
end

