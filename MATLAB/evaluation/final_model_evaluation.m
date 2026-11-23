
%% ============================================================
% PROJECT 252 - FINAL CONSOLIDATED MODEL EVALUATION
%
% Models:
% 1. Matched m=8
% 2. Physics-refined m=8
%
% Reference:
% Finite-difference CFD, Re=100, delta=0.075
%
% Evaluates:
% - CFD validation and test errors
% - Fourth-order transport residual
% - Sidewall and interior physics
% - Primary vortex location
%
% No training.
%% ============================================================

clc;
clear;
close all;

%% 1. Project paths

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

matchedFile = fullfile(root,'results', ...
    'streamfunction_only','boundary_variants', ...
    'matched_training','variant_m8_best.mat');

refinedFile = fullfile(root,'results', ...
    'streamfunction_only','final_physics_m8', ...
    'projected_best.mat');

dataFile = fullfile(root,'data','training', ...
    'cfd_supervised_dataset.mat');

cfdFile = fullfile(root,'results','cfd', ...
    'cavity_fdm_Re100_N129.mat');

outDir = fullfile(root,'results', ...
    'streamfunction_only','final_evaluation');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

assert(isfile(matchedFile));
assert(isfile(refinedFile));
assert(isfile(dataFile));
assert(isfile(cfdFile));

%% 2. Load models and reference data

M = load(matchedFile);
F = load(refinedFile);
D = load(dataFile);
C = load(cfdFile);

nets = {
    M.netCheckpoint
    F.netCheckpoint
};

modelNames = [
    "Matched m=8"
    "Refined m=8"
];

assert(all(cellfun(@(n) isa(n,'dlnetwork'),nets)));

assert(M.m==8);
assert(D.Re==100);
assert(abs(D.delta-0.075)<1e-12);

m = 8;
Re = 100;
nu = 1/Re;
delta = D.delta;

nModels = numel(nets);

fprintf('\n=======================================\n');
fprintf('PROJECT 252 - FINAL MODEL EVALUATION\n');
fprintf('=======================================\n');

fprintf('Models: Matched m=8 and Refined m=8\n');
fprintf('Re = %d, delta = %.4f\n',Re,delta);

%% 3. Fixed physics evaluation grid

N = 41;

xPhysics = linspace(0.02,0.98,N);
yPhysics = linspace(0.02,0.98,N);

[XP,YP] = meshgrid(xPhysics,yPhysics);

physicsXY = [XP(:)';YP(:)'];

sideMask = XP<=0.15 | XP>=0.85;

interiorMask = ~sideMask & ...
    YP>0.15 & YP<0.85;

%% 4. CFD grid for vortex comparison

xCFD = C.x(:)';
yCFD = C.y(:)';

[XC,YC] = meshgrid(xCFD,yCFD);

assert(isequal(size(C.psi),size(XC)), ...
    'CFD streamfunction grid dimensions differ.');

[psiCFDMin,cfdIndex] = min(C.psi(:));

vortexCFDX = XC(cfdIndex);
vortexCFDY = YC(cfdIndex);

fprintf('\n===== CFD REFERENCE =====\n');

fprintf('CFD vortex: (%.6f, %.6f)\n', ...
    vortexCFDX,vortexCFDY);

fprintf('CFD minimum psi: %.8f\n',psiCFDMin);

%% 5. Allocate final metrics

validationU = zeros(nModels,1);
validationV = zeros(nModels,1);
validationPsi = zeros(nModels,1);
validationCombined = zeros(nModels,1);

testU = zeros(nModels,1);
testV = zeros(nModels,1);
testPsi = zeros(nModels,1);
testCombined = zeros(nModels,1);

physicsGlobal = zeros(nModels,1);
physicsSide = zeros(nModels,1);
physicsInterior = zeros(nModels,1);

maxTransport = zeros(nModels,1);
sideErrorShare = zeros(nModels,1);

vortexX = zeros(nModels,1);
vortexY = zeros(nModels,1);
vortexPsiMin = zeros(nModels,1);
vortexDistance = zeros(nModels,1);

residualFields = zeros(N,N,nModels);

%% 6. Evaluate each model

for k = 1:nModels

    net = nets{k};

    fprintf('\n====================================\n');
    fprintf('MODEL: %s\n',modelNames(k));
    fprintf('====================================\n');

    %% Validation CFD agreement

    [validationU(k),validationV(k), ...
        validationPsi(k)] = ...
        evaluateCFDSplit(net,D,'Val',delta,m);

    validationCombined(k) = hypot( ...
        validationU(k),validationV(k))/sqrt(2);

    %% Test CFD agreement

    [testU(k),testV(k),testPsi(k)] = ...
        evaluateCFDSplit(net,D,'Test',delta,m);

    testCombined(k) = ...
        hypot(testU(k),testV(k))/sqrt(2);

    %% Physics residual on common grid

    rFlat = zeros(1,numel(XP));

    batchSize = 24;

    for first = 1:batchSize:numel(XP)

        idx = first:min( ...
            first+batchSize-1,numel(XP));

        X = dlarray(physicsXY(:,idx),'CB');

        r = dlfeval(@transportResidual, ...
            net,X,nu,delta,m);

        rFlat(idx) = toNumeric(r);

    end

    assert(all(isfinite(rFlat)));

    R = reshape(rFlat,N,N);

    residualFields(:,:,k) = R;

    physicsGlobal(k) = sqrt(mean(R(:).^2));

    physicsSide(k) = ...
        sqrt(mean(R(sideMask).^2));

    physicsInterior(k) = ...
        sqrt(mean(R(interiorMask).^2));

    maxTransport(k) = max(abs(R(:)));

    sideErrorShare(k) = ...
        100*sum(R(sideMask).^2)/sum(R(:).^2);

    %% Primary vortex from full CFD coordinate grid

    psiFlat = zeros(1,numel(XC));

    allXY = [XC(:)';YC(:)'];

    for first = 1:256:numel(XC)

        idx = first:min(first+255,numel(XC));

        X = dlarray(allXY(:,idx),'CB');

        p = streamfunctionOnly( ...
            net,X,delta,m);

        psiFlat(idx) = toNumeric(p);

    end

    psiGrid = reshape(psiFlat,size(XC));

    [vortexPsiMin(k),idxMin] = min(psiGrid(:));

    vortexX(k) = XC(idxMin);
    vortexY(k) = YC(idxMin);

    vortexDistance(k) = hypot( ...
        vortexX(k)-vortexCFDX, ...
        vortexY(k)-vortexCFDY);

    %% Model summary

    fprintf('Validation combined RMSE: %.8f\n', ...
        validationCombined(k));

    fprintf('Test combined RMSE: %.8f\n', ...
        testCombined(k));

    fprintf('Test streamfunction RMSE: %.8f\n', ...
        testPsi(k));

    fprintf('Transport RMSE: %.8f\n', ...
        physicsGlobal(k));

    fprintf('Sidewall transport RMSE: %.8f\n', ...
        physicsSide(k));

    fprintf('Interior transport RMSE: %.8f\n', ...
        physicsInterior(k));

    fprintf('Maximum |transport residual|: %.8f\n', ...
        maxTransport(k));

    fprintf('Vortex center: (%.5f, %.5f)\n', ...
        vortexX(k),vortexY(k));

    fprintf('Vortex center displacement: %.8f\n', ...
        vortexDistance(k));

end

%% 7. Final comparison table

T = table( ...
    modelNames, ...
    validationU,validationV,validationPsi, ...
    validationCombined, ...
    testU,testV,testPsi,testCombined, ...
    physicsGlobal,physicsSide,physicsInterior, ...
    maxTransport,sideErrorShare, ...
    vortexX,vortexY,vortexPsiMin,vortexDistance, ...
    'VariableNames',{ ...
    'Model', ...
    'ValURMSE','ValVRMSE','ValPsiRMSE', ...
    'ValCombinedRMSE', ...
    'TestURMSE','TestVRMSE','TestPsiRMSE', ...
    'TestCombinedRMSE', ...
    'TransportRMSE','SidewallRMSE','InteriorRMSE', ...
    'MaxTransportResidual','SidewallSSEPercent', ...
    'VortexX','VortexY','PsiMinimum', ...
    'VortexDisplacement'});

fprintf('\n====================================\n');
fprintf('FINAL COMPARISON TABLE\n');
fprintf('====================================\n');

disp(T);

%% 8. Relative improvements

velocityImprovement = 100* ...
    (1-validationCombined(2)/validationCombined(1));

physicsImprovement = 100* ...
    (1-physicsGlobal(2)/physicsGlobal(1));

fprintf('\n===== REFINEMENT IMPROVEMENT =====\n');

fprintf('Validation velocity improvement: %.4f%%\n', ...
    velocityImprovement);

fprintf('Physics residual improvement: %.4f%%\n', ...
    physicsImprovement);

%% 9. Final model selection

% Selection uses validation and physics data only.
% The reserved test metrics do not determine selection.

velocityLimit = ...
    1.02*validationCombined(1);

if validationCombined(2)<=velocityLimit && ...
        physicsGlobal(2)<physicsGlobal(1)

    selectedModel = "Refined m=8";
    selectedIndex = 2;

else

    selectedModel = "Matched m=8";
    selectedIndex = 1;

end

fprintf('\n===== FINAL MODEL SELECTION =====\n');

fprintf('Selected: %s\n',selectedModel);

fprintf('Validation velocity RMSE: %.8f\n', ...
    validationCombined(selectedIndex));

fprintf('Transport RMSE: %.8f\n', ...
    physicsGlobal(selectedIndex));

%% 10. Save final selected checkpoint

netCheckpoint = nets{selectedIndex};

save(fullfile(outDir, ...
    'final_selected_pinn.mat'), ...
    'netCheckpoint','m','Re','delta', ...
    'selectedModel','T', ...
    'velocityImprovement','physicsImprovement');

%% 11. Save metrics

writetable(T,fullfile(outDir, ...
    'final_model_metrics.csv'));

save(fullfile(outDir, ...
    'final_evaluation_results.mat'), ...
    'T','XP','YP','residualFields', ...
    'psiCFDMin','vortexCFDX','vortexCFDY', ...
    'selectedModel','selectedIndex');

%% 12. Final figures

fig = figure('Color','w', ...
    'Position',[100 100 1200 450]);

tiledlayout(1,2);

nexttile;

bar(validationCombined);

set(gca,'XTick',1:nModels, ...
    'XTickLabel',cellstr(modelNames));

ylabel('Combined validation velocity RMSE');
title('CFD Accuracy');
grid on;

nexttile;

bar(physicsGlobal);

set(gca,'XTick',1:nModels, ...
    'XTickLabel',cellstr(modelNames));

ylabel('Transport residual RMSE');
title('Fourth-Order Physics');
grid on;

exportgraphics(fig,fullfile(outDir, ...
    'final_model_comparison.png'), ...
    'Resolution',300);

fprintf('\n====================================\n');
fprintf('FINAL EVALUATION COMPLETED\n');
fprintf('====================================\n');

fprintf('Results saved in:\n%s\n',outDir);

%% ============================================================
% LOCAL FUNCTION 1: DATASET EVALUATION
%% ============================================================

function [uRMSE,vRMSE,psiRMSE] = ...
    evaluateCFDSplit(net,D,split,delta,m)

    switch split
        case 'Val'
            xx = D.xVal;
            yy = D.yVal;
            ut = D.uVal;
            vt = D.vVal;
            pt = D.psiVal;

        case 'Test'
            xx = D.xTest;
            yy = D.yTest;
            ut = D.uTest;
            vt = D.vTest;
            pt = D.psiTest;

        otherwise
            error('Unknown dataset split.');
    end

    n = numel(xx);

    sumU = 0;
    sumV = 0;
    sumP = 0;

    for first = 1:128:n

        idx = first:min(first+127,n);

        X = dlarray([
            reshape(xx(idx),1,[]);
            reshape(yy(idx),1,[])
        ],'CB');

        [psi,u,v] = dlfeval( ...
            @physicalFields,net,X,delta,m);

        u = toNumeric(u);
        v = toNumeric(v);
        psi = toNumeric(psi);

        du = u(:)-reshape(ut(idx),[],1);
        dv = v(:)-reshape(vt(idx),[],1);
        dp = psi(:)-reshape(pt(idx),[],1);

        sumU = sumU+sum(du.^2);
        sumV = sumV+sum(dv.^2);
        sumP = sumP+sum(dp.^2);

    end

    uRMSE = sqrt(sumU/n);
    vRMSE = sqrt(sumV/n);
    psiRMSE = sqrt(sumP/n);

end

%% ============================================================
% LOCAL FUNCTION 2: STREAMFUNCTION
%% ============================================================

function psi = streamfunctionOnly(net,X,delta,m)

    x = X(1,:);
    y = X(2,:);

    Ntheta = forward(net,X);

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

    psiBase = y.^m.*(y-1).*g;

    A = 256.*x.^2.*(1-x).^2.* ...
        y.^2.*(1-y).^2;

    psi = psiBase+A.*Ntheta;

end

%% ============================================================
% LOCAL FUNCTION 3: PHYSICAL FIELDS
%% ============================================================

function [psi,u,v] = physicalFields(net,X,delta,m)

    psi = streamfunctionOnly(net,X,delta,m);

    dp = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    u = dp(2,:);
    v = -dp(1,:);

end

%% ============================================================
% LOCAL FUNCTION 4: FOURTH-ORDER TRANSPORT
%% ============================================================

function r = transportResidual(net,X,nu,delta,m)

    [psi,u,v] = physicalFields(net,X,delta,m);

    dp = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dpx = dlgradient(sum(dp(1,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dpy = dlgradient(sum(dp(2,:),'all'),X, ...
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
        EnableHigherDerivatives=true);

    advection = u.*dw(1,:)+v.*dw(2,:);

    diffusion = nu.*(dwx(1,:)+dwy(2,:));

    r = advection-diffusion;

end

%% ============================================================
% LOCAL FUNCTION 5: NUMERIC CONVERSION
%% ============================================================

function a = toNumeric(x)

    a = double(gather(extractdata(x)));
    a = reshape(a,1,[]);

end

