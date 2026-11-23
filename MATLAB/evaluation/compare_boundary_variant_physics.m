
%% ============================================================
% PROJECT 252 - STAGE 9R.3
% FOURTH-ORDER PHYSICS COMPARISON
%
% Models:
%   m=2 matched training
%   m=6 matched training
%   m=8 matched training
%   Sidewall 100 original formulation
%
% All evaluated on identical fixed coordinates.
% No training and no test-set access.
%% ============================================================

clc;
clear;
close all;

%% 1. Paths

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

trainDir = fullfile(root,'results', ...
    'streamfunction_only','boundary_variants', ...
    'matched_training');

sidewallFile = fullfile(root,'results', ...
    'streamfunction_only','sidewall', ...
    'projected_best.mat');

outDir = fullfile(root,'results', ...
    'streamfunction_only','boundary_variants', ...
    'physics_comparison');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% 2. Model configurations

names = [
    "Matched m=2"
    "Matched m=6"
    "Matched m=8"
    "Sidewall 100"
];

exponents = [2 6 8 2];

modelFiles = {
    fullfile(trainDir,'variant_m2_best.mat')
    fullfile(trainDir,'variant_m6_best.mat')
    fullfile(trainDir,'variant_m8_best.mat')
    sidewallFile
};

nModels = numel(modelFiles);

nets = cell(nModels,1);

for k = 1:nModels

    assert(isfile(modelFiles{k}), ...
        'Missing checkpoint: %s',modelFiles{k});

    S = load(modelFiles{k});

    nets{k} = S.netCheckpoint;

    assert(isa(nets{k},'dlnetwork'));

    if k<=3
        assert(S.m==exponents(k));
    end

end

fprintf('\n====================================\n');
fprintf('STAGE 9R.3 - PHYSICS COMPARISON\n');
fprintf('====================================\n');

fprintf('All four checkpoints loaded.\n');

%% 3. Physics settings

Re = 100;
nu = 1/Re;
delta = 0.075;

%% 4. Fixed evaluation grid

N = 41;

x = linspace(0.02,0.98,N);
y = linspace(0.02,0.98,N);

[X,Y] = meshgrid(x,y);

XY = [X(:)';Y(:)'];

nPoints = size(XY,2);

batchSize = 24;

%% 5. Regional masks

sideMask = X<=0.15 | X>=0.85;

upperMask = ~sideMask & Y>=0.85;
lowerMask = ~sideMask & Y<=0.15;

interiorMask = ...
    ~(sideMask | upperMask | lowerMask);

%% 6. Allocate results

globalRMSE = zeros(nModels,1);
sideRMSE = zeros(nModels,1);
interiorRMSE = zeros(nModels,1);

advectionRMSE = zeros(nModels,1);
diffusionRMSE = zeros(nModels,1);

maxResidual = zeros(nModels,1);
peakX = zeros(nModels,1);
peakY = zeros(nModels,1);

sideShare = zeros(nModels,1);
elapsedSeconds = zeros(nModels,1);

residualFields = zeros(N,N,nModels);

%% 7. Evaluate models

for k = 1:nModels

    net = nets{k};
    m = exponents(k);

    fprintf('\nEvaluating %s (m=%d)...\n', ...
        names(k),m);

    rFlat = zeros(1,nPoints);
    aFlat = zeros(1,nPoints);
    dFlat = zeros(1,nPoints);

    tic;

    for first = 1:batchSize:nPoints

        idx = first:min( ...
            first+batchSize-1,nPoints);

        Xin = dlarray(XY(:,idx),'CB');

        [r,a,d] = dlfeval( ...
            @transportResidual, ...
            net,Xin,nu,delta,m);

        rFlat(idx) = toNumeric(r);
        aFlat(idx) = toNumeric(a);
        dFlat(idx) = toNumeric(d);

    end

    elapsedSeconds(k) = toc;

    assert(all(isfinite(rFlat)));

    R = reshape(rFlat,N,N);
    A = reshape(aFlat,N,N);
    D = reshape(dFlat,N,N);

    residualFields(:,:,k) = R;

    globalRMSE(k) = sqrt(mean(R(:).^2));
    sideRMSE(k) = sqrt(mean(R(sideMask).^2));

    interiorRMSE(k) = ...
        sqrt(mean(R(interiorMask).^2));

    advectionRMSE(k) = sqrt(mean(A(:).^2));
    diffusionRMSE(k) = sqrt(mean(D(:).^2));

    [maxResidual(k),peakIndex] = ...
        max(abs(R(:)));

    peakX(k) = X(peakIndex);
    peakY(k) = Y(peakIndex);

    sideShare(k) = ...
        100*sum(R(sideMask).^2)/sum(R(:).^2);

    fprintf('Global transport RMSE: %.6f\n', ...
        globalRMSE(k));

    fprintf('Sidewall RMSE: %.6f\n', ...
        sideRMSE(k));

    fprintf('Interior RMSE: %.6f\n', ...
        interiorRMSE(k));

    fprintf('Sidewall error share: %.4f%%\n', ...
        sideShare(k));

    fprintf('Maximum residual: %.6f\n', ...
        maxResidual(k));

end

%% 8. Comparison table

T = table( ...
    names,exponents(:), ...
    globalRMSE,sideRMSE,interiorRMSE, ...
    advectionRMSE,diffusionRMSE, ...
    maxResidual,peakX,peakY, ...
    sideShare,elapsedSeconds, ...
    'VariableNames',{ ...
    'Model','Exponent','TransportRMSE', ...
    'SidewallRMSE','InteriorRMSE', ...
    'AdvectionRMSE','DiffusionRMSE', ...
    'MaxAbsResidual','PeakX','PeakY', ...
    'SidewallErrorShare','ElapsedSeconds'});

fprintf('\n===== COMPLETE PHYSICS COMPARISON =====\n');
disp(T);

%% 9. Relative improvement versus matched m=2

relativeReduction = ...
    100*(1-globalRMSE/globalRMSE(1));

T.RelativeReductionVsM2 = relativeReduction;

fprintf('\n===== TRANSPORT RMSE REDUCTION =====\n');

for k = 1:nModels

    fprintf('%s: %.3f%%\n', ...
        names(k),relativeReduction(k));

end

%% 10. Common-scale residual maps

fig1 = figure('Color','w', ...
    'Position',[100 100 1100 900]);

tiledlayout(2,2,'TileSpacing','compact');

commonLimit = max(abs(residualFields(:)));

for k = 1:nModels

    nexttile;

    contourf(X,Y,residualFields(:,:,k),30, ...
        'LineColor','none');

    clim([-commonLimit,commonLimit]);

    colorbar;
    axis equal tight;

    title(names(k));
    xlabel('x');
    ylabel('y');

end

sgtitle('Stage 9R.3: Full Transport Residual Fields');

exportgraphics(fig1,fullfile(outDir, ...
    'variant_transport_residual_maps.png'), ...
    'Resolution',300);

%% 11. RMSE comparison

fig2 = figure('Color','w', ...
    'Position',[100 100 1000 450]);

bar([globalRMSE,sideRMSE,interiorRMSE]);

set(gca,'XTick',1:nModels, ...
    'XTickLabel',cellstr(names));

xtickangle(20);

legend('Global','Sidewall','Interior', ...
    'Location','best');

ylabel('Transport residual RMSE');

title('Stage 9R.3: Physics Accuracy Comparison');

grid on;

exportgraphics(fig2,fullfile(outDir, ...
    'variant_physics_rmse.png'), ...
    'Resolution',300);

%% 12. Save results

writetable(T,fullfile(outDir, ...
    'variant_physics_comparison.csv'));

save(fullfile(outDir, ...
    'variant_physics_comparison.mat'), ...
    'T','X','Y','residualFields', ...
    'names','exponents','nu','delta');

fprintf('\n===== STAGE 9R.3 COMPLETED =====\n');

fprintf('All comparison results saved.\n');

%% ============================================================
% LOCAL FUNCTION: FOURTH-ORDER TRANSPORT
%% ============================================================

function [r,a,d] = ...
    transportResidual(net,X,nu,delta,m)

    x = X(1,:);
    y = X(2,:);

    Ntheta = forward(net,X);

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

    psiBase = y.^m.*(y-1).*g;

    envelope = 256.*x.^2.*(1-x).^2 ...
        .*y.^2.*(1-y).^2;

    psi = psiBase+envelope.*Ntheta;

    %% Velocity

    dp = dlgradient(sum(psi,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    u = dp(2,:);
    v = -dp(1,:);

    %% Vorticity

    dpx = dlgradient(sum(dp(1,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dpy = dlgradient(sum(dp(2,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    omega = -(dpx(1,:)+dpy(2,:));

    %% Third derivatives

    dw = dlgradient(sum(omega,'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    %% Fourth derivatives

    dwx = dlgradient(sum(dw(1,:),'all'),X, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    dwy = dlgradient(sum(dw(2,:),'all'),X, ...
        EnableHigherDerivatives=true);

    %% Transport equation

    a = u.*dw(1,:)+v.*dw(2,:);

    d = nu.*(dwx(1,:)+dwy(2,:));

    r = a-d;

end

%% ============================================================
% LOCAL FUNCTION: NUMERIC CONVERSION
%% ============================================================

function value = toNumeric(x)

    value = double(gather(extractdata(x)));
    value = reshape(value,1,[]);

end

