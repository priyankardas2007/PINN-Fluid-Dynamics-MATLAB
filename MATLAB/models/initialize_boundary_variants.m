
%% ============================================================
% PROJECT 252 - STAGE 9R.1
% INITIALIZE AND VERIFY BOUNDARY EXTENSION VARIANTS
%
% Models: m = 2, 6, 8
%
% Same trained neural-network weights.
% Different analytical boundary-lifting functions.
%
% No training or test-set evaluation.
%% ============================================================

clc;
clear;
close all;

%% 1. Paths

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

modelFile = fullfile(root,'results', ...
    'streamfunction_only','warmup', ...
    'streamfunction_warmup_best.mat');

dataFile = fullfile(root,'data','training', ...
    'cfd_supervised_dataset.mat');

outDir = fullfile(root,'results', ...
    'streamfunction_only','boundary_variants');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

assert(isfile(modelFile), ...
    'Warm-up checkpoint not found.');

assert(isfile(dataFile), ...
    'CFD training dataset not found.');

%% 2. Load common network

S = load(modelFile);
D = load(dataFile);

baseNet = S.netCheckpoint;

assert(isa(baseNet,'dlnetwork'));

assert(D.Re==100);
assert(abs(D.delta-0.075)<1e-12);

Re = 100;
delta = 0.075;

exponents = [2 6 8];
nModels = numel(exponents);

%% 3. Verify single-output architecture

testX = dlarray(rand(2,8),'CB');
testY = forward(baseNet,testX);

assert(size(testY,1)==1, ...
    'Expected a single-output network.');

fprintf('\n====================================\n');
fprintf('STAGE 9R.1 - BOUNDARY VARIANTS\n');
fprintf('====================================\n');

fprintf('Common network: Warm-up 300\n');
fprintf('Reynolds number: %d\n',Re);
fprintf('Lid smoothing delta: %.4f\n',delta);

%% 4. Boundary test coordinates

nWall = 101;
s = linspace(0,1,nWall);

bottomXY = [s;zeros(1,nWall)];
topXY = [s;ones(1,nWall)];

leftXY = [zeros(1,nWall);s];
rightXY = [ones(1,nWall);s];

wallPoints = {
    bottomXY
    topXY
    leftXY
    rightXY
};

wallNames = [
    "Bottom"
    "Top"
    "Left"
    "Right"
];

%% 5. Allocate results

boundaryMaxError = zeros(nModels,1);

validationU = zeros(nModels,1);
validationV = zeros(nModels,1);
validationPsi = zeros(nModels,1);
validationCombined = zeros(nModels,1);

%% 6. Evaluate each architecture

for k = 1:nModels

    m = exponents(k);

    % Same neural-network weights for all variants.
    netCheckpoint = baseNet;

    fprintf('\n===== MODEL m = %d =====\n',m);

    %% Boundary verification

    maxError = 0;

    for wall = 1:4

        XY = wallPoints{wall};

        X = dlarray(XY,'CB');

        [~,u,v] = dlfeval( ...
            @physicalFields, ...
            netCheckpoint,X,delta,m);

        u = toNumeric(u);
        v = toNumeric(v);

        % Expected velocity boundary conditions
        targetU = zeros(1,nWall);
        targetV = zeros(1,nWall);

        if wall==2
            targetU = lidVelocity(s,delta);
        end

        errorU = max(abs(u-targetU));
        errorV = max(abs(v-targetV));

        wallError = max(errorU,errorV);
        maxError = max(maxError,wallError);

        fprintf('%s wall error: %.8e\n', ...
            wallNames(wall),wallError);

    end

    boundaryMaxError(k) = maxError;

    assert(maxError<1e-4, ...
        'Boundary condition verification failed.');

    fprintf('PASS: Exact velocity boundaries verified.\n');

    %% CFD validation

    [uRMSE,vRMSE,psiRMSE] = ...
        validateVariant(netCheckpoint,D,delta,m);

    combined = hypot(uRMSE,vRMSE)/sqrt(2);

    validationU(k) = uRMSE;
    validationV(k) = vRMSE;
    validationPsi(k) = psiRMSE;
    validationCombined(k) = combined;

    fprintf('\nValidation u RMSE: %.8f\n',uRMSE);
    fprintf('Validation v RMSE: %.8f\n',vRMSE);
    fprintf('Validation psi RMSE: %.8f\n',psiRMSE);
    fprintf('Combined velocity RMSE: %.8f\n',combined);

    %% Save variant configuration

    save(fullfile(outDir, ...
        sprintf('boundary_variant_m%d_initial.mat',m)), ...
        'netCheckpoint','m','Re','delta', ...
        'uRMSE','vRMSE','psiRMSE', ...
        'combined','maxError');

end

%% 7. Comparison table

T = table( ...
    exponents(:),boundaryMaxError, ...
    validationU,validationV, ...
    validationPsi,validationCombined, ...
    'VariableNames',{ ...
    'Exponent','BoundaryMaxError', ...
    'uRMSE','vRMSE','psiRMSE', ...
    'CombinedVelocityRMSE'});

fprintf('\n===== STAGE 9R.1 COMPARISON =====\n');

disp(T);

%% 8. Visualization

fig = figure('Color','w', ...
    'Position',[100 100 1100 450]);

tiledlayout(1,2);

nexttile;

bar(exponents,validationCombined);

xlabel('Boundary exponent m');
ylabel('Combined validation velocity RMSE');
title('Initial CFD Agreement');
grid on;

nexttile;

bar(exponents,validationPsi);

xlabel('Boundary exponent m');
ylabel('Streamfunction validation RMSE');
title('Initial Streamfunction Agreement');
grid on;

exportgraphics(fig,fullfile(outDir, ...
    'boundary_variant_initial_comparison.png'), ...
    'Resolution',300);

%% 9. Save

writetable(T,fullfile(outDir, ...
    'boundary_variant_initial_metrics.csv'));

save(fullfile(outDir, ...
    'boundary_variant_initial_metrics.mat'), ...
    'T','exponents','Re','delta');

fprintf('\n====================================\n');
fprintf('STAGE 9R.1 COMPLETED\n');
fprintf('====================================\n');

fprintf('Three variant configurations saved.\n');
fprintf('Original warm-up network unchanged.\n');

%% ============================================================
% LOCAL FUNCTION 1: LID VELOCITY
%% ============================================================

function g = lidVelocity(x,delta)

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

end

%% ============================================================
% LOCAL FUNCTION 2: PHYSICAL FIELDS
%% ============================================================

function [psi,u,v] = ...
    physicalFields(net,X,delta,m)

    Ntheta = forward(net,X);

    x = X(1,:);
    y = X(2,:);

    g = lidVelocity(x,delta);

    % New boundary extension
    psiBase = y.^m.*(y-1).*g;

    % Original correction envelope
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
    validateVariant(net,D,delta,m)

    n = numel(D.xVal);

    batchSize = 128;

    sumU = 0;
    sumV = 0;
    sumPsi = 0;

    for first = 1:batchSize:n

        idx = first:min(first+batchSize-1,n);

        X = dlarray([
            reshape(D.xVal(idx),1,[]);
            reshape(D.yVal(idx),1,[])
        ],'CB');

        [psi,u,v] = dlfeval( ...
            @physicalFields,net,X,delta,m);

        u = toNumeric(u);
        v = toNumeric(v);
        psi = toNumeric(psi);

        targetU = D.uVal(idx);
        targetV = D.vVal(idx);
        targetPsi = D.psiVal(idx);

        du = u(:)-targetU(:);
        dv = v(:)-targetV(:);
        dp = psi(:)-targetPsi(:);

        sumU = sumU+sum(du.^2);
        sumV = sumV+sum(dv.^2);
        sumPsi = sumPsi+sum(dp.^2);

    end

    rmseU = sqrt(sumU/n);
    rmseV = sqrt(sumV/n);
    rmsePsi = sqrt(sumPsi/n);

end

%% ============================================================
% LOCAL FUNCTION 4: NUMERIC CONVERSION
%% ============================================================

function a = toNumeric(x)

    a = double(gather(extractdata(x)));
    a = reshape(a,1,[]);

end

