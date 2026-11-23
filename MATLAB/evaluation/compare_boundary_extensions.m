
%% ============================================================
% PROJECT 252 - STAGE 9Q.3
% ALTERNATIVE ANALYTICAL BOUNDARY EXTENSIONS
%
% Compare f_m(y) = y^m * (y-1)
% m = 2, 4, 6, 8
%
% Same physical smoothed-lid function g(x).
% No neural network modification or training.
%% ============================================================

clc;
clear;
close all;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

outDir = fullfile(root,'results', ...
    'streamfunction_only','boundary_diagnostic');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

nu = 0.01;
delta = 0.075;

exponents = [2 4 6 8];

%% 1. Same evaluation grid as Stage 9Q.2

N = 41;

x = linspace(0.02,0.98,N);
y = linspace(0.02,0.98,N);

[X,Y] = meshgrid(x,y);

%% 2. Calculate g'', g'''' at x grid points

xDL = dlarray(double(x),'CB');

[g2DL,g4DL] = dlfeval( ...
    @lidDerivatives,xDL,delta);

g2 = double(gather(extractdata(g2DL)));
g4 = double(gather(extractdata(g4DL)));

g2 = reshape(g2,1,[]);
g4 = reshape(g4,1,[]);

G2 = repmat(g2,N,1);
G4 = repmat(g4,N,1);

%% 3. Regional masks

sideMask = X<=0.15 | X>=0.85;

coreMask = X>=0.15 & X<=0.85 & ...
           Y>=0.15 & Y<=0.85;

upperMask = Y>=0.85;

%% 4. Allocate comparison metrics

numCases = numel(exponents);

globalRMSE = zeros(numCases,1);
sideRMSE = zeros(numCases,1);
coreRMSE = zeros(numCases,1);
upperRMSE = zeros(numCases,1);

fourthRMSE = zeros(numCases,1);
mixedRMSE = zeros(numCases,1);

maxAbsDiffusion = zeros(numCases,1);

Dfields = zeros(N,N,numCases);

%% 5. Evaluate each extension

for k = 1:numCases

    m = exponents(k);

    % f(y) = y^m*(y-1)
    F = Y.^m.*(Y-1);

    % f''(y)
    F2 = m*(m-1)*Y.^(m-2).*(Y-1) ...
        + 2*m*Y.^(m-1);

    % Fourth-x contribution
    Dxxxx = -nu.*F.*G4;

    % Mixed derivative contribution
    Dmixed = -2*nu.*F2.*G2;

    % Complete analytical base diffusion
    D = Dxxxx+Dmixed;

    Dfields(:,:,k) = D;

    globalRMSE(k) = sqrt(mean(D(:).^2));

    sideRMSE(k) = sqrt(mean(D(sideMask).^2));
    coreRMSE(k) = sqrt(mean(D(coreMask).^2));
    upperRMSE(k) = sqrt(mean(D(upperMask).^2));

    fourthRMSE(k) = sqrt(mean(Dxxxx(:).^2));
    mixedRMSE(k) = sqrt(mean(Dmixed(:).^2));

    maxAbsDiffusion(k) = max(abs(D(:)));

end

%% 6. Results table

T = table( ...
    exponents(:),globalRMSE, ...
    sideRMSE,coreRMSE,upperRMSE, ...
    fourthRMSE,mixedRMSE,maxAbsDiffusion, ...
    'VariableNames',{ ...
    'Exponent','GlobalRMSE','SideRMSE', ...
    'CoreRMSE','UpperRMSE', ...
    'FourthXRMSE','MixedRMSE','MaxAbsDiffusion'});

fprintf('\n====================================\n');
fprintf('STAGE 9Q.3 - BOUNDARY EXTENSION TEST\n');
fprintf('====================================\n');

disp(T);

%% 7. Verify m=2 against Stage 9Q.1

referenceFile = fullfile(outDir, ...
    'boundary_component_diagnostics.mat');

if isfile(referenceFile)

    S = load(referenceFile);

    assert(isequal(size(S.Dbase),[N N]));

    reconstructionError = max(abs( ...
        Dfields(:,:,1)-S.Dbase),[],'all');

    fprintf('\n===== ORIGINAL EXTENSION CHECK =====\n');
    fprintf('m=2 reconstruction error: %.8e\n', ...
        reconstructionError);

    assert(reconstructionError<1e-4, ...
        'm=2 does not reproduce original base field.');

    fprintf('PASS: Original base function verified.\n');

end

%% 8. Confirm boundary lifting identities

fprintf('\n===== BOUNDARY CONDITION CHECK =====\n');

for k = 1:numCases

    m = exponents(k);

    f0 = 0^m*(0-1);
    f1 = 1^m*(1-1);

    fp0 = m*0^(m-1)*(0-1)+0^m;
    fp1 = m*1^(m-1)*(1-1)+1^m;

    assert(abs(f0)<1e-12);
    assert(abs(f1)<1e-12);
    assert(abs(fp0)<1e-12);
    assert(abs(fp1-1)<1e-12);

    fprintf('m=%d: boundary identities PASS\n',m);

end

%% 9. Comparison plot

fig = figure('Color','w', ...
    'Position',[100 100 1300 420]);

tiledlayout(1,3);

nexttile;

plot(exponents,globalRMSE,'-o', ...
    'LineWidth',1.8);

xlabel('Vertical exponent m');
ylabel('Base diffusion RMSE');
title('Global diffusion');
grid on;

nexttile;

plot(exponents,sideRMSE,'-o', ...
    'LineWidth',1.8);

xlabel('Vertical exponent m');
ylabel('Base diffusion RMSE');
title('Sidewall diffusion');
grid on;

nexttile;

plot(exponents,upperRMSE,'-o', ...
    'LineWidth',1.8);

xlabel('Vertical exponent m');
ylabel('Base diffusion RMSE');
title('Upper boundary diffusion');
grid on;

exportgraphics(fig,fullfile(outDir, ...
    'boundary_extension_comparison.png'), ...
    'Resolution',300);

%% 10. Spatial comparison

fig2 = figure('Color','w', ...
    'Position',[100 100 1300 700]);

tiledlayout(2,2,'TileSpacing','compact');

commonLimit = max(abs(Dfields(:)));

for k = 1:numCases

    nexttile;

    contourf(X,Y,Dfields(:,:,k),30, ...
        'LineColor','none');

    clim([-commonLimit,commonLimit]);
    colorbar;

    axis equal tight;

    title(sprintf('m = %d',exponents(k)));

    xlabel('x');
    ylabel('y');

end

sgtitle('Analytical Base Diffusion: Alternative Extensions');

exportgraphics(fig2,fullfile(outDir, ...
    'boundary_extension_fields.png'), ...
    'Resolution',300);

%% 11. Save

writetable(T,fullfile(outDir, ...
    'boundary_extension_comparison.csv'));

save(fullfile(outDir, ...
    'boundary_extension_comparison.mat'), ...
    'T','Dfields','X','Y', ...
    'exponents','nu','delta');

fprintf('\nStage 9Q.3 completed.\n');

%% ============================================================
% Smoothed-lid derivatives
%% ============================================================

function [g2,g4] = lidDerivatives(x,delta)

    g = (1-exp(-(x/delta).^2)) .* ...
        (1-exp(-((1-x)/delta).^2));

    dg = dlgradient(sum(g,'all'),x, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    g2 = dlgradient(sum(dg,'all'),x, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    g3 = dlgradient(sum(g2,'all'),x, ...
        EnableHigherDerivatives=true, ...
        RetainData=true);

    g4 = dlgradient(sum(g3,'all'),x, ...
        EnableHigherDerivatives=true);

end

