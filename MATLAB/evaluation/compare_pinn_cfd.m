
%% ============================================================
% PROJECT 252 - STAGE 9L
% QUANTITATIVE PINN VS CFD VALIDATION
%
% Models:
%   Recovery 800
%   Adam 200
%   PCGrad 200
%
% Reference:
%   FDM, Re = 100, N = 129, delta = 0.075
%% ============================================================

clc;
clear;
close all;

%% 1. Project paths
rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

velocityDir = fullfile(rootDir,'results','velocity');
cfdDir = fullfile(rootDir,'results','cfd');
errorDir = fullfile(rootDir,'results','error');

if ~exist(errorDir,'dir')
    mkdir(errorDir);
end

%% 2. Load saved fields
pinnFile = fullfile(velocityDir, ...
    'stage9j_flow_fields.mat');

cfdFile = fullfile(cfdDir, ...
    'cavity_fdm_Re100_N129.mat');

assert(isfile(pinnFile),'Stage 9J file missing.');
assert(isfile(cfdFile),'CFD reference file missing.');

P = load(pinnFile);
C = load(cfdFile);

assert(C.converged, ...
    'CFD reference has not converged.');

assert(C.Re == P.Re && abs(C.delta-P.delta)<1e-12, ...
    'PINN and CFD physical settings do not match.');

%% 3. Common coordinates

X = P.Xgrid;
Y = P.Ygrid;

% Interpolate CFD fields to PINN evaluation grid
uRef = interp2(C.X,C.Y,C.u,X,Y,'linear');
vRef = interp2(C.X,C.Y,C.v,X,Y,'linear');
psiRef = interp2(C.X,C.Y,C.psi,X,Y,'linear');

assert(all(isfinite([uRef(:);vRef(:);psiRef(:)])), ...
    'Invalid interpolated CFD values.');

modelNames = string(P.modelNames);
numModels = numel(modelNames);

%% 4. Region masks

% Full cavity, including walls
fullMask = true(size(X));

% Interior core, excluding near-wall region
coreMask = X>=0.05 & X<=0.95 & ...
           Y>=0.05 & Y<=0.95;

%% 5. Allocate metrics

uRMSE = zeros(numModels,1);
vRMSE = zeros(numModels,1);
psiRMSE = zeros(numModels,1);

uMAE = zeros(numModels,1);
vMAE = zeros(numModels,1);

uCoreRMSE = zeros(numModels,1);
vCoreRMSE = zeros(numModels,1);

maxVelocityError = zeros(numModels,1);

%% 6. Compute model errors

fprintf('\n=======================================\n');
fprintf('STAGE 9L - PINN VS CFD VALIDATION\n');
fprintf('=======================================\n');

for m = 1:numModels

    uPred = P.U(:,:,m);
    vPred = P.V(:,:,m);
    psiPred = P.Psi(:,:,m);

    du = uPred-uRef;
    dv = vPred-vRef;
    dpsi = psiPred-psiRef;

    uRMSE(m) = sqrt(mean(du(fullMask).^2));
    vRMSE(m) = sqrt(mean(dv(fullMask).^2));
    psiRMSE(m) = sqrt(mean(dpsi(fullMask).^2));

    uMAE(m) = mean(abs(du(fullMask)));
    vMAE(m) = mean(abs(dv(fullMask)));

    uCoreRMSE(m) = sqrt(mean(du(coreMask).^2));
    vCoreRMSE(m) = sqrt(mean(dv(coreMask).^2));

    speedError = hypot(du,dv);
    maxVelocityError(m) = max(speedError(:));

    fprintf('\nModel: %s\n',modelNames(m));
    fprintf('u RMSE: %.8e\n',uRMSE(m));
    fprintf('v RMSE: %.8e\n',vRMSE(m));
    fprintf('psi RMSE: %.8e\n',psiRMSE(m));
    fprintf('Core u RMSE: %.8e\n',uCoreRMSE(m));
    fprintf('Core v RMSE: %.8e\n',vCoreRMSE(m));

end

%% 7. Save comparison table

T = table(modelNames,uRMSE,vRMSE,psiRMSE, ...
    uMAE,vMAE,uCoreRMSE,vCoreRMSE, ...
    maxVelocityError, ...
    'VariableNames',{'Model','uRMSE','vRMSE', ...
    'psiRMSE','uMAE','vMAE','uCoreRMSE', ...
    'vCoreRMSE','MaxVelocityError'});

fprintf('\n===== PINN VS CFD METRICS =====\n');
disp(T);

writetable(T,fullfile(errorDir, ...
    'pinn_vs_cfd_metrics.csv'));

%% 8. Velocity error maps

allUError = P.U-repmat(uRef,1,1,numModels);
allVError = P.V-repmat(vRef,1,1,numModels);

uLimit = max(abs(allUError(:)));
vLimit = max(abs(allVError(:)));

uLimit = max(uLimit,eps);
vLimit = max(vLimit,eps);

fig1 = figure('Color','w', ...
    'Position',[100 100 1200 1000]);

tiledlayout(numModels,2, ...
    'TileSpacing','compact');

for m = 1:numModels

    nexttile;

    contourf(X,Y,allUError(:,:,m),30, ...
        'LineColor','none');

    clim([-uLimit,uLimit]);
    colorbar;
    axis equal tight;

    title(modelNames(m)+" - u error");
    xlabel('x');
    ylabel('y');

    nexttile;

    contourf(X,Y,allVError(:,:,m),30, ...
        'LineColor','none');

    clim([-vLimit,vLimit]);
    colorbar;
    axis equal tight;

    title(modelNames(m)+" - v error");
    xlabel('x');
    ylabel('y');

end

sgtitle('PINN Minus CFD: Velocity Error Fields');

exportgraphics(fig1,fullfile(errorDir, ...
    'pinn_vs_cfd_velocity_errors.png'), ...
    'Resolution',300);

%% 9. Centerline comparison

[~,mid] = min(abs(X(1,:)-0.5));

x = X(1,:);
y = Y(:,1);

fig2 = figure('Color','w', ...
    'Position',[100 100 1200 500]);

tiledlayout(1,2);

nexttile;
plot(uRef(:,mid),y,'k-', ...
    'LineWidth',2,'DisplayName','CFD 129');
hold on;

for m = 1:numModels

    plot(P.U(:,mid,m),y, ...
        'LineWidth',1.5, ...
        'DisplayName',char(modelNames(m)));

end

xlabel('u');
ylabel('y');
title('Vertical centerline: u(0.5,y)');
legend('Location','best');
grid on;

nexttile;

plot(x,vRef(mid,:),'k-', ...
    'LineWidth',2,'DisplayName','CFD 129');
hold on;

for m = 1:numModels

    plot(x,P.V(mid,:,m), ...
        'LineWidth',1.5, ...
        'DisplayName',char(modelNames(m)));

end

xlabel('x');
ylabel('v');
title('Horizontal centerline: v(x,0.5)');
legend('Location','best');
grid on;

sgtitle('PINN vs CFD Centerline Velocities');

exportgraphics(fig2,fullfile(errorDir, ...
    'pinn_vs_cfd_centerlines.png'), ...
    'Resolution',300);

%% 10. Metric comparison plots

fig3 = figure('Color','w');

metricMatrix = [uRMSE,vRMSE,psiRMSE];

bar(metricMatrix);

set(gca,'XTick',1:numModels, ...
    'XTickLabel',cellstr(modelNames));

xtickangle(15);

ylabel('RMSE');
legend('u','v','psi','Location','best');
title('PINN vs CFD: Quantitative Errors');

grid on;

exportgraphics(fig3,fullfile(errorDir, ...
    'pinn_vs_cfd_rmse.png'), ...
    'Resolution',300);

%% 11. Save detailed numerical data

save(fullfile(errorDir, ...
    'pinn_vs_cfd_validation.mat'), ...
    'T','X','Y','uRef','vRef','psiRef', ...
    'allUError','allVError','coreMask');

%% 12. Identify lowest velocity RMSE model

combinedRMSE = sqrt((uRMSE.^2+vRMSE.^2)/2);

[bestValue,bestIndex] = min(combinedRMSE);

fprintf('\n===== BEST MODEL BY VELOCITY RMSE =====\n');

fprintf('Model: %s\n',modelNames(bestIndex));

fprintf('Combined velocity RMSE: %.8e\n', ...
    bestValue);

fprintf('\nStage 9L completed.\n');

