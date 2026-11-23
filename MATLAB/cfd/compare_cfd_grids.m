
%% PROJECT 252 - STAGE 9K.2
% Quantitative CFD grid-refinement comparison

clc;
clear;
close all;

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cfdDir = fullfile(rootDir,'results','cfd');

%% Load two CFD solutions

coarse = load(fullfile(cfdDir, ...
    'cavity_fdm_Re100_N65.mat'));

fine = load(fullfile(cfdDir, ...
    'cavity_fdm_Re100_N129.mat'));

assert(coarse.converged && fine.converged, ...
    'Both CFD solutions must be converged.');

assert(coarse.N == 65 && fine.N == 129, ...
    'Unexpected grid sizes.');

assert(coarse.Re == fine.Re && ...
    coarse.delta == fine.delta, ...
    'Physical parameters do not match.');

%% Extract fine-grid values at coarse-grid coordinates

idx = 1:2:129;

uFine = fine.u(idx,idx);
vFine = fine.v(idx,idx);
psiFine = fine.psi(idx,idx);

assert(max(abs(coarse.x-fine.x(idx))) < 1e-12);
assert(max(abs(coarse.y-fine.y(idx))) < 1e-12);

%% Calculate differences

du = coarse.u-uFine;
dv = coarse.v-vFine;
dpsi = coarse.psi-psiFine;

rmseU = sqrt(mean(du(:).^2));
rmseV = sqrt(mean(dv(:).^2));
rmsePsi = sqrt(mean(dpsi(:).^2));

maxU = max(abs(du(:)));
maxV = max(abs(dv(:)));
maxPsi = max(abs(dpsi(:)));

fprintf('\n===== CFD GRID REFINEMENT =====\n');

fprintf('u RMSE: %.8e\n',rmseU);
fprintf('v RMSE: %.8e\n',rmseV);
fprintf('psi RMSE: %.8e\n',rmsePsi);

fprintf('\nMaximum absolute differences:\n');
fprintf('u: %.8e\n',maxU);
fprintf('v: %.8e\n',maxV);
fprintf('psi: %.8e\n',maxPsi);

%% Vortex center differences

dx = fine.vortexX-coarse.vortexX;
dy = fine.vortexY-coarse.vortexY;

displacement = hypot(dx,dy);

fprintf('\nVortex center displacement: %.8e\n', ...
    displacement);

fprintf('Coarse minimum psi: %.8f\n', ...
    coarse.psiMin);

fprintf('Fine minimum psi: %.8f\n', ...
    fine.psiMin);

%% Plot centerline comparison

[~,midC] = min(abs(coarse.x-0.5));
[~,midF] = min(abs(fine.x-0.5));

fig = figure('Color','w', ...
    'Position',[100 100 1100 450]);

tiledlayout(1,2);

nexttile;
plot(coarse.u(:,midC),coarse.y,'o-', ...
    'LineWidth',1.2,'MarkerSize',3);
hold on;

plot(fine.u(:,midF),fine.y,'-', ...
    'LineWidth',1.8);

xlabel('u');
ylabel('y');
title('Vertical centerline');
legend('65 x 65','129 x 129','Location','best');
grid on;

nexttile;
plot(coarse.x,coarse.v(midC,:),'o-', ...
    'LineWidth',1.2,'MarkerSize',3);
hold on;

plot(fine.x,fine.v(midF,:),'-', ...
    'LineWidth',1.8);

xlabel('x');
ylabel('v');
title('Horizontal centerline');
legend('65 x 65','129 x 129','Location','best');
grid on;

sgtitle('CFD Grid Refinement Comparison');

exportgraphics(fig,fullfile(cfdDir, ...
    'cfd_grid_refinement_centerlines.png'), ...
    'Resolution',300);

%% Plot spatial velocity difference

fig2 = figure('Color','w');

tiledlayout(1,2);

nexttile;
contourf(coarse.X,coarse.Y,du,25, ...
    'LineColor','none');
colorbar;
axis equal tight;
title('u: coarse - fine');

nexttile;
contourf(coarse.X,coarse.Y,dv,25, ...
    'LineColor','none');
colorbar;
axis equal tight;
title('v: coarse - fine');

sgtitle('CFD Grid Refinement: Velocity Differences');

exportgraphics(fig2,fullfile(cfdDir, ...
    'cfd_grid_refinement_errors.png'), ...
    'Resolution',300);

%% Save comparison metrics

metrics = table(rmseU,rmseV,rmsePsi, ...
    maxU,maxV,maxPsi,displacement);

writetable(metrics,fullfile(cfdDir, ...
    'cfd_grid_refinement_metrics.csv'));

fprintf('\nGrid comparison completed.\n');

