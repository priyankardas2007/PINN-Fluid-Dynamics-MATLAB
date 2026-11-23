
%% ============================================================
% PROJECT 252 - STAGE 9K
% INDEPENDENT FINITE-DIFFERENCE CFD SOLVER
%
% 2D steady incompressible lid-driven cavity
% Streamfunction-vorticity formulation
%
% Re = 100
% Smoothed lid, delta = 0.075
%
% Numerical method:
% - Central differences for Poisson equation
% - First-order upwind convection
% - Implicit vorticity diffusion
% - Pseudo-time marching to steady state
%
% MATLAB R2026a
%% ============================================================

clc;
clear;
close all;

%% 1. PROJECT PATHS

rootDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));

outputDir = fullfile(rootDir,'results','cfd');

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

%% 2. PHYSICAL PARAMETERS

Re = 100;
nu = 1/Re;
delta = 0.075;

%% 3. NUMERICAL PARAMETERS

N = 129;

maxIterations = 20000;
minimumIterations = 1000;

steadyTolerance = 2e-4;

printInterval = 500;

h = 1/(N-1);

% Fixed time step, conservative for expected speeds
dt = 0.35*h/1.5;

%% 4. CREATE SPATIAL GRID

x = linspace(0,1,N);
y = linspace(0,1,N);

[X,Y] = meshgrid(x,y);

%% 5. SMOOTHED LID VELOCITY

lidVelocity = ...
    (1-exp(-(x/delta).^2)) .* ...
    (1-exp(-((1-x)/delta).^2));

%% 6. INITIALIZE FLOW FIELDS

psi = zeros(N,N);
omega = zeros(N,N);

u = zeros(N,N);
v = zeros(N,N);

u(end,:) = lidVelocity;

%% 7. BUILD SPARSE LAPLACIAN

n = N-2;

e = ones(n,1);

T = spdiags([e -2*e e],-1:1,n,n);

I = speye(n);

L = (kron(I,T)+kron(T,I))/h^2;

% Poisson equation: -L*psi = omega
poissonSolver = decomposition(-L,'chol');

% Implicit diffusion
diffusionMatrix = speye(n*n)-dt*nu*L;

diffusionSolver = decomposition( ...
    diffusionMatrix,'chol');
%% Matrix verification

fprintf('\n===== MATRIX VERIFICATION =====\n');

fprintf('Laplacian dimensions: %d x %d\n',size(L));

fprintf('Diffusion dimensions: %d x %d\n', ...
    size(diffusionMatrix));

assert(isequal(size(L),size(diffusionMatrix)), ...
    'Matrix dimensions do not match.');

fprintf('PASS: Matrix dimensions verified.\n');
fprintf('\n====================================\n');
fprintf('STAGE 9K - FINITE DIFFERENCE CFD\n');
fprintf('====================================\n');

fprintf('Reynolds number: %d\n',Re);
fprintf('Viscosity: %.5f\n',nu);
fprintf('Lid delta: %.4f\n',delta);
fprintf('Grid: %d x %d\n',N,N);
fprintf('Grid spacing: %.8f\n',h);
fprintf('Pseudo-time step: %.8f\n',dt);

%% 8. ITERATION HISTORY

residualHistory = zeros(maxIterations,1);
speedHistory = zeros(maxIterations,1);

converged = false;

simulationTimer = tic;

fprintf('\n===== CFD ITERATIONS STARTED =====\n');

%% 9. PSEUDO-TIME ITERATION

for iteration = 1:maxIterations

    %% 9A. Solve streamfunction Poisson equation

    rhsPsi = omega(2:end-1,2:end-1);

    psiInterior = poissonSolver\rhsPsi(:);

    psi(2:end-1,2:end-1) = ...
        reshape(psiInterior,n,n);

    %% 9B. Wall vorticity (Thom approximation)

    omega(1,2:end-1) = ...
        -2*psi(2,2:end-1)/h^2;

    omega(end,2:end-1) = ...
        -2*psi(end-1,2:end-1)/h^2 ...
        -2*lidVelocity(2:end-1)/h;

    omega(2:end-1,1) = ...
        -2*psi(2:end-1,2)/h^2;

    omega(2:end-1,end) = ...
        -2*psi(2:end-1,end-1)/h^2;

    %% 9C. Interior velocity

    u(2:end-1,2:end-1) = ...
        (psi(3:end,2:end-1) ...
        -psi(1:end-2,2:end-1))/(2*h);

    v(2:end-1,2:end-1) = ...
        -(psi(2:end-1,3:end) ...
        -psi(2:end-1,1:end-2))/(2*h);

    % Boundary velocities
    u(1,:) = 0;
    u(end,:) = lidVelocity;
    u(2:end-1,1) = 0;
    u(2:end-1,end) = 0;

    v(1,:) = 0;
    v(end,:) = 0;
    v(:,1) = 0;
    v(:,end) = 0;

    %% 9D. Evaluate interior vorticity gradients

    w = omega(2:end-1,2:end-1);

    wLeft = omega(2:end-1,1:end-2);
    wRight = omega(2:end-1,3:end);

    wDown = omega(1:end-2,2:end-1);
    wUp = omega(3:end,2:end-1);

    ui = u(2:end-1,2:end-1);
    vi = v(2:end-1,2:end-1);

    % First-order upwind x derivative
    wx = zeros(n,n);

    positiveU = ui>=0;

    wx(positiveU) = ...
        (w(positiveU)-wLeft(positiveU))/h;

    wx(~positiveU) = ...
        (wRight(~positiveU)-w(~positiveU))/h;

    % First-order upwind y derivative
    wy = zeros(n,n);

    positiveV = vi>=0;

    wy(positiveV) = ...
        (w(positiveV)-wDown(positiveV))/h;

    wy(~positiveV) = ...
        (wUp(~positiveV)-w(~positiveV))/h;

    convection = ui.*wx+vi.*wy;

    %% 9E. Boundary contribution to diffusion

    boundaryTerm = zeros(n,n);

    boundaryTerm(1,:) = ...
        boundaryTerm(1,:) + ...
        omega(1,2:end-1)/h^2;

    boundaryTerm(end,:) = ...
        boundaryTerm(end,:) + ...
        omega(end,2:end-1)/h^2;

    boundaryTerm(:,1) = ...
        boundaryTerm(:,1) + ...
        omega(2:end-1,1)/h^2;

    boundaryTerm(:,end) = ...
        boundaryTerm(:,end) + ...
        omega(2:end-1,end)/h^2;

    %% 9F. Implicit diffusion / explicit convection

    rhsOmega = w ...
        -dt*convection ...
        +dt*nu*boundaryTerm;

    newOmegaVector = ...
        diffusionSolver\rhsOmega(:);

    newOmega = reshape(newOmegaVector,n,n);

    %% 9G. Steady-state change residual

    difference = (newOmega-w)/dt;

    steadyResidual = max(abs(difference(:)));

    assert(all(isfinite(newOmega(:))), ...
        'Non-finite vorticity encountered.');

    %% 9H. Update vorticity

    omega(2:end-1,2:end-1) = newOmega;

    %% 9I. Record diagnostics

    residualHistory(iteration) = steadyResidual;

    maxSpeed = max(hypot(u(:),v(:)));

    speedHistory(iteration) = maxSpeed;

    if maxSpeed*dt/h > 0.8
        warning('High advective CFL number detected.');
    end

    %% 9J. Print iteration progress

    if mod(iteration,printInterval)==0

        fprintf([ ...
            'Iteration %6d | ' ...
            'Steady residual %.6e | ' ...
            'Max speed %.5f\n'], ...
            iteration,steadyResidual,maxSpeed);

    end

    %% 9K. Convergence test

    if iteration >= minimumIterations && ...
            steadyResidual < steadyTolerance

        converged = true;

        fprintf('\nConvergence criterion reached.\n');

        break;

    end

end

elapsedTime = toc(simulationTimer);

actualIterations = iteration;

residualHistory = ...
    residualHistory(1:actualIterations);

speedHistory = ...
    speedHistory(1:actualIterations);

%% 10. RECOMPUTE FINAL STREAMFUNCTION

rhsPsi = omega(2:end-1,2:end-1);

psiInterior = poissonSolver\rhsPsi(:);

psi(2:end-1,2:end-1) = ...
    reshape(psiInterior,n,n);

%% 11. RECOMPUTE FINAL WALL VORTICITY

omega(1,2:end-1) = ...
    -2*psi(2,2:end-1)/h^2;

omega(end,2:end-1) = ...
    -2*psi(end-1,2:end-1)/h^2 ...
    -2*lidVelocity(2:end-1)/h;

omega(2:end-1,1) = ...
    -2*psi(2:end-1,2)/h^2;

omega(2:end-1,end) = ...
    -2*psi(2:end-1,end-1)/h^2;

%% 12. RECOMPUTE FINAL VELOCITY

u(2:end-1,2:end-1) = ...
    (psi(3:end,2:end-1) ...
    -psi(1:end-2,2:end-1))/(2*h);

v(2:end-1,2:end-1) = ...
    -(psi(2:end-1,3:end) ...
    -psi(2:end-1,1:end-2))/(2*h);

u(1,:) = 0;
u(end,:) = lidVelocity;
u(2:end-1,1) = 0;
u(2:end-1,end) = 0;

v(1,:) = 0;
v(end,:) = 0;
v(:,1) = 0;
v(:,end) = 0;

%% 13. FINAL DIAGNOSTICS

speed = hypot(u,v);

maxSpeed = max(speed(:));

psiInterior = psi(2:end-1,2:end-1);

[psiMin,indexMin] = min(psiInterior(:));

[row,col] = ind2sub(size(psiInterior),indexMin);

vortexX = x(col+1);
vortexY = y(row+1);

% Finite difference divergence in the interior
ux = (u(2:end-1,3:end) ...
    -u(2:end-1,1:end-2))/(2*h);

vy = (v(3:end,2:end-1) ...
    -v(1:end-2,2:end-1))/(2*h);

divergence = ux+vy;

divergenceRMSE = sqrt(mean(divergence(:).^2));

%% 14. PRINT SUMMARY

fprintf('\n===== CFD SOLVER SUMMARY =====\n');

fprintf('Converged: %d\n',converged);

fprintf('Iterations: %d\n',actualIterations);

fprintf('Elapsed time: %.2f seconds\n',elapsedTime);

fprintf('Final steady residual: %.8e\n', ...
    residualHistory(end));

fprintf('Maximum speed: %.6f\n',maxSpeed);

fprintf('Minimum streamfunction: %.8f\n',psiMin);

fprintf('Candidate vortex center: (%.5f, %.5f)\n', ...
    vortexX,vortexY);

fprintf('Divergence RMSE: %.8e\n',divergenceRMSE);

if ~converged
    warning(['Steady-state tolerance was not reached. ' ...
        'Do not treat this field as a converged CFD reference.']);
end

%% 15. SAVE NUMERICAL SOLUTION

saveFile = fullfile(outputDir, ...
    sprintf('cavity_fdm_Re%d_N%d.mat',Re,N));

save(saveFile, ...
    'x','y','X','Y','psi','omega','u','v', ...
    'Re','nu','delta','N','h','dt', ...
    'actualIterations','elapsedTime', ...
    'converged','steadyTolerance', ...
    'residualHistory','speedHistory', ...
    'maxSpeed','psiMin','vortexX','vortexY', ...
    'divergenceRMSE');

fprintf('\nSolution saved:\n%s\n',saveFile);

%% 16. VELOCITY AND VORTICITY PLOTS

fig1 = figure('Color','w', ...
    'Position',[100 100 1500 450]);

tiledlayout(1,3);

nexttile;
contourf(X,Y,u,30,'LineColor','none');
colorbar;
axis equal tight;
title('Horizontal velocity u');
xlabel('x');
ylabel('y');

nexttile;
contourf(X,Y,v,30,'LineColor','none');
colorbar;
axis equal tight;
title('Vertical velocity v');
xlabel('x');
ylabel('y');

nexttile;
contourf(X,Y,omega,30,'LineColor','none');
colorbar;
axis equal tight;
title('Vorticity');
xlabel('x');
ylabel('y');

sgtitle('Finite Difference Cavity Flow');

exportgraphics(fig1,fullfile(outputDir, ...
    sprintf('cfd_fields_N%d.png',N)), ...
    'Resolution',300);

%% 17. STREAMFUNCTION CONTOURS

fig2 = figure('Color','w');

contour(X,Y,psi,30,'LineWidth',1.1);

axis equal tight;

xlabel('x');
ylabel('y');

title('CFD Streamfunction Contours');

grid on;

exportgraphics(fig2,fullfile(outputDir, ...
    sprintf('cfd_streamlines_N%d.png',N)), ...
    'Resolution',300);

%% 18. CENTERLINE VELOCITIES

[~,mid] = min(abs(x-0.5));

fig3 = figure('Color','w', ...
    'Position',[100 100 1100 450]);

tiledlayout(1,2);

nexttile;
plot(u(:,mid),y,'LineWidth',1.5);
grid on;
xlabel('u');
ylabel('y');
title('Vertical centerline u(0.5,y)');

nexttile;
plot(x,v(mid,:),'LineWidth',1.5);
grid on;
xlabel('x');
ylabel('v');
title('Horizontal centerline v(x,0.5)');

exportgraphics(fig3,fullfile(outputDir, ...
    sprintf('cfd_centerlines_N%d.png',N)), ...
    'Resolution',300);

%% 19. CONVERGENCE PLOT

fig4 = figure('Color','w');

semilogy(1:actualIterations, ...
    max(residualHistory,eps), ...
    'LineWidth',1.2);

hold on;

yline(steadyTolerance,'r--', ...
    'Target tolerance');

xlabel('Iteration');
ylabel('Maximum vorticity change / dt');

title('Finite Difference CFD Convergence');

grid on;

exportgraphics(fig4,fullfile(outputDir, ...
    sprintf('cfd_convergence_N%d.png',N)), ...
    'Resolution',300);

%% 20. EXPORT CENTERLINE DATA

centerlineU = table(y(:),u(:,mid), ...
    'VariableNames',{'y','u'});

centerlineV = table(x(:),v(mid,:)', ...
    'VariableNames',{'x','v'});

writetable(centerlineU,fullfile(outputDir, ...
    sprintf('cfd_centerline_u_N%d.csv',N)));

writetable(centerlineV,fullfile(outputDir, ...
    sprintf('cfd_centerline_v_N%d.csv',N)));

fprintf('\n===== STAGE 9K RUN COMPLETED =====\n');

