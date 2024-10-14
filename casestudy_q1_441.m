%% ESE 441 Epidemic Model Case study 
% Keeler Tardiff and Tyler White 
%% No external input 
V1 = [1.2, .9, 2];       % Infection rates for different simulations
K1 = [.6, .4, .5];     % Saturation constants for infection
K2 = [.3, .6, .7];     % Saturation constants for recovery
alpha = [.4, .3, .5]; % Reinfection rates
r = .2;                % Constant recovery rate
time_length = [0 100];

% intial conditions 
ic = [0.9, 0.1];  % x1 = 90% susceptible, x2 = 10% infected

for i = 1:3
    % first solve differntial equation
    [t, x] = ode45(@(t, x) epidemic_model(t, x, V1(i), K1(i), r, K2(i), alpha(i)), time_length, ic);
    
    xeq1 = x(end, 1);  % susceptible
    xeq2 = 0;          % infected
    
    % x1 using x1 = (alpha * K1) / V1
    x1eq_analytical = (alpha(i) * K1(i)) / V1(i);
    
    % Jacobian matrix at (xeq1, 0)
    J = [0, -V1(i)*xeq1/K1(i) + alpha(i);
         0,  V1(i)*xeq1/K1(i) - r/K2(i) - alpha(i)];
    
    eigenvalues = eig(J);
    
    % obtaining relevant eigenvalue 2 to focus on stability 
    lambda_2 = eigenvalues(2);
    
    % stability of eigen value 2 check 
    if lambda_2 < 0
        stability = 'Stable';
    elseif lambda_2 > 0
        stability = 'Unstable';
    else
        stability = 'Neutrally Stable';
    end
    
    % values of eq point and stability 
    fprintf('Simulation %d: Equilibrium Point (x1_sim, x1_analytic, x2) = (%.4f, %.4f, %.4f)\n', ...
        i, xeq1, x1eq_analytical, xeq2);
    fprintf('Simulation %d: Eigenvalue λ2 = %.4f -> %s\n', i, lambda_2, stability);
  
    % LINEARIZED SYSTEM 
    A = J;  
    linear_ic = [ic(1) - xeq1, ic(2) - xeq2];  % Initial condition deviations from equilibrium
    
    % solving the linearized system 
    [t_linear, x_linear] = ode45(@(t, x) linearized_model(t, x, A), time_length, linear_ic);
    
    % comparing the linearized solution with nonlinear by adding in eq
    % points
    x_linear(:, 1) = x_linear(:, 1) + xeq1;
    x_linear(:, 2) = x_linear(:, 2) + xeq2;
    
    % new figure for each simulation 
    figure;
    plot(t, x(:, 1), 'r', 'LineWidth', 1.5); % Nonlinear susceptible 
    hold on;
    plot(t, x(:, 2), 'b', 'LineWidth', 1.5); % Nonlinear infected 
    plot(t_linear, x_linear(:, 1), 'r:', 'LineWidth', 1.5); % Linearized susceptible 
    plot(t_linear, x_linear(:, 2), 'b:', 'LineWidth', 1.5); % Linearized infected 
    scatter(t(end), xeq1, 100, 'r', 'filled');  
    scatter(t(end), xeq2, 100, 'b', 'filled');  
    legend('Nonlinear x_1 (Susceptible)', 'Nonlinear x_2 (Infected)', ...
        'Linearized x_1 (Susceptible)', 'Linearized x_2 (Infected)', ...
        'Location', 'best');
    xlabel('Time');
    ylabel('Population');
    title(sprintf('Simulation %d: V1 = %.1f, K1 = %.1f, K2 = %.1f, \\alpha = %.2f, Stability: %s', ...
        i, V1(i), K1(i), K2(i), alpha(i), stability));

    % noting analytical eq value 
    annotation('textbox', [0.15, 0.7, 0.1, 0.1], 'String', sprintf('Analytic x_1: %.4f', x1eq_analytical), 'FitBoxToText', 'on');
   
    grid on;
end

%% Function used to simulate the epidemic model
function dxdt = epidemic_model(t, x, V1, K1, r, K2, alpha)
    x1 = x(1);  % susceptible 
    x2 = x(2);  % infected 
    dx1 = -V1*x1*x2 / (K1 + x2) + alpha*x2;
    dx2 = V1*x1*x2 / (K1 + x2) - r*x2 / (x2 + K2) - alpha*x2;
    dxdt = [dx1; dx2];
end

% Linearized system 
function dxdt = linearized_model(t, x, A)
    % A = J 
    dxdt = A * x;
end
