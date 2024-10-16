%% ESE 441 Epidemic Model Case study 
% Keeler Tardiff and Tyler White 
%% No external input 
V1 = [1.2, 0.9, 2];      % infection rates 
K1 = [0.6, 0.4, 0.5];    % saturation constants for infection
K2 = [0.3, 0.6, 0.7];    % saturation constants for recovery
alpha = [0.4, 0.3, 0.5]; % reinfection rates
r = 0.2;                 % constant recovery rate
time_length = [0 100];

% testing initial conditions to see how linearized and nonlinearized compare
ics = [
    0.9, 0.1;   % 90% susceptible, 10% infected
    0.35, 0.65; % 35% susceptible, 65% infected
    0.50, 0.50; % 50% susceptible, 50% infected
];
for i = 1:3
    for j = 1:size(ics, 1)
        ic = ics(j, :);
        
        [t, x] = ode45(@(t, x) epidemic_model(t, x, V1(i), K1(i), r, K2(i), alpha(i)), time_length, ic);
        
        %  equilibrium points 
        xeq1 = x(end, 1);  % susceptible equilibrium
        xeq2 = 0;          % infected equilibrium 
        
        % xeq1 = alpha*K1/V1
        x1eq_analytical = (alpha(i) * K1(i)) / V1(i);
        
        % Jacobian matrix at (xeq1, 0) and eigenvalues 
        J = [0, -V1(i)*xeq1/K1(i) + alpha(i);
             0,  V1(i)*xeq1/K1(i) - r/K2(i) - alpha(i)];
        eigenvalues = eig(J);
        lambda_2 = eigenvalues(2);
        
        % Stability
        if lambda_2 < 0
            stability = 'Stable';
        elseif lambda_2 > 0
            stability = 'Unstable';
        else
            stability = 'Neutrally Stable';
        end
        
        % Assign Jacobian matrix J to A
        A = J;
        
        % linearized system
        linear_ic = [ic(1) - xeq1, ic(2) - xeq2];  
        [t_linear, delta_x] = ode45(@(t, x) linearized_model(t, x, A), time_length, linear_ic);
        
        % interpolate the linearized solution to match the time points of the nonlinear solution
        % checking interpolation by making sure time vectors match
        if t_linear(1) <= t(1) && t_linear(end) >= t(end)
            x_linear_interp = interp1(t_linear, delta_x, t);
        else
            error('Time vectors for systems do not match.');
        end
        
        % adding eq back into linearized 
        x_linear = zeros(length(t), 2);  % Preallocate the x_linear matrix
        x_linear(:, 1) = x_linear_interp(:, 1) + xeq1;
        x_linear(:, 2) = x_linear_interp(:, 2) + xeq2;
        
        figure;
        plot(t, x(:, 1), 'r', 'LineWidth', 1.5);  % nonlinear susceptible
        hold on;
        plot(t, x(:, 2), 'b', 'LineWidth', 1.5);  % nonlinear infected
        plot(t, x_linear(:, 1), 'r:', 'LineWidth', 1.5);  % linearized susceptible 
        plot(t, x_linear(:, 2), 'b:', 'LineWidth', 1.5);  % linearized infected 
        legend('Nonlinear x_1 (Susceptible)', 'Nonlinear x_2 (Infected)', ...
               'Linearized x_1 (Susceptible)', 'Linearized x_2 (Infected)', ...
               'Location', 'best');
        xlabel('Time');
        ylabel('Population');
        title(sprintf('Simulation: V1 = %.1f, K1 = %.1f, K2 = %.1f, \\alpha = %.2f, Stability: %s', ...
               V1(i), K1(i), K2(i), alpha(i), stability));
        annotation('textbox', [0.15, 0.7, 0.1, 0.1], 'String', sprintf('Analytic x_1: %.4f', x1eq_analytical), 'FitBoxToText', 'on');
        
        grid on;
        fprintf('Simulation %d, Initial Condition %d: Equilibrium (x1_sim, x1_analytic, x2) = (%.4f, %.4f, %.4f)\n', ...
            i, j, xeq1, x1eq_analytical, xeq2);
        fprintf('Simulation %d, Initial Condition %d: Eigenvalue λ2 = %.4f -> %s\n', i, j, lambda_2, stability);
        
    end
end

%% Function used to simulate the epidemic model
function dxdt = epidemic_model(t, x, V1, K1, r, K2, alpha)
    x1 = x(1);  % susceptible 
    x2 = x(2);  % infected 
    dx1 = -V1*x1*x2 / (K1 + x2) + alpha*x2;
    dx2 = V1*x1*x2 / (K1 + x2) - r*x2 / (x2 + K2) - alpha*x2;
    dxdt = [dx1; dx2];
end

%% Linearized system 
function dxdt = linearized_model(t, x, A)
    dxdt = A * x;  % Linearized system dynamics
end
