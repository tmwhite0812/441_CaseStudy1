%% ESE 441 Epidemic Model Case Study
% Keeler Tardiff and Tyler White
%% Parameters
V1 = [.2, .8];        % Infection rates
K1 = [.4, .6];        % Saturation constants for infection
K2 = .5;              % Saturation constant for recovery
alpha = [.25, .50];   % Reinfection rates
r = 0.2;              % Constant recovery rate
weeks = [0 5];      % Time span

% Define the B matrix for controllability
B = [1 0; 0 1];  % Simple B matrix that allows direct control of both states

%% Desired Equilibrium Point (for disease eradication)
desired_x1_eq = 0.7;  % Example target for susceptible population
desired_x2_eq = 0.0;  % Desired equilibrium for infected population

% Control gains (feedback gains for u = K^T * x)
K = [-3, 0; 0, -5];  % Feedback gains designed to place eigenvalues in the left half-plane

%% Initial Conditions
ics = [
    0.90, 0.10;  % 90% susceptible, 10% infected
    0.50, 0.50;  % 50% susceptible, 50% infected
];

%% Simulation Loop
for i = 1:length(V1)  % Iterate over each parameter set
    for j = 1:size(ics, 1)  % Iterate over initial conditions
        ic = ics(j, :);

        % Check if initial conditions are valid
        if abs(sum(ic) - 1) > 1e-3
            error('Initial conditions must sum to 1.');
        end

        % Simulate Nonlinear Model with Feedback Control
        [t, x] = ode45(@(t, x) epidemic_model_with_feedback(t, x, V1(i), K1(i), ...
                              r, K2, alpha(i), K, desired_x1_eq, desired_x2_eq), weeks, ic);

        % Extract equilibrium points from nonlinear simulation
        xeq1 = x(end, 1);  % Susceptible equilibrium
        xeq2 = x(end, 2);  % Infected equilibrium

        % Linearize around the equilibrium point (xeq1, xeq2)
        A = [0, -V1(i) * xeq1 / K1(i) + alpha(i);
             0, V1(i) * xeq1 / K1(i) - r / K2 - alpha(i)];

        % Simulate the Linearized Model around the equilibrium point
        linear_ic = [0, 0];  % Zero deviation from equilibrium
        [t_linear, delta_x] = ode45(@(t, x) linearized_model(t, x, A, B, K), weeks, linear_ic);

        % Interpolate linearized results to match nonlinear time steps
        x_linear_interp = interp1(t_linear, delta_x, t);
        x_linear = x_linear_interp + [xeq1, xeq2];  % Offset by equilibrium points

        %% Plot the Results for Comparison
        figure;
        plot(t, x(:, 1) * 100, 'r', 'LineWidth', 1.5);  % Nonlinear Susceptible
        hold on;
        plot(t, x(:, 2) * 100, 'b', 'LineWidth', 1.5);  % Nonlinear Infected
        plot(t, x_linear(:, 1) * 100, 'r--', 'LineWidth', 1.5);  % Linearized Susceptible
        plot(t, x_linear(:, 2) * 100, 'b--', 'LineWidth', 1.5);  % Linearized Infected
        legend('x_1 (Linearized Susceptible)', 'x_2 (Linearized Infected)', ...
               'x_1 (Desired Equilibrium)', 'x_2 (Desired Equilibrium)', ...
               'Location', 'best');
        xlabel('Weeks');
        ylabel('Population Percentage');
        title(sprintf('Initial Conditions: Susceptible = %.1f, Infected = %.1f', ic(1), ic(2)));
        ylim([0 100]);  % Fix the y-axis to range from 0 to 100
        grid on;

        % Save plot for each simulation
        saveas(gcf, sprintf('comparison_V1_%.1f_K1_%.1f_IC_%.2f_%.2f.png', V1(i), K1(i), ic(1), ic(2)));

        fprintf('Simulation %d, Initial Condition %d: Equilibrium (x1, x2) = (%.4f, %.4f)\n', i, j, xeq1, xeq2);
    end
end

%% Function for the Linearized Model
function dxdt = linearized_model(t, x, A, B, K)
    u = K * x;  % Feedback control
    dxdt = A * x + B * u;  % Linearized system dynamics
end

%% Function for the Epidemic Model with Feedback Control
function dxdt = epidemic_model_with_feedback(t, x, V1, K1, r, K2, alpha, K, x1_desired, x2_desired)
    x1 = x(1);  % Susceptible
    x2 = x(2);  % Infected
    
    % Feedback control: u = K^T * (x - desired_x)
    u = K * (x - [x1_desired; x2_desired]);

    u1 = u(1);  % Control input for susceptible equation
    u2 = u(2);  % Control input for infected equation

    dx1 = -V1 * x1 * x2 / (K1 + x2) + alpha * x2 + u1;
    dx2 = V1 * x1 * x2 / (K1 + x2) - r * x2 / (x2 + K2) - alpha * x2 + u2;

    dxdt = [dx1; dx2];
end
