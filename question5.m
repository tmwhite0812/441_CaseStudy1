%% ESE 441 Epidemic Model Case Study
% Keeler Tardiff and Tyler White
%% Parameters
V1 = [.2, .8];        % Infection rates
K1 = [.3, .7];        % Saturation constants for infection
K2 = .5;              % Saturation constant for recovery
alpha = [.25, .50];   % Reinfection rates
r = 0.2;              % Constant recovery rate
weeks = [0 100];      % Time span

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
        [t, x] = ode45(@(t, x) epidemic_model_with_feedback(t, x, V1(i), K1(i), r, K2, alpha(i), K, desired_x1_eq, desired_x2_eq), weeks, ic);

        % Equilibrium Points
        xeq1 = x(end, 1);  % Susceptible equilibrium
        xeq2 = x(end, 2);  % Infected equilibrium

        % Jacobian (A matrix) and eigenvalues for stability analysis
        A = [0, -V1(i) * xeq1 / K1(i) + alpha(i);
             0, V1(i) * xeq1 / K1(i) - r / K2 - alpha(i)];
        
        eigenvalues = eig(A);
        lambda_1 = eigenvalues(1);
        lambda_2 = eigenvalues(2);

        % Check stability based on eigenvalues
        if all(eigenvalues < 0)
            stability = 'Stable';
        elseif any(eigenvalues > 0)
            stability = 'Unstable';
        else
            stability = 'Neutrally Stable';
        end

        %% Controllability Check
        C = ctrb(A, B);  % Controllability matrix
        rank_C = rank(C);  % Rank of the controllability matrix
        
        if rank_C == size(A, 1)
            controllability_status = 'Controllable';
        else
            controllability_status = 'Not Controllable';
        end

        %% Plot Results
        figure;
        plot(t, x(:, 1) * 100, 'r', 'LineWidth', 1.5);  % Susceptible
        hold on;
        plot(t, x(:, 2) * 100, 'b', 'LineWidth', 1.5);  % Infected
        legend('x_1 (Susceptible)', 'x_2 (Infected)', 'Location', 'best');
        xlabel('Week Since Intervention');
        ylabel('Population Percentage');
        title(sprintf('Simulation: V1 = %.1f, K1 = %.1f, K2 = %.1f, \\alpha = %.2f, IC: [%.2f, %.2f], Stability: %s, Controllability: %s', ...
               V1(i), K1(i), K2, alpha(i), ic(1) * 100, ic(2) * 100, stability, controllability_status));
        annotation('textbox', [0.15, 0.7, 0.1, 0.1], 'String', sprintf('Desired Eq: [%.2f, %.2f]', desired_x1_eq, desired_x2_eq), 'FitBoxToText', 'on');
        grid on;

        % Save plot for each simulation
        saveas(gcf, sprintf('simulation_V1_%.1f_IC_%.2f_%.2f_with_feedback.png', V1(i), ic(1), ic(2)));

        fprintf('Simulation %d, Initial Condition %d: Equilibrium (x1, x2) = (%.4f, %.4f)\n', i, j, xeq1, xeq2);
        fprintf('Controllability Check: %s (Rank: %d)\n', controllability_status, rank_C);
    end
end

%% Function for the Epidemic Model with Feedback Control
function dxdt = epidemic_model_with_feedback(t, x, V1, K1, r, K2, alpha, K, x1_desired, x2_desired)
    % State variables
    x1 = x(1);  % Susceptible
    x2 = x(2);  % Infected
    
    % Feedback control: u = K^T * (x - desired_x)
    u = K * (x - [x1_desired; x2_desired]);

    % Extract control inputs
    u1 = u(1);  % Control input for dx1 (susceptible)
    u2 = u(2);  % Control input for dx2 (infected)

    % ODE system with control inputs u1 and u2 applied to the correct equations
    dx1 = -V1 * x1 * x2 / (K1 + x2) + alpha * x2 + u1;  % Susceptible equation
    dx2 = V1 * x1 * x2 / (K1 + x2) - r * x2 / (x2 + K2) - alpha * x2 + u2;  % Infected equation

    % Return the derivative
    dxdt = [dx1; dx2];
end
