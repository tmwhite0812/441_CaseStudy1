%% ESE 441 Epidemic Model Case Study (Networked Model with Control Inputs)

% Parameters (same as before)
V1 = [0.3, 1.0, 2.5];
K1 = [0.8, 0.5, 0.3];
K2 = [0.2, 0.5, 0.9];
alpha = [0.1, 0.3, 0.7];
r = [0.8, 0.5, 0.2];
weeks = [0 10];  % Reduced simulation period

% Interaction matrices
C = [0, 1, 2; 2, 0, 8; 5, 10, 0];
D = [0, 5, 10; 1, 0, 12; 3, 8, 0];

% Control gains
K = [-3, 0; 0, -5];

% Desired equilibrium point
desired_x1_eq = 0.7;
desired_x2_eq = 0.0;

% Initial conditions for each region
ics = [0.95, 0.05; 0.50, 0.50; 0.20, 0.80];
initial_conditions_vector = reshape(ics', [], 1);

% ODE solver options: Reduce precision and limit output points
options = odeset('RelTol',1e-3, 'AbsTol',1e-6, 'MaxStep',0.1);

% Use a faster, stiff solver (ode15s)
[t, x] = ode15s(@(t, x) networked_epidemic_with_feedback(t, x, V1, K1, K2, r, alpha, C, D, K, desired_x1_eq, desired_x2_eq), weeks, initial_conditions_vector, options);

%% Visualization (Same as before)
figure;
subplot(3, 1, 1);
hold on;
plot(t, x(:, 1) * 100, 'r', 'LineWidth', 1.5);
plot(t, x(:, 2) * 100, 'b', 'LineWidth', 1.5);
plot(t, x(:, 3) * 100, 'g', 'LineWidth', 1.5);
plot(t, x(:, 4) * 100, 'm', 'LineWidth', 1.5);
plot(t, x(:, 5) * 100, 'k', 'LineWidth', 1.5);
plot(t, x(:, 6) * 100, 'c', 'LineWidth', 1.5);
xlabel('Weeks');
ylabel('Population Percentage');
legend({'R1 Susceptible', 'R1 Infected', 'R2 Susceptible', 'R2 Infected', 'R3 Susceptible', 'R3 Infected'}, 'Location', 'best');
title('Comparison of Susceptible and Infected Populations Across Regions');
grid on;

subplot(3, 1, 2);
interaction_S = sum(C, 2);
interaction_I = sum(D, 2);
bar([interaction_S, interaction_I]);
set(gca, 'XTickLabel', {'Region 1', 'Region 2', 'Region 3'});
xlabel('Regions');
ylabel('Interaction Contribution');
legend({'Susceptible Interactions', 'Infected Interactions'}, 'Location', 'NorthEast');
title('Inter-Regional Interactions for Susceptible and Infected Populations');

subplot(3, 1, 3);
diff_susceptible = diff(x(:, 1:2:end), 1);
diff_infected = diff(x(:, 2:2:end), 1);
plot(t(2:end), diff_susceptible, 'LineWidth', 1.5);
hold on;
plot(t(2:end), diff_infected, '--', 'LineWidth', 1.5);
xlabel('Weeks');
ylabel('Change in Population Percentage');
legend({'R1 Susceptible', 'R2 Susceptible', 'R3 Susceptible', 'R1 Infected', 'R2 Infected', 'R3 Infected'}, 'Location', 'best');
title('Change in Susceptible and Infected Populations Over Time');
grid on;

%% Function for the Networked Epidemic Model with Feedback Control
function dxdt = networked_epidemic_with_feedback(t, x, V1, K1, K2, r, alpha, C, D, K, x1_desired, x2_desired)
    N = length(V1);  % Number of regions
    dxdt = zeros(2 * N, 1);  % Initialize the derivative vector
    
    % Compute derivatives for each region
    for i = 1:N
        x1i = x(2 * i - 1);
        x2i = x(2 * i);

        % Compute interaction terms
        interaction_x1 = sum(C(i, :) .* (x(1:2:end)' - x1i));
        interaction_x2 = sum(D(i, :) .* (x(2:2:end)' - x2i));

        % Control inputs using feedback strategy
        u = K * ([x1i; x2i] - [x1_desired; x2_desired]);
        u1 = u(1);  
        u2 = u(2);  

        % ODEs for the susceptible and infected populations
        dx1i = -V1(i) * x1i * x2i / (K1(i) + x2i) + alpha(i) * x2i + u1 - interaction_x1;
        dx2i = V1(i) * x1i * x2i / (K1(i) + x2i) - r(i) * x2i / (x2i + K2(i)) - alpha(i) * x2i + u2 - interaction_x2;

        % Store the derivatives
        dxdt(2 * i - 1) = dx1i;
        dxdt(2 * i) = dx2i;
    end
end
