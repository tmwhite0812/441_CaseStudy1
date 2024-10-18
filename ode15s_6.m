%% ESE 441 Epidemic Model Case Study (Networked Model with Control Inputs)
% Keeler Tardiff and Tyler White

%% Parameters for Each Region
V1 = [0.3, 0.9, 2];        % Infection rates for each region
K1 = [0.6, 0.4, 0.5];      % Saturation constants for infection
K2 = [0.3, 0.6, 0.7];      % Saturation constants for recovery
alpha = [0.4, 0.3, 0.5];   % Reinfection rates
r = 0.2;                   % Constant recovery rate across all regions
weeks = [0 5];            % Simulation period in weeks

%% Interaction Matrices (Movement Between Regions)
C = [0, 5, 3;   % Updated Susceptible interaction matrix
     0, 0, 2;
     0, 0, 0];

D = [0, 4, 3;   % Updated Infected interaction matrix
     0, 0, 5;
     0, 0, 0];

%% Define the B Matrix for Controllability
B = [1 0; 0 1];  % Allows direct control of both states

%% Desired Equilibrium Point (Disease-Free Target)
desired_x1_eq = 0.1;  % Target for susceptible population fraction
desired_x2_eq = 0.0;  % Target for infected population (disease-free)

%% Control Gains (Feedback Strategy u = K^T * (x - desired_x))
K = [-3, 0; 0, -5];  % Control gains for each state

%% Initial Conditions for Each Region (Susceptible and Infected Fractions)
ics = [
    0.95, 0.05;   % Region 1: Mostly susceptible
    0.50, 0.50;   % Region 2: Half-half split
    0.20, 0.80    % Region 3: Mostly infected
];

%% Reshape the Initial Conditions into a Vector for ode45
initial_conditions_vector = reshape(ics', [], 1);  % Flatten to a column vector

%% Run the Simulation for the Networked System
[t, x] = ode45(@(t, x) networked_epidemic_with_feedback(t, x, V1, K1, r, K2, alpha, C, D, K, desired_x1_eq, desired_x2_eq), weeks, initial_conditions_vector);

%% Plot Combined Results for All Regions
%% Plot Results for All Regions on a Single Plot
figure;
hold on;

% Plot susceptible and infected populations for each region
for i = 1:3
    plot(t, x(:, (i - 1) * 2 + 1) * 100, 'LineWidth', 1.5);  % Susceptible percentage
    plot(t, x(:, (i - 1) * 2 + 2) * 100, 'LineWidth', 1.5);  % Infected percentage
end

% Customize the plot
xlabel('Weeks');
ylabel('Population Percentage');
legend({'Region 1 Susceptible', 'Region 1 Infected', ...
        'Region 2 Susceptible', 'Region 2 Infected', ...
        'Region 3 Susceptible', 'Region 3 Infected'}, ...
        'Location', 'best');
title('Susceptible and Infected Populations Across All Regions');
grid on;
hold off;


%% Function for the Networked Epidemic Model with Feedback Control
function dxdt = networked_epidemic_with_feedback(t, x, V1, K1, r, K2, alpha, C, D, K, x1_desired, x2_desired)
    N = length(V1);  % Number of regions
    dxdt = zeros(2 * N, 1);  % Initialize the derivative vector

    % Iterate through each region to compute derivatives
    for i = 1:N
        % Extract the state variables for region i
        x1i = x(2 * i - 1);  % Susceptible fraction in region i
        x2i = x(2 * i);      % Infected fraction in region i

        % Compute interaction terms for susceptible and infected populations
        sumx1 = 0;
        sumx2 = 0;
        for j = 1:N
            interaction_x1 = C(i, j) * (x1i - x(2 * j - 1));  % Susceptible interactions
            interaction_x2 = D(i, j) * (x2i - x(2 * j));      % Infected interactions
            sumx1 = sumx1 + interaction_x1;
            sumx2 = sumx2 + interaction_x2;
        end

        % Compute control input for each state using feedback strategy
        u = K * ([x1i; x2i] - [x1_desired; x2_desired]);

        % Extract control inputs for susceptible and infected populations
        u1 = u(1);  % Control input for susceptible population
        u2 = u(2);  % Control input for infected population

        % ODEs for the susceptible and infected populations with control inputs
        dx1i = -V1(i) * x1i * x2i / (K1(i) + x2i) + alpha(i) * x2i + u1 - sumx1;
        dx2i = V1(i) * x1i * x2i / (K1(i) + x2i) - r * x2i / (x2i + K2(i)) - alpha(i) * x2i + u2 - sumx2;

        % Store the derivatives in the correct positions in the derivative vector
        dxdt(2 * i - 1) = dx1i;  % Susceptible equation for region i
        dxdt(2 * i) = dx2i;      % Infected equation for region i
    end
end
