% Define parameters
V1 = 1.2;    % Infection rate
K1 = 0.6;    % Saturation constant for infection
K2 = 0.3;    % Saturation constant for recovery
alpha = 0.4; % Reinfection rate
r = 0.2;     % Recovery rate

% Feedback gains
k11 = 0.5; 
k12 = 0.2; 
k21 = 0.1; 
k22 = 0.3;



% Initial guess for the equilibrium point [x1, x2]
x0 = [.25, .25s];

% Solve the system of nonlinear equations using fsolve, passing parameters
options = optimset('Display', 'iter'); % Display iteration steps (optional)
[x_sol, fval] = fsolve(@(x) equilibrium_system(x, V1, K1, K2, alpha, r, k11, k12, k21, k22), x0, options);

% Display the solution
fprintf('Equilibrium point: x1 = %.4f, x2 = %.4f\n', x_sol(1), x_sol(2));

% Define the system of nonlinear equations, passing all parameters
function F = equilibrium_system(x, V1, K1, K2, alpha, r, k11, k12, k21, k22)
    x1 = x(1);
    x2 = x(2);
    
    % First equation (from dx1/dt = 0)
    F(1) = x2 * (V1 * x1 / (K1 + x2) - alpha - k12) - k11 * x1;
    
    % Second equation (from dx2/dt = 0)
    F(2) = x2 * (V1 * x1 / (K1 + x2) - r / (x2 + K2) - alpha + k22) + k21 * x1;
end