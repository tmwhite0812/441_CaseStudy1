%% ESE 441 Epidemic Model Case Study 
% Keeler Tardiff and Tyler White 
%% No inputs of u(t)
V1 = [.20, .80];           % Infection rates
K1 = [0.30, 0.70];         % Saturation constants for infection  
K2 = 0.5;                  % Saturation constant for recovery 
alpha = [0.25, 0.50];      % Reinfection rates
recovery = [0.2, .5];      % Recovery rates  
weeks = [0 100];           % Simulation time

ics = [
    0.99, 0.01;   % 99% Susceptible, 1% Infected
    0.90, 0.10;   % 90% Susceptible, 10% Infected
    0.50, 0.50;   % 50% Susceptible, 50% Infected 
];

for i = 1:length(V1)  
    for k = 1:length(K1) 
        for r = 1:length(recovery) 
            for a = 1:length(alpha)

                % grouping by parameter setting 
                figure;
                sgtitle(sprintf('V1 = %.2f, K1 = %.2f, K2 = %.2f, \\alpha = %.2f, r = %.2f', ...
                        V1(i), K1(k), K2, alpha(a), recovery(r)));

                % simulation of the parameters
                for j = 1:size(ics, 1)  

                    % iterating through the initial conditions  
                    ic = ics(j, :); 

                    % solving the differential equation 
                    [t, x] = ode45(@(t, x) epidemic_model(t, x, V1(i), K1(k), ...
                                      recovery(r), K2, alpha(a)), weeks, ic);

                    % extract the last value of the eq point to compute jacobian
                    % and linearize around the eq point 
                    xeq1 = x(end, 1);  
                    xeq2 = x(end, 2);  % will always be 0 

                    % jacobian and eig value computation
                    J = [0, -V1(i) * xeq1 / K1(k) + alpha(a);
                         0,  V1(i) * xeq1 / K1(k) - recovery(r) / K2 - alpha(a)];
                    eigenvalues = eig(J);
                    lambda_2 = eigenvalues(2);  % using second eigenvalue for stability.. first is 0

                    % stability check 
                    if lambda_2 < 0
                        stability = 'Stable';
                    elseif lambda_2 > 0
                        stability = 'Unstable';
                    else
                        stability = 'Neutrally Stable';
                    end

                    % linearzation of the system using same jacobian mtrx
                    A = J; 
                    linear_ic = [ic(1) - xeq1, ic(2) - xeq2]; 
                    [t_linear, delta_x] = ode45(@(t, x) linearized_model(t, x, A), weeks, linear_ic);

                    % interpolate to match original time points
                    x_linear_interp = interp1(t_linear, delta_x, t);
                    x_linear = zeros(length(t), 2);
                    x_linear(:, 1) = x_linear_interp(:, 1) + xeq1;  % x + delta x 
                    x_linear(:, 2) = x_linear_interp(:, 2) + xeq2;

                    % matching based on the initial conditions 
                    subplot(size(ics, 1), 1, j); 
                    plot(t, x(:, 1), 'r', 'LineWidth', 1.5);   
                    hold on;
                    plot(t, x(:, 2), 'b', 'LineWidth', 1.5);   
                    plot(t, x_linear(:, 1), 'r:', 'LineWidth', 1.5);   
                    plot(t, x_linear(:, 2), 'b:', 'LineWidth', 1.5);   
                    legend('Nonlinear x_1 (Susceptible)', 'Nonlinear x_2 (Infected)', ...
                           'Linearized x_1 (Susceptible)', 'Linearized x_2 (Infected)', ...
                           'Location', 'best');
                    xlabel('Weeks');
                    ylabel('Population');
                    title(sprintf('Initial Condition: x1 = %.2f, x2 = %.2f', ic(1), ic(2)));
                    xlim([0 100]);
                    ylim([0 1]);  
                    yticks(0:0.1:1); 
                    grid on;

                    % debugging to print to console 
                    fprintf('V1 = %.2f, K1 = %.2f, r = %.2f, \\alpha = %.2f, Initial Condition %d: (xeq1, xeq2) = (%.4f, %.4f)\n', ...
                            V1(i), K1(k), recovery(r), alpha(a), j, xeq1, xeq2);
                    fprintf('Eigenvalue λ2 = %.4f -> %s\n', lambda_2, stability);
                end
            end
        end
    end
end

%% Functions used 
function dxdt = epidemic_model(t, x, V1, K1, r, K2, alpha)
    x1 = x(1);  
    x2 = x(2);  
    dx1 = -V1 * x1 * x2 / (K1 + x2) + alpha * x2;
    dx2 = V1 * x1 * x2 / (K1 + x2) - r * x2 / (x2 + K2) - alpha * x2;
    dxdt = [dx1; dx2];
end

function dxdt = linearized_model(t, x, A)
    dxdt = A * x;
end
