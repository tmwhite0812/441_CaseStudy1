%% ESE 441 Epidemic Model Case Study 
% Keeler Tardiff and Tyler White 
%% No inputs of u(t)
V1 = [.20, .80];         % Infection rates
K1 = [0.30, 0.70];       % Saturation constants for infection  
K2 = 0.5;                % Saturation constant for recovery 
alpha = [0.25, 0.50];    % Reinfection rates
recovery = 0.2;   % Recovery rates  
weeks = [0 100];          % Simulation time

ics = [
    0.99, 0.01;   % 99% Susceptible, 1% Infected
    0.90, 0.10;   % 90% Susceptible, 10% Infected
    0.50, 0.50;   % 50% Susceptible, 50% Infected 
];

for i = 1:3  % iterating through infection rates
    for k1 = 1:length(K1_values)  % iterating through K1 values
        for r_idx = 1:length(r_values)  % iterating through recovery rates
            for alpha_idx = 1:length(alpha_values)  % iterating through alpha values
                alpha = alpha_values(alpha_idx);
                for j = 1:size(ics, 1)  % iterating through initial conditions
                    ic = ics(j, :); 

                    % solving diff eq using function 
                    [t, x] = ode45(@(t, x) epidemic_model(t, x, V1(i), K1_values(k1), ...
                                  r_values(r_idx), K2, alpha), weeks, ic);

                    % extracting eq point 
                    xeq1 = x(end, 1);  
                    xeq2 = x(end, 2);  

                    % analytical evaluation... personal usage 
                    x1_analytical = (alpha * K1_values(k1)) / V1(i);

                    % jacobian and eigen value extraction, looking at second eigen
                    % value because first is 0 
                    J = [0, -V1(i) * xeq1 / K1_values(k1) + alpha;
                         0,  V1(i) * xeq1 / K1_values(k1) - r_values(r_idx) / K2 - alpha];
                    eigenvalues = eig(J);
                    lambda_2 = eigenvalues(2);

                    % stability check 
                    if lambda_2 < 0
                        stability = 'Stable';
                    elseif lambda_2 > 0
                        stability = 'Unstable';
                    else
                        stability = 'Neutrally Stable';
                    end

                    % LINEARIZATION 
                    A = J;
                    linear_ic = [ic(1) - xeq1, ic(2) - xeq2];  
                    [t_linear, delta_x] = ode45(@(t, x) linearized_model(t, x, A), weeks, linear_ic);

                    % interpolation to match times 
                    x_linear_interp = interp1(t_linear, delta_x, t);
                    x_linear = zeros(length(t), 2);
                    % adding xeq back into to make the x+deltax for linearization 
                    x_linear(:, 1) = x_linear_interp(:, 1) + xeq1;
                    x_linear(:, 2) = x_linear_interp(:, 2) + xeq2;

                    % plotting 
                    figure;
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
                    title({sprintf('V1 = %.1f, K1 = %.1f, K2 = %.1f, \\alpha = %.2f, r = %.1f, Stability: %s', ...
                           V1(i), K1_values(k1), K2, alpha, r_values(r_idx), stability), ...
                           sprintf('Initial Condition: x1 = %.2f, x2 = %.2f', ic(1), ic(2))});
                    annotation('textbox', [0.15, 0.7, 0.1, 0.1], ...
                        'String', sprintf('Analytic x_1: %.4f', x1_analytical), 'FitBoxToText', 'on');
                    xlim([0 100]);
                    ylim([0 1]);  % Fixed Y-axis to range from 0 to 1
                    yticks(0:0.1:1);  % Set ticks at increments of 0.1
                    grid on;

                    fprintf('Simulation %d, K1 = %.1f, r = %.1f, \\alpha = %.2f, Initial Condition %d: Equilibrium (x1_sim, x1_analytical, x2) = (%.4f, %.4f, %.4f)\n', ...
                        i, K1_values(k1), r_values(r_idx), alpha, j, xeq1, x1_analytical, xeq2);
                    fprintf('Simulation %d, K1 = %.1f, r = %.1f, \\alpha = %.2f, Initial Condition %d: Eigenvalue λ2 = %.4f -> %s\n', ...
                        i, K1_values(k1), r_values(r_idx), alpha, j, lambda_2, stability);
                end
            end
        end
    end
end

%% functions used 
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


