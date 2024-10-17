%% ESE 441 Epidemic Model Case Study 
% Keeler Tardiff and Tyler White 
%% Parameters
V1 = [.20, .80];         % Infection rates
K1 = [0.30, 0.70];       % Saturation constants for infection  
K2 = 0.5;                % Saturation constant for recovery 
alpha = [0.35, 0.45];    % Reinfection rates
recovery = [0.0, 0.2];   % Recovery rates  
weeks = [0 100];          % Simulation time

ics = [0.99, 0.01; 0.50, 0.50];  
% 99% Susceptible, 1% Infected   % 50% Susceptible, 50% Infected 

% Nested loops for all parameter combinations
for i = 1:length(V1)  
    for a = 1:length(alpha)  
        for k = 1:length(K1)  
            for r = 1:length(recovery)  
                for j = 1:size(ics, 1)  
                    ic = ics(j, :); 
                    
                    % Display current parameters
                    fprintf('Running simulation for V1=%.2f, alpha=%.2f, K1=%.2f, r=%.2f, IC=[%.2f, %.2f]\n', ...
                        V1(i), alpha(a), K1(k), recovery(r), ic(1), ic(2));

                    % Solve the differential equation
                    [t, x] = ode45(@(t, x) epidemic_model(t, x, V1(i), K1(k), ...
                                      recovery(r), K2, alpha(a)), weeks, ic);

                    % Extract equilibrium point 
                    xeq1 = x(end, 1);  
                    xeq2 = x(end, 2);  

                    % Analytical evaluation
                    x1_analytical = (alpha(a) * K1(k)) / V1(i);

                    % Jacobian matrix construction
                    J = [0, -V1(i) * xeq1 / K1(k) + alpha(a);
                         0,  V1(i) * xeq1 / K1(k) - recovery(r) / K2 - alpha(a)];

                    eigenvalues = eig(J);  
                    lambda_2 = eigenvalues(2);

                    % Stability check and phase portrait call if stable
                    if lambda_2 < 0
                        stability = 'Stable';
                        % phase_portrait(ic, V1(i), K1(k), K2, recovery(r), ...
                        %     alpha(a), 0.2, ...
                        %     sprintf('Phase Portrait: V1 = %.2f, K1 = %.2f, r = %.2f, \\alpha = %.2f, IC = [%.2f, %.2f]', ...
                        %     V1(i), K1(k), recovery(r), alpha(a), ic(1), ic(2)));
                    elseif lambda_2 > 0
                        stability = 'Unstable';
                    else
                        stability = 'Neutrally Stable';
                    end

                    % Linearization 
                    A = J;
                    linear_ic = [ic(1) - xeq1, ic(2) - xeq2];  
                    [t_linear, delta_x] = ode45(@(t, x) linearized_model(t, x, A), weeks, linear_ic);

                    % Interpolation to match times 
                    x_linear_interp = interp1(t_linear, delta_x, t);
                    x_linear = zeros(length(t), 2);
                    x_linear(:, 1) = x_linear_interp(:, 1) + xeq1;
                    x_linear(:, 2) = x_linear_interp(:, 2) + xeq2;

                    % Plotting 
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
                    title({sprintf('Simulation: V1 = %.1f, K1 = %.1f, K2 = %.1f, \\alpha = %.2f, r = %.1f, Stability: %s', ...
                           V1(i), K1(k), K2, alpha(a), recovery(r), stability), ...
                           sprintf('Initial Condition: x1 = %.2f, x2 = %.2f', ic(1), ic(2))});
                    annotation('textbox', [0.15, 0.7, 0.1, 0.1], ...
                        'String', sprintf('Analytic x_1: %.4f', x1_analytical), 'FitBoxToText', 'on');
                    xlim([0 100]);
                    ylim([0 1]);  
                    yticks(0:0.1:1);  
                    grid on;

                    fprintf('Simulation %d, K1 = %.1f, r = %.1f, \\alpha = %.2f, Initial Condition %d: Equilibrium (x1_sim, x1_analytical, x2) = (%.4f, %.4f, %.4f)\n', ...
                        i, K1(k), recovery(r), alpha(a), j, xeq1, x1_analytical, xeq2);
                    fprintf('Simulation %d, K1 = %.1f, r = %.1f, \\alpha = %.2f, Initial Condition %d: Eigenvalue λ2 = %.4f -> %s\n', ...
                        i, K1(k), recovery(r), alpha(a), j, lambda_2, stability);
                end
            end
        end
    end
end

%% Functions 
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

function phase_portrait(ic, V1, K1, K2, r, alpha, scale, labelText)
    [x1, x2] = meshgrid(0:0.01:1, 0:0.01:1);  
    x1dot = -(V1 * x1 .* x2) ./ (K1 + x2) + alpha * x2;
    x2dot = (V1 * x1 .* x2) ./ (K1 + x2) - (r * x2) ./ (x2 + K2) - alpha * x2;

    figure;
    quiver(x1, x2, x1dot, x2dot, scale * 2, 'LineWidth', 1.5); 
    xlim([0, 1]);  
    ylim([0, 1]);
    title(labelText);
end

