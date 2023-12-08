clear, clc
% Model parameters
a = 1;
b = 0.1;
sigma = 5;

% Model's Drift and difussion functions
F = @(x, t) a * (b - x);
G = @(x, t) sigma * sqrt(x);

% Initial Conditions
X0 = 1250; %First week value of X
N = 240; %weeks in 5 years
T = 1; %weekly
n_trials = 100;

% Stochastic Differential Equation (SDE) Object
% see: https://es.mathworks.com/help/finance/sde.html

sdeModel = sde(F, G, 'StartState', X0, 'StartTime', 0, 'Correlation', 6);
%model 10, 0, 6

% making n_trials simulations of SDE 
[X, t] = simulate(sdeModel, N, 'DeltaTime', T/N, 'nTrials', n_trials);

% Plot Time Series Output
figure;
colors = lines(min(size(X, 3), n_trials));
for dim = 1:min(size(X, 3), n_trials)
    plot(t, X(:, :, dim), 'Color', colors(dim, :));
    hold on;
end
hold off;
xlabel('Time');
ylabel('Series Value');
title('M10a Time Serie Model Simulations');

% First Derivative Calculations
dt = T/N;
% dt = 10;
dX = diff(X) / dt;
t_derivative = t(1:end-1) + dt/2;

figure;
plot(t_derivative, dX(:, :, 1), 'Color', 'red');
xlabel('Time');
ylabel('First Difference');
title('Time Series First Difference');

% Second Derivative Calculation
ddX = diff(dX) / dt;
t_segunda_derivada = t_derivative(1:end-1) + dt/2;

% Second Derivative Plot
figure;
plot(t_segunda_derivada, ddX(:, :, 1), 'Color', 'green');
xlabel('Time');
ylabel('Second Difference');
title('Time Series Second Difference')

%Write a Table of Data

output_data = t;

for i=1:n_trials
    output_data = [output_data, X(:,:,i)];
end

table = array2table(output_data);
disp(table);


% Histogram Plot
figure;
for i = 1:20:n_trials
    [f, xi] = ksdensity(X(:, :, i));
    plot(xi, f, 'Color', colors(i, :), 'LineWidth', 0.5);
    hold on;
    fill(xi, f, colors(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

title('Histograms for Simulations');
xlabel('Series Value');
ylabel('Frequency');
% legend(arrayfun(@(x) ['Simulation ' num2str(x)], 1:n_trials, 'UniformOutput', false), 'Location', 'BestOutside');
hold off;

save("simulations.dat", "table")
save("Variables.mat")
