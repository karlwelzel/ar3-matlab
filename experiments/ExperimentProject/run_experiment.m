% This script runs a single experiment and saves its results to an .xlsx file.
% Ensure that you run `setup` before this. The options for each parameter that
% are presented in this file are suggestions, but do not cover the whole range
% of possible parameter combinations.
% We use `inner_` to refer options at the AR3 subproblem inner solver level.
% We use `inner_inner_` to refer options at the AR2 subproblem inner solver level.
% MCMR extends MCM into any order.

format short g;
% setup;

% AR2 or AR3
p = {
     % 2
     3
    };

% Choose initial sigma0
update_sigma0 = {
                 % 1e-8 % any positive real number
                 'TAYLOR'
                };

% Different test problems
problems = {
            % 1 % MGH test, 1 to 35
            'rosenbrock' % Multidimensional Rosenbrock
            % 'nonlinear_least_squares'
            % 'ill_cond_bm' % Well-conditioned Regularized 3rd-order polynomials
            % 'ill_cond_H' % Ill-conditioned Hessian ...
            % 'ill_cond_T' % Ill-conditioned Tensor ...
            % 'ill_cond_HT' % Ill-conditioned Hessian and Tensor ...
           };

if isnumeric(problems{1})
    problem = string(problems{1});
else
    % Set dimensions for Non-MGH problems
    if strcmp(problems{1}, "d1_fun")
        problem = jsonencode(struct(name = problems{1}, dim = 1));
    else
        problem = jsonencode(struct(name = problems{1}, dim = 100));
    end
end

% Starting point
x0_type = {
           'default'
           % 'randn'
           % 'rand'
           % 'uniform'
          };

% Three update options for the main algorithm
update_type = {
               % 'Simple'
               'Interpolation_m'
               % 'BGMS'
              };

% Activate pre-rejection module
update_use_prerejection = {
                           true
                           % false
                          };

% Three update options for the inner solver AR2
inner_update_type = {
                     'Simple'
                     % 'Interpolation_m'
                     % 'BGMS'
                    };

% Termination rule of the inner solver of AR3
inner_stop_rule = {
                   'First_Order'  % Absolute termination condition (TC.a)
                   % 'ARP_Theory'  % Relative termination condition (TC.r)
                   % 'General_Norm'  % Generalized norm termination condition (TC.g) from appendix
                  };

% Run experiment for the given parameters
if p{1} == 3
    params = struct( ...
                    p = 3, ...
                    problem = problem, ...
                    x0_type = x0_type{1}, ...
                    update_sigma0 = update_sigma0{1}, ...
                    update_type = update_type{1}, ...
                    update_use_prerejection = update_use_prerejection{1}, ...
                    stop_rule = 'First_Order', ...
                    stop_tolerance_g = 1e-8, ...
                    inner_solver = 'AR2', ...
                    inner_update_type = inner_update_type{1}, ...
                    inner_stop_rule = inner_stop_rule{1}, ...
                    inner_stop_tolerance_g = 1e-9, ...
                    inner_inner_solver = 'MCMR', ...
                    inner_inner_stop_rule = 'First_Order', ...
                    inner_inner_stop_tolerance_g = 1e-10);
else
    params = struct( ...
                    p = 2, ...
                    problem = problem, ...
                    x0_type = x0_type{1}, ...
                    update_sigma0 = update_sigma0{1}, ...
                    update_type = update_type{1}, ...
                    stop_rule = 'First_Order', ...
                    stop_tolerance_g = 1e-8, ...
                    inner_solver = 'MCMR', ...
                    inner_stop_rule = inner_stop_rule{1}, ...
                    inner_stop_tolerance_g = 1e-9);
    update_use_prerejection = {false};
end
table = struct2table(training(params));

% Save the run history to its own file
folder = append('./records/', problem);
[~, ~] = mkdir(folder);
name = append(folder, '/', update_type{1}, '_', inner_stop_rule{1}, ...
              '_', x0_type{1}, '_', num2str(p{1}), '_sigma_', num2str(update_sigma0{1}), ...
              '_prerejection_', string(update_use_prerejection{1}), '.xlsx');
disp(name);
writetable(table, name);

% Display final iteration
table.sub_status = double(table.sub_status); % enforce double
disp(table(end, :));
