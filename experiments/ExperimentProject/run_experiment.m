% This script runs a single experiment and saves its results to an .xlsx file.
% Ensure that you run `setup` before this. The options for each parameter that
% are presented in this file are suggestions, but do not cover the whole range
% of possible parameter combinations.

format short g;
% setup;

p = {
     2
     % 3
    };

update_sigma0 = {
                 % 1e-8
                 % 'LIPSCHITZ'
                 % 'INVEXIFICATION'
                 'TAYLOR'
                };

problems = {
            % 'rosenbrock'
            % 'chebysv_rosenbrock'
            % 'quartic'
            % 'square'
            % 'd1_fun'
            % 'nonlinear_least_squares'
            % 'ill_cond_bm'
            % 'ill_cond_H'
            % 'ill_cond_T'
            % 'ill_cond_HT'
            % 'separable_function'
            % 'non_separable_function'
            1 % MGH test
           };

if isnumeric(problems{1})
    problem = string(problems{1});
else
    if strcmp(problems{1}, "d1_fun")
        problem = jsonencode(struct(name = problems{1}, dim = 1));
    else
        problem = jsonencode(struct(name = problems{1}, dim = 100));
    end
end

x0_type = {
           'default'
           % -1.205
           % 'randn'
           % 'rand'
           % 'uniform'
          };

update_type = {
               % 'Simple'
               % 'Interpolation_t'
               'Interpolation_m'
               % 'BGMS'
              };

update_use_prerejection = {
                           true
                           % false
                          };

inner_solver = {
                'AR2'
                % 'fminunc_trust-region',
                % 'fminunc_lbfgs',
                % 'fminunc_dfp',
                % 'fminunc_steepdesc',
               };

inner_update_type = {
                     'Simple'
                     % 'Interpolation_t'
                     % 'Interpolation_m'
                     % 'BGMS'
                    };

inner_inner_solver = {
                      'MCMR'
                      % 'trust-region'
                     };

inner_stop_rule = {
                   'First_Order'
                   % 'ARP_Theory'
                   % 'General_Norm'
                   % 'Cartis_G'
                   % 'Cartis_S'
                  };

inner_inner_stop_rule = {
                         'First_Order'
                         % 'ARP_Theory'
                         % 'Cartis_G'
                         % 'Cartis_S'
                        };

% Run experiment for the given parameters
params = struct( ...
                p = p{1}, ...
                problem = problem, ...
                x0_type = x0_type{1}, ...
                update_sigma0 = update_sigma0{1}, ...
                update_type = update_type{1}, ...
                update_use_prerejection = update_use_prerejection{1}, ...
                stop_rule = 'First_Order', ...
                stop_tolerance_g = 1e-8, ...
                inner_solver = inner_solver{1}, ...
                inner_update_type = inner_update_type{1}, ...
                inner_stop_rule = inner_stop_rule{1}, ...
                inner_stop_tolerance_g = 1e-9, ...
                inner_inner_solver = inner_inner_solver{1}, ...
                inner_inner_stop_rule = inner_inner_stop_rule{1}, ...
                inner_inner_stop_tolerance_g = 1e-10);
table = struct2table(training(params));

% Save the run history to its own file
folder = append('records/', problem);
[~, ~] = mkdir(folder);
name = append(folder, '/', update_type{1}, '_', inner_stop_rule{1}, ...
              '_', x0_type{1}, '_', num2str(p{1}), '_sigma_', num2str(update_sigma0{1}), ...
              '_prerejection_', string(update_use_prerejection{1}), '.xlsx');
disp(name);
writetable(table, name);

% Display final iteration
table.sub_status = double(table.sub_status); % enforce double
disp(table(end, :));
