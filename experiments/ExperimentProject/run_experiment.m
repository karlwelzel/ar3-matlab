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
     3
     % 2
     % 3
    };

% Choose initial sigma0
update_sigma0 = {
                 % 1e-8 % any potsitive real number
                 'TAYLOR'
                };

% Different test problems
problems = {
            % 3 % MGH test, 1 to 35
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
        problem = jsonencode(struct(name = problems{1}, dim = 1000000));
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
                   % 'First_Order'
                   'ARP_Theory'
                   % 'General_Norm'
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
                    ... verbosity = 1, ...
                    inner_solver = 'AR2', ...
                    inner_update_type = inner_update_type{1}, ...
                    inner_stop_rule = inner_stop_rule{1}, ...
                    inner_stop_tolerance_g = 1e-9, ...
                    inner_inner_solver = 'GLRT', ...
                    inner_inner_stop_rule = 'ARP_Theory', ...
                    inner_inner_stop_tolerance_g = 1e-10);
elseif p{1} == 2
    params = struct( ...
                    p = 2, ...
                    problem = problem, ...
                    x0_type = x0_type{1}, ...
                    update_sigma0 = update_sigma0{1}, ...
                    update_type = update_type{1}, ...
                    stop_rule = 'First_Order', ...
                    stop_tolerance_g = 1e-8, ...
                    inner_solver = 'GLRT', ...
                    inner_stop_rule = inner_stop_rule{1}, ...
                    inner_stop_tolerance_g = 1e-9);
    update_use_prerejection = {false};
else
    % No inner_solver needed for AR1
    params = struct( ...
                    p = 1, ...
                    problem = problem, ...
                    x0_type = x0_type{1}, ...
                    update_sigma0 = update_sigma0{1}, ...
                    update_type = update_type{1}, ...
                    stop_rule = 'First_Order', ...
                    stop_tolerance_g = 1e-8);
    update_use_prerejection = {false};
end

% -------------------------------------------------------------------------
% Solver / problem mode logic
% - Detect GLRT vs MCMR for p=2 or p=3.
% - For JSON problems, auto-switch certain names to *_matfree when using GLRT.
% - Forbid MCMR with *_matfree variants (requires dense Hessian).
% - For MGH (numeric) problems, always report DENSE mode.
% -------------------------------------------------------------------------

% Detect solver configuration (shared for both MGH and non-MGH)
solver_desc = "";
is_glrt     = false;
is_mcmr     = false;

if params.p == 2
    if isfield(params, "inner_solver")
        if strcmp(params.inner_solver, "GLRT")
            solver_desc = "AR2 inner solver: GLRT";
            is_glrt = true;
        elseif strcmp(params.inner_solver, "MCMR")
            solver_desc = "AR2 inner solver: MCMR";
            is_mcmr = true;
        else
            solver_desc = "AR2 inner solver: " + string(params.inner_solver);
        end
    end
elseif params.p == 3
    if isfield(params, "inner_solver") && strcmp(params.inner_solver, "AR2") && ...
            isfield(params, "inner_inner_solver")
        if strcmp(params.inner_inner_solver, "GLRT")
            solver_desc = "AR3 inner-inner solver (AR2): GLRT";
            is_glrt = true;
        elseif strcmp(params.inner_inner_solver, "MCMR")
            solver_desc = "AR3 inner-inner solver (AR2): MCMR";
            is_mcmr = true;
        else
            solver_desc = "AR3 inner-inner solver (AR2): " + ...
                string(params.inner_inner_solver);
        end
    end
end

if isnumeric(problems{1})
    % ---- MGH problems: always dense, no *_matfree variants ---------------
    if solver_desc ~= ""
        fprintf("Solver configuration: %s. MGH problem %d will be run in DENSE mode.\n", ...
                solver_desc, problems{1});
    else
        fprintf("MGH problem %d will be run in DENSE mode (solver not GLRT/MCMR-specific).\n", ...
                problems{1});
    end
else
    % ---- Non-MGH problems: JSON-encoded ---------------------------------
    obj  = jsondecode(params.problem);
    name = string(obj.name);

    % Strip "_matfree" if present to get the base problem name
    is_matfree_name = endsWith(name, "_matfree");
    base_name       = name;
    if is_matfree_name
        base_name = erase(name, "_matfree");
    end

    % Only these 6 problems have *_matfree variants
    base_candidates = ["rosenbrock", ...
                       "chebysv_rosenbrock", ...
                       "nonlinear_least_squares", ...
                       "ill_cond_bm", ...
                       "ill_cond_H", ...
                       "ill_cond_T", ...
                       "ill_cond_HT"];
    is_supported_problem = any(base_name == base_candidates);

    % Forbid MCMR with *_matfree problems
    if is_supported_problem && is_matfree_name && is_mcmr
        error("Inconsistent configuration: problem '%s' is a '_matfree' variant " + ...
              "but the solver uses MCMR, which requires a dense Hessian.", name);
    end

    % If GLRT is used on a supported problem and it's not yet *_matfree,
    % automatically add the suffix.
    if is_glrt && is_supported_problem && ~is_matfree_name
        obj.name        = char(base_name + "_matfree");
        params.problem  = jsonencode(obj);
        name            = string(obj.name);
        is_matfree_name = true;
    end

    % Only the 6 problems *with* "_matfree" are treated as matrix-free.
    is_matrix_free = is_supported_problem && is_matfree_name;

    if solver_desc ~= ""
        if is_matrix_free
            fprintf("Solver configuration: %s. Problem '%s' will be run in MATRIX-FREE mode.\n", ...
                    solver_desc, name);
        else
            fprintf("Solver configuration: %s. Problem '%s' will be run in DENSE mode.\n", ...
                    solver_desc, name);
        end
    end
end
% -------------------------------------------------------------------------

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
