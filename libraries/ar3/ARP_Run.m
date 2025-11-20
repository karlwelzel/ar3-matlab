classdef ARP_Run < Optimization_Run
    % Executes one run of the ARp algorithm

    % This class store various intermediate values, constructs the history
    % struct and controls the interplay between sigma update, subproblem solve
    % and termination condition.

    properties
        f (1, 1) double = nan
        g (:, 1) double = nan
        H = nan
        T = nan
        step (:, 1) double = nan
        norm_step (1, 1) double = nan
        norm_g (1, 1) double = nan
        sigma (1, 1) double = nan
        start_time (1, 1) uint64 = nan
        total_function_evals (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
        total_derivative_evals (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
        total_model_evals (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
        total_model_derivative_evals (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
        total_chol (1, 1) double {mustBeInteger, mustBeInteger} = 0
        total_hvp (1, 1) double {mustBeInteger, mustBeInteger} = 0
        status (1, 1) Optimization_Status = Optimization_Status.RUNNING
        subproblem_status (1, 1) Optimization_Status = Optimization_Status.RUNNING
    end

    methods

        function obj = ARP_Run(f_handle, x0, parameters, optional)
            obj.f_handle = f_handle;
            obj.x = x0;
            obj.parameters = parameters;
            obj.optional = optional;

            obj.default_history_row = struct(f = nan, ...
                                             norm_g = nan, ...
                                             norm_step = nan, ...
                                             total_fun = nan, ...
                                             total_der = nan, ...
                                             total_solves = nan, ...
                                             total_model = nan, ...
                                             total_model_der = nan, ...
                                             total_chol = nan, ...
                                             total_hvp = nan, ...
                                             time = nan, ...
                                             sigma = nan, ...
                                             decrease_ratio = nan, ...
                                             rho_num = nan, ...
                                             rho_den = nan, ...
                                             persistent_alpha = nan, ...
                                             interp_fallback_ratio = nan, ...
                                             sub_status = nan);
            obj.history = obj.default_history_row;
            obj.current_history_row = obj.default_history_row;

            % Set up metric collection with monitor and wandb
            if isfield(optional, "monitor")
                optional.monitor.Metrics = ["f" "norm_g"];
            end

            if isfield(optional, "wandb")
                assert(optional.wandb.run ~= py.None, "wandb.init() was not called before ar3");
            end
        end

        function log_metrics(obj)
            if obj.parameters.verbosity > 0
                disp("ARp with p = " + obj.parameters.p);
                disp(struct2table(obj.current_history_row));
            end

            log_metrics@Optimization_Run(obj);
        end

        function [f_value] = evaluate_function(obj, x_plus)
            f_value = obj.f_handle(x_plus);
            obj.total_function_evals = obj.total_function_evals + 1;
        end

        function update_derivatives(obj, order, sigma)
            % Ensure that the derivatives are calculated in the first iteration
            if obj.iteration == 1
                obj.step = zeros(length(obj.x), 1);
                obj.total_function_evals = obj.total_function_evals + 1;
            end

            if ~(isscalar(obj.step) && isnan(obj.step))
                % Update current iterate
                obj.x = obj.x + obj.step;

                % Update derivatives
                if order == 1
                    [obj.f, obj.g] = obj.f_handle(obj.x);
                elseif order == 2
                    [obj.f, obj.g, obj.H] = obj.f_handle(obj.x);
                elseif order == 3
                    [obj.f, obj.g, obj.H, obj.T] = obj.f_handle(obj.x);
                end

                % Update counters
                obj.norm_g = norm(obj.g);
                obj.sigma = sigma;
                obj.total_derivative_evals = obj.total_derivative_evals + 1;
            end

            % Track history
            obj.current_history_row.f = obj.f;
            obj.current_history_row.norm_g = obj.norm_g;
            obj.current_history_row.total_fun = obj.total_function_evals;
            obj.current_history_row.total_der = obj.total_derivative_evals;
            obj.current_history_row.total_solves = obj.iteration - 1;
            obj.current_history_row.total_model = obj.total_model_evals;
            obj.current_history_row.total_model_der = obj.total_model_derivative_evals;
            obj.current_history_row.total_chol = obj.total_chol;
            obj.current_history_row.total_hvp = obj.total_hvp;
            obj.current_history_row.time = toc(obj.start_time);
        end

        function solve_suproblem(obj, sigma)
            % Solve the subproblem with the chosen solver
            subproblem_parameters = obj.parameters.subproblem_parameters;
            subproblem_parameters.termination_rule.outer_run = obj;

            if obj.parameters.p == 1
                obj.step = -obj.g / sigma;
            elseif obj.parameters.p == 2
                if class(subproblem_parameters) == "MCMR_Parameters" % requires full matrix H
                    [obj.subproblem_status, obj.step, num_iterations] = ...
                        subproblem_parameters.run(obj.f, obj.g, obj.H, sigma, 3);
                    obj.total_model_evals = obj.total_model_evals + num_iterations;
                    obj.total_model_derivative_evals = obj.total_model_derivative_evals + num_iterations;
                    obj.total_chol = obj.total_chol + num_iterations;
                    obj.total_hvp = 0;
                elseif class(subproblem_parameters) == "GLRT_Parameters"
                    [obj.subproblem_status, obj.step, num_iterations] = ...
                        subproblem_parameters.run(obj.f, obj.g, obj.H, sigma);
                    obj.total_model_evals = obj.total_model_evals + num_iterations;
                    obj.total_model_derivative_evals = obj.total_model_derivative_evals + num_iterations;
                    obj.total_chol = 0;
                    obj.total_hvp = obj.total_hvp + num_iterations;
                elseif class(subproblem_parameters) == "Fminunc_Parameters"
                    model_handle = @(s) ar2_model_derivatives(s, 0, obj.g, obj.H, sigma);
                    [obj.subproblem_status, obj.step, sub_history] = ...
                        subproblem_parameters.run(model_handle, zeros(length(obj.x), 1));
                    obj.total_model_evals = obj.total_model_evals + sub_history(end).total_fun;
                    obj.total_model_derivative_evals = obj.total_model_derivative_evals + sub_history(end).total_der;
                    obj.total_chol = 0;
                    obj.total_hvp = 0;
                else
                    error("Invalid subproblem solver: " + class(obj.parameters.subproblem_parameters));
                end
            elseif obj.parameters.p == 3
                model_handle = @(s) ar3_model_derivatives(s, 0, obj.g, obj.H, obj.T, sigma);
                if class(subproblem_parameters) == "ARP_Parameters" && subproblem_parameters.p == 2
                    sigma_min = subproblem_parameters.sigma_update_parameters.sigma_min;
                    subproblem_parameters.sigma_update_parameters.sigma0 = sigma_min;
                    % subproblem_parameters.sigma_update_parameters.sigma0 = ...
                    %   max((3 / 2) * norm(obj.T, "fro"), sigma_min);
                    [obj.subproblem_status, obj.step, sub_history] = ...
                        subproblem_parameters.run(model_handle, zeros(length(obj.x), 1));
                elseif class(subproblem_parameters) == "QQR_Parameters"
                    [obj.subproblem_status, obj.step, sub_history] = ...
                        subproblem_parameters.run(model_handle, sigma, zeros(length(obj.x), 1));
                elseif class(subproblem_parameters) == "Fminunc_Parameters"
                    [obj.subproblem_status, obj.step, sub_history] = ...
                        subproblem_parameters.run(model_handle, zeros(length(obj.x), 1));
                else
                    error("Invalid subproblem solver: " + class(obj.parameters.subproblem_parameters));
                end

                obj.total_model_evals = obj.total_model_evals + sub_history(end).total_fun;
                obj.total_model_derivative_evals = obj.total_model_derivative_evals + sub_history(end).total_der;
                if isa(obj.H, "function_handle") && isfield(sub_history, "total_hvp")
                    obj.total_hvp = obj.total_hvp + sub_history(end).total_fun + sub_history(end).total_hvp;
                else
                    obj.total_hvp = 0;
                end
                if ~isa(obj.H, "function_handle") && isfield(sub_history, "total_chol")
                    obj.total_chol = obj.total_chol + sub_history(end).total_chol;
                else
                    obj.total_chol = 0;
                end
            end

            obj.norm_step = norm(obj.step);
            obj.current_history_row.norm_step = obj.norm_step;

            obj.current_history_row.sub_status = int32(obj.subproblem_status);

            % Issue a warning if the subproblem solver failed
            % if obj.subproblem_status ~= Optimization_Status.SUCCESS
            %     warning("The subproblem solver " + class(subproblem_parameters) + ...
            %             " failed with status " + string(obj.subproblem_status) + ...
            %             " for sigma = " + sigma);
            % end
        end

        function [terminate] = should_terminate(obj)
            [terminate, obj.status] = obj.parameters.termination_rule.should_terminate(obj);
        end

        function [terminate] = construct_and_solve_subproblem(obj, sigma)
            obj.update_derivatives(obj.parameters.p, sigma);
            terminate = obj.should_terminate();
            if ~terminate
                obj.solve_suproblem(sigma);

                if norm(obj.step ./ max(1, obj.x), "inf") < 10 * eps
                    terminate = true;
                    obj.status = Optimization_Status.NUMERICAL_ISSUES;
                end
            end

            obj.current_history_row.sigma = sigma;
        end

        function run(obj)
            obj.start_time = tic;
            obj.parameters.sigma_update_parameters.sigma_update_loop(obj);
            obj.log_metrics();
        end

    end
end
