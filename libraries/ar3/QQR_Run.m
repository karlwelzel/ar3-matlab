classdef QQR_Run < ARP_Run
    % Executes one run of the QQR algorithm

    % The QQR algorithm is designed to tackle cubic problems with a fourth-
    % order regularization term. See Cartis and Zhu "Second-order methods for
    % quartically-regularised cubic polynomials, with applications to high-
    % order tensor methods" for details.

    properties
        sigma (1, 1) double {mustBeNonnegative} = 1
        hessian_perturbation (1, 1) double {mustBeNonnegative} = 0  % \tilde{p}
        hessian_scaling (1, 1) double {mustBePositive} = 1  % alpha_1
        regularization_scaling (1, 1) double {mustBePositive} = 1  % alpha_2
        scaled_hessian (:, :) double = nan
        scaled_sigma (1, 1) double = nan
    end

    methods

        function obj = QQR_Run(f_handle, sigma, x0, parameters, optional)
            obj@ARP_Run(f_handle, x0, parameters, optional);
            obj.sigma = sigma;

            % Overwrite ARP_Run's history columns
            obj.default_history_row = struct(f = nan, ...
                                             norm_g = nan, ...
                                             norm_step = nan, ...
                                             total_fun = nan, ...
                                             total_der = nan, ...
                                             total_model = nan, ...
                                             total_model_der = nan, ...
                                             decrease_ratio = nan, ...
                                             der_evals_ratio = nan, ...
                                             model_evals_ratio = nan, ...
                                             time = nan, ...
                                             hessian_perturbation = nan, ...
                                             hessian_scaling = nan, ...
                                             regularization_scaling = nan, ...
                                             sigma = nan, ...
                                             model_plus = nan);
            obj.history = obj.default_history_row;
            obj.current_history_row = obj.default_history_row;
        end

        function log_metrics(obj)
            if obj.parameters.verbosity > 0
                disp("QQR");
                disp(struct2table(obj.current_history_row));
            end

            log_metrics@Optimization_Run(obj);
        end

        function solve_suproblem(obj, ~)
            % Solve the subproblem with the chosen solver
            subproblem_parameters = obj.parameters.subproblem_parameters;
            subproblem_parameters.termination_rule.outer_run = obj;

            if class(subproblem_parameters) == "MCMR_Parameters"
                model_handle = @(s) mcmr_model(s, obj.f, obj.g, obj.scaled_hessian, obj.scaled_sigma, 4);

                [sub_status, obj.step, num_iterations] = ...
                  subproblem_parameters.run(obj.f, obj.g, obj.scaled_hessian, obj.scaled_sigma, 4);

                obj.total_model_evals = obj.total_model_evals + num_iterations;
                obj.total_model_derivative_evals = obj.total_model_derivative_evals + num_iterations;
            else
                error("Invalid subproblem solver: " + class(obj.parameters.subproblem_parameters));
            end

            obj.norm_step = norm(obj.step);
            obj.current_history_row.norm_step = obj.norm_step;

            obj.current_history_row.model_plus = model_handle(obj.step);
            obj.current_history_row.sigma = obj.scaled_sigma;
            obj.current_history_row.hessian_perturbation = obj.hessian_perturbation;
            obj.current_history_row.hessian_scaling = obj.hessian_scaling;
            obj.current_history_row.regularization_scaling = obj.regularization_scaling;

            % Issue a warning if the subproblem solver failed
            if sub_status ~= Optimization_Status.SUCCESS
                warning("The subproblem solver " + class(obj.parameters.subproblem_parameters) + ...
                        "failed with status " + string(sub_status) + " for sigma = " + obj.scaled_sigma);
            end
        end

        function run(obj)
            while true
                obj.update_derivatives(2); % Evaluate gradient and Hessian
                if obj.should_terminate()
                    break
                end

                obj.scaled_hessian = obj.hessian_scaling * (obj.H + obj.hessian_perturbation * eye(size(obj.H)));
                obj.scaled_sigma = obj.regularization_scaling * obj.sigma;

                if (obj.norm_g / obj.scaled_sigma)^(1 / 3) < eps
                    % If sigma is this large, the corresponding step will
                    % necessarily be too small to make an impact
                    obj.status = Optimization_Status.NUMERICAL_ISSUES;
                end

                obj.solve_suproblem();

                if norm(obj.step ./ max(1, obj.x), "inf") < 10 * eps
                    obj.status = Optimization_Status.NUMERICAL_ISSUES;
                    break
                end

                f_plus = obj.evaluate_function(obj.x + obj.step);
                predicted_decrease = -mcmr_model(obj.step, 0, obj.g, obj.scaled_hessian, obj.scaled_sigma, 4);
                actual_decrease = obj.f - f_plus;

                if abs(predicted_decrease) > eps
                    decrease_ratio = actual_decrease / predicted_decrease;
                else
                    decrease_ratio = 1;
                end
                obj.current_history_row.decrease_ratio = decrease_ratio;

                if decrease_ratio >= obj.parameters.successful_cutoff
                    % Successful iteration, reset scalings
                    obj.hessian_perturbation = 0;
                    obj.hessian_scaling = 1;
                    obj.regularization_scaling = 1;
                else
                    % Unsucessful iteration
                    lambda = (obj.step' * obj.H * obj.step) / obj.norm_step^2;
                    obj.step = nan; % Don't update iterate

                    if lambda > obj.parameters.lambda_c
                        obj.hessian_scaling = obj.parameters.hessian_increase * obj.hessian_scaling;
                    elseif lambda >= -obj.parameters.lambda_c
                        obj.hessian_perturbation = obj.parameters.hessian_perturbation;
                    else
                        obj.hessian_scaling = obj.parameters.hessian_increase^(-1) * obj.hessian_scaling;
                    end

                    obj.regularization_scaling = obj.parameters.regularization_increase * obj.regularization_scaling;
                end
                obj.log_metrics();
            end

            obj.log_metrics();
        end

    end

end

function m = mcmr_model(s, f, der1f, der2f, sigma, r)
    m = f + der1f' * s + (1 / 2) * s' * der2f * s + (sigma / r) * norm(s)^r;
end
