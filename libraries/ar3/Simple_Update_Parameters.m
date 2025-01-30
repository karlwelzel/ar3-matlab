classdef Simple_Update_Parameters < Sigma_Update_Parameters
    % Stores parameters of the simple sigma update

    properties
        successful_cutoff (1, 1) double {mustBePositive, lt(successful_cutoff, 1)} = 0.01 % eta_1
        very_successful_cutoff (1, 1) double {mustBePositive, lt(very_successful_cutoff, 1)} = 0.95 % eta_2
        sigma_decrease (1, 1) double {mustBePositive, lt(sigma_decrease, 1)} = 0.5 % gamma_1
        sigma_increase (1, 1) double {gt(sigma_increase, 1)} = 3 % gamma_2
    end

    methods

        function sigma_update_loop(obj, run)
            % Perform the sigma update loop

            arguments (Input)
                obj
                % The given parameters object

                run Optimization_Run
                % An object that stores all information about the current run
            end

            % Initialize values
            sigma = max(obj.estimate_sigma0(run), obj.sigma_min);

            while true
                terminate = run.construct_and_solve_subproblem(sigma);
                if terminate
                    break
                end

                % Decide whether run.step is directionally transient
                if obj.use_prerejection
                    [taylor_poly, model_poly] = obj.construct_polynomials(run, sigma);
                    if run.subproblem_status == Optimization_Status.NOT_LOWER_BOUNDED
                        transient = true;
                        persistent_alpha = 0;
                    else
                        tolerance = max(polyval(polyder(model_poly), 1), 0);
                        persistent_alpha = analyze_persistent_min(taylor_poly, tolerance);
                        transient = 1 > persistent_alpha + 10 * eps;
                    end
                    run.current_history_row.persistent_alpha = persistent_alpha;
                end

                % Compute predicted decrease
                predicted_decrease = obj.compute_predicted_decrease(run, sigma);
                run.current_history_row.rho_den = predicted_decrease;

                % Always compute function value / actual decrease for debugging
                actual_decrease = obj.compute_actual_decrease(run);
                run.current_history_row.rho_num = actual_decrease;

                if predicted_decrease / max(1, abs(run.f)) < eps
                    % Large cancellation error in calculating decrease_ratio
                    if obj.assume_decrease
                        % Assume that there is sufficient function decrease
                        decrease_ratio = 1;
                        % Record that the accepted step may contain numerical error in f
                        run.current_history_row.rho_num = nan;
                    else
                        % Terminate algorithm
                        run.status = Optimization_Status.ILL_CONDITIONED;
                        break
                    end
                elseif obj.use_prerejection && transient
                    decrease_ratio = nan;

                    % We did not use the function eval, so it is not counted
                    run.total_function_evals = run.total_function_evals - 1;
                else
                    decrease_ratio = actual_decrease / predicted_decrease;
                end

                % Update sigma depending on the value of rho (decrease_ratio)
                if decrease_ratio >= obj.very_successful_cutoff
                    % Very successful iteration
                    sigma = obj.sigma_decrease * sigma;
                elseif decrease_ratio >= obj.successful_cutoff
                    % Successful iteration
                    sigma = sigma;
                else
                    % Unsucessful iteration (decrease_ratio too small or nan)
                    run.step = nan; % Don't update iterate
                    sigma = obj.sigma_increase * sigma;
                end
                sigma = max(sigma, obj.sigma_min);

                run.current_history_row.decrease_ratio = decrease_ratio;
                run.log_metrics();
            end
        end

    end
end
