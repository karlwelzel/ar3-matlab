classdef BGMS_Update_Parameters < Sigma_Update_Parameters
    % Stores parameters of the BGMS sigma update

    properties
        max_sigma_iterations (1, 1) {mustBeInteger, mustBePositive} = 20 % J
        max_relative_decrease (1, 1) double {mustBePositive} = 1e3 % eta_1
        max_relative_step_norm (1, 1) double {mustBePositive} = 3 % eta_2
        sigma_decrease (1, 1) double {mustBePositive, lt(sigma_decrease, 1)} = 0.5 % gamma_1
        sigma_increase (1, 1) double {gt(sigma_increase, 1)} = 10 % gamma_2
        alpha (1, 1) double {mustBePositive} = 1e-8
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

            sigma_ini = max(obj.estimate_sigma0(run), obj.sigma_min);

            while true

                % Step 1 of Algorithm 4.1
                j = 0;
                sigma = 0;

                while true
                    % Step 2 of Algorithm 4.1
                    terminate = run.construct_and_solve_subproblem(sigma);
                    if terminate
                        return
                    end

                    if j == 0 && run.subproblem_status == Optimization_Status.NOT_LOWER_BOUNDED
                        run.step = nan; % Don't update iterate
                        sigma = sigma_ini;
                        j = 1;

                        run.log_metrics();
                        continue
                    end

                    % Step 3 of Algorithm 4.1
                    predicted_decrease = obj.compute_predicted_decrease(run, sigma);
                    run.current_history_row.rho_den = predicted_decrease;

                    % Always compute function value for debugging
                    actual_decrease = obj.compute_actual_decrease(run);
                    run.current_history_row.rho_num = actual_decrease;

                    relative_decrease = predicted_decrease / max(1, abs(run.f));
                    relative_step_norm = norm(run.step, "inf") / max(1, norm(run.x, "inf"));

                    run.current_history_row.decrease_ratio = relative_decrease;

                    if ~obj.use_prerejection || j >= obj.max_sigma_iterations || ...
                      (relative_decrease <= obj.max_relative_decrease && ...
                       relative_step_norm <= obj.max_relative_step_norm)
                        % Step 4 of Algorithm 4.1
                        if actual_decrease >= obj.alpha * run.norm_step^(run.parameters.p + 1) || ...
                          (relative_decrease < eps && obj.assume_decrease)
                            % Step 5 of Algorithm 4.1
                            if sigma == 0
                                sigma_ini = max(obj.sigma_decrease * sigma_ini, obj.sigma_min);
                            else
                                sigma_ini = max(obj.sigma_decrease * sigma, obj.sigma_min);
                            end
                            run.log_metrics();
                            break % end j-loop
                        else
                            sigma = max(sigma_ini, obj.sigma_increase * sigma);
                            run.step = nan; % Don't update iterate
                        end
                    else
                        % Step 2 of Algorithm 4.1
                        sigma = max(sigma_ini, obj.sigma_increase * sigma);
                        run.step = nan; % Don't update iterate

                        % We did not use the function eval, so it is not counted
                        run.total_function_evals = run.total_function_evals - 1;
                    end
                    run.log_metrics();
                    j = j + 1;
                end
            end
        end

    end
end
