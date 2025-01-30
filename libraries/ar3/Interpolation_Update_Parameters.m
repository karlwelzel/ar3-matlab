classdef Interpolation_Update_Parameters < Simple_Update_Parameters
    % Stores parameters of the Interpolation sigma update

    % The class inherits from Simple_Update_Parameters and so supports all
    % parameters of the simple update and some additional ones.

    properties
        sigma_large_decrease (1, 1) double {mustBePositive, lt(sigma_large_decrease, 1)} = 1 / 10 % gamma_3
        interp_reduction (1, 1) double {mustBePositive, lt(interp_reduction, 1)} = 1 / 100 % beta
        interp_alpha_max (1, 1) double {gt(interp_alpha_max, 1)} = 2 % alpha_max
        sigma_max_increase (1, 1) double {gt(sigma_max_increase, 1)} = 100 % gamma_max
        min_success_diff (1, 1) double {mustBePositive} = 1e-8 % epsilon_chi
    end

    methods

        function [sigma, alpha] = optimize_sigma(obj, taylor_poly, prev_sigma, norm_step, ...
                                                 constraint_poly, decrease_sigma)
            % Finds the best sigma s.t. one local min of m satisfies a constraint
            %
            % If decrease_sigma is true the function tries to find the largest
            % sigma smaller than or equal to prev_sigma such that
            %     polyval(constraint_poly, alpha) <= 0
            % holds for a local minimizer alpha of the corresponding model.
            % Otherwise, the function computes the smallest sigma larger than
            % or equal to prev_sigma which satisfies the inequality constraint.
            % If there is no such sigma the function returns NaN.
            % If obj.reject_transient is true, only consider persistent minimizers.

            p = length(taylor_poly) - 1;
            taylor_der_poly = padded_polyder(taylor_poly);
            local_min_poly = [-padded_polyder(taylor_der_poly), 0] + p * taylor_der_poly;
            prev_sigma_poly = (-1)^decrease_sigma * [prev_sigma * norm_step^(p + 1), taylor_der_poly];
            compute_sigma = @(alpha) -polyval(taylor_der_poly, alpha) ./ (norm_step^(p + 1) * alpha.^p);

            % Step 1: Compute all alpha on the boundary of the feasible set
            if obj.use_prerejection
                alpha_persistent = analyze_persistent_min(taylor_poly);
                if alpha_persistent < inf
                    constraints = {constraint_poly, prev_sigma_poly, [1, -alpha_persistent]};
                else
                    constraints = {constraint_poly, prev_sigma_poly};
                end
            else
                constraints = {constraint_poly, prev_sigma_poly, taylor_der_poly, local_min_poly};
            end
            alpha_options = pos_boundary_points(constraints);

            % Step 2: Select the best alpha and sigma
            if ~isempty(alpha_options)
                if decrease_sigma
                    [sigma, index] = max(compute_sigma(alpha_options));
                else
                    [sigma, index] = min(compute_sigma(alpha_options));
                end
                sigma = max(sigma, 0);
                alpha = alpha_options(index);
            else
                sigma = nan;
                alpha = nan;
            end
        end

        function [fallback_ratio, sigma] = unsuccessful_sigma(obj, taylor_poly, prev_sigma, norm_step, interp_poly)
            p = length(taylor_poly) - 1;
            eta = obj.successful_cutoff;

            % As a proxy for the real decrease, measure the actual decrease
            % using the interpolation polynomial
            actual_decrease_poly = [-interp_poly(1:end - 1), 0];

            if obj.decrease_measure == Decrease_Measure.TAYLOR
                % Measure the predicted decrease using the Taylor polynomial
                predicted_decrease_poly = [0, -taylor_poly(1:end - 1), 0];
            elseif obj.decrease_measure == Decrease_Measure.MODEL
                % Measure the predicted decrease using the ARP model
                % To get rid of sigma we use that the derivative of the model
                % at the relevant alphas is always zero to express sigma as a
                % function of alpha.
                taylor_der_poly = padded_polyder(taylor_poly);
                predicted_decrease_poly = [0, -taylor_poly(1:end - 1) + (1 / (p + 1)) * taylor_der_poly, 0];
            end

            constraint_poly = actual_decrease_poly - eta * predicted_decrease_poly;

            % The constant term is zero and the extra factor of alpha cancels
            % out in rho, so we get rid of it
            constraint_poly = constraint_poly(1:end - 1);

            [sigma, ~] = obj.optimize_sigma(taylor_poly, prev_sigma, norm_step, -constraint_poly, false);

            % Safeguard sigma using minimum and maximum increase
            if isnan(sigma) || sigma < obj.sigma_increase * prev_sigma
                sigma = obj.sigma_increase * prev_sigma;
                fallback_ratio = obj.sigma_increase;
            elseif sigma > obj.sigma_max_increase * prev_sigma
                sigma = obj.sigma_max_increase * prev_sigma;
                fallback_ratio = obj.sigma_max_increase;
            else
                % Use result from optimize_sigma
                fallback_ratio = -1;
            end
        end

        function [fallback_ratio, sigma] = successful_sigma(obj, taylor_poly, prev_sigma, norm_step, interp_poly)
            p = length(taylor_poly) - 1;
            beta = obj.interp_reduction;

            % Make the model independent of sigma by using that the model
            % derivative at the relevant alphas is zero
            taylor_der_poly = padded_polyder(taylor_poly);
            model_poly = [0, taylor_poly] - [0, (1 / (p + 1)) * taylor_der_poly, 0];

            if polyval(interp_poly, 1) >= polyval(taylor_poly, 1)
                diff_poly = model_poly - interp_poly;
            else
                diff_poly = model_poly - [0, taylor_poly];
            end
            current_diff = polyval(diff_poly, 1);

            % Safeguard sigma when diff is too small or alpha too large
            if current_diff >= obj.min_success_diff
                constraint_poly = diff_poly - [zeros(1, p + 1), beta * current_diff];
                [sigma, alpha] = obj.optimize_sigma(taylor_poly, prev_sigma, norm_step, constraint_poly, true);
                if isnan(alpha) || alpha > obj.interp_alpha_max
                    sigma = obj.sigma_large_decrease * prev_sigma;
                    fallback_ratio = obj.sigma_large_decrease;
                else
                    % Use result from optimize_sigma
                    fallback_ratio = -1;
                end
            else
                sigma = obj.sigma_decrease * prev_sigma;
                fallback_ratio = obj.sigma_decrease;
            end
        end

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

                % Compute predicted decrease
                [taylor_poly, model_poly] = obj.construct_polynomials(run, sigma);
                taylor_plus = polyval(taylor_poly, 1);
                model_plus = polyval(model_poly, 1);

                if obj.decrease_measure == Decrease_Measure.TAYLOR
                    predicted_decrease = run.f - taylor_plus;
                elseif obj.decrease_measure == Decrease_Measure.MODEL
                    predicted_decrease = run.f - model_plus;
                end
                run.current_history_row.rho_den = predicted_decrease;

                if predicted_decrease < -eps
                    error("The predicted decrease is negative");
                end

                % Always compute function value / actual decrease for debugging
                actual_decrease = obj.compute_actual_decrease(run);
                run.current_history_row.rho_num = actual_decrease;

                % Decide whether run.step is directionally transient
                if obj.use_prerejection
                    if run.subproblem_status == Optimization_Status.NOT_LOWER_BOUNDED
                        transient = true;
                        persistent_alpha = 0;
                    else
                        tolerance = max(polyval(polyder(model_poly), 1), 0);
                        persistent_alpha = ...
                            analyze_persistent_min(taylor_poly, tolerance);
                        transient = 1 > persistent_alpha + 10 * eps;
                    end
                    run.current_history_row.persistent_alpha = persistent_alpha;
                end

                if predicted_decrease / max(1, abs(run.f)) < eps
                    % Large cancellation error in calculating decrease_ratio
                    if obj.assume_decrease
                        % Assume that there is sufficient function decrease
                        decrease_ratio = 1;
                        % Record accepted steps that may contain numerical error in f
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
                    f_plus = run.f - actual_decrease;
                    interp_poly = [f_plus - taylor_plus, taylor_poly];
                    decrease_ratio = actual_decrease / predicted_decrease;
                end

                % Update sigma depending on the value of rho (decrease_ratio)
                if decrease_ratio >= 1 + eps
                    % Extremely successful iteration
                    [fallback_ratio, sigma] = obj.successful_sigma(taylor_poly, sigma, run.norm_step, interp_poly);
                    run.current_history_row.interp_fallback_ratio = fallback_ratio;
                elseif decrease_ratio >= obj.very_successful_cutoff
                    % Very successful iteration
                    sigma = obj.sigma_decrease * sigma;
                elseif decrease_ratio >= obj.successful_cutoff
                    % Successful iteration
                    sigma = sigma;
                elseif decrease_ratio >= -eps || isnan(decrease_ratio)
                    % Unsucessful iteration
                    run.step = nan; % Don't update iterate
                    sigma = obj.sigma_increase * sigma;
                elseif decrease_ratio > -inf
                    % Extremely unsuccessful iteration
                    run.step = nan; % Don't update iterate
                    [fallback_ratio, sigma] = obj.unsuccessful_sigma(taylor_poly, sigma, run.norm_step, interp_poly);
                    run.current_history_row.interp_fallback_ratio = fallback_ratio;
                else % decrease_ratio == -inf
                    % Something has gone wrong, but we do see this sometimes
                    run.step = nan; % Don't update iterate
                    sigma = obj.sigma_max_increase * sigma;
                end
                sigma = max(sigma, obj.sigma_min);

                run.current_history_row.decrease_ratio = decrease_ratio;
                run.log_metrics();
            end
        end

    end
end

function boundary = pos_boundary_points(poly_constraints)
    % Computes the all positive points on the boundary of a feasible set
    % defined by the given polynomial constraints. They are all assumed to be
    % in the form
    %     polyval(poly_constraints(i), x) <= 0

    num_constraints = length(poly_constraints);

    boundary = [];
    for i = 1:num_constraints
        points = real_roots(poly_constraints{i});
        points = points(points > 0);
        for j = [1:(i - 1), (i + 1):num_constraints]
            points = points(polyval(poly_constraints{j}, points) <= 10 * eps);
        end
        boundary = [boundary; points];
    end
end
