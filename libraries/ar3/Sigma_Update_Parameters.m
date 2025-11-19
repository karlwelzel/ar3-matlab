classdef (Abstract) Sigma_Update_Parameters < Parameters
    % Base class for all sigma update parameter classes

    % This class implements convenience functions like constructing one-
    % dimensional Taylor polynomials in the step subspace or calculating actual
    % and predicted function decreases. Different methods for estimating the
    % initial regularization parameter sigma0 are also implemented here.

    properties
        sigma0 = 1
        sigma_min (1, 1) double {mustBePositive} = 1e-8
        decrease_measure (1, 1) Decrease_Measure = Decrease_Measure.TAYLOR
        use_prerejection (1, 1) logical = true
        assume_decrease (1, 1) logical = false
    end

    methods (Static)

        function obj = from_struct(params)
            if params.type == "Simple"
                obj = Simple_Update_Parameters;
            elseif params.type == "BGMS"
                obj = BGMS_Update_Parameters;
            elseif params.type == "Interpolation"
                obj = Interpolation_Update_Parameters;
            elseif params.type == "Interpolation_m"
                obj = Interpolation_Update_Parameters;
                obj.decrease_measure = Decrease_Measure.MODEL;
            elseif params.type == "Interpolation_t"
                obj = Interpolation_Update_Parameters;
                obj.decrease_measure = Decrease_Measure.TAYLOR;
            else
                error("Unknown sigma update type: " + params.type);
            end
            params = rmfield(params, "type");
            obj = obj.update(params);
        end

        function [taylor_poly, model_poly] = construct_polynomials(run, sigma)
            if run.parameters.p == 1
                taylor_poly = [
                               run.g' * run.step
                               run.f
                              ]';
                model_poly = [(sigma / 2) * run.norm_step^2, taylor_poly];
            elseif run.parameters.p == 2
                taylor_poly = [
                               (1 / 2) * run.step' * mat_vec(run.H, run.step)
                               run.g' * run.step
                               run.f
                              ]';
                model_poly = [(sigma / 3) * run.norm_step^3, taylor_poly];
                % Note that the AR2 model only coincides with this polynomial
                % for nonnegative alpha
            elseif run.parameters.p == 3
                T_s = tensor_vec(run.T, run.step);
                taylor_poly = [
                               (1 / 6) * run.step' * mat_vec(T_s, run.step)
                               (1 / 2) * run.step' * mat_vec(run.H, run.step)
                               run.g' * run.step
                               run.f
                              ]';
                model_poly = [(sigma / 4) * run.norm_step^4, taylor_poly];
            end
        end

    end

    methods

        function [decr] = compute_predicted_decrease(obj, run, sigma)
            if obj.decrease_measure == Decrease_Measure.TAYLOR
                s = 0;
            elseif obj.decrease_measure == Decrease_Measure.MODEL
                s = sigma;
            end

            if run.parameters.p == 1
                decr = -ar1_model_derivatives(run.step, 0, run.g, s);
            elseif run.parameters.p == 2
                decr = -ar2_model_derivatives(run.step, 0, run.g, run.H, s);
            elseif run.parameters.p == 3
                decr = -ar3_model_derivatives(run.step, 0, run.g, run.H, run.T, s);
            end

            if decr < -eps
                error("The predicted decrease is negative");
            end
        end

        function [decr] = compute_actual_decrease(obj, run)
            f_plus = run.evaluate_function(run.x + run.step);
            decr = run.f - f_plus;
        end

        function [sigma0] = estimate_sigma0(obj, run)
            if isnumeric(obj.sigma0)
                sigma0 = obj.sigma0;
            elseif isa(obj.sigma0, "string") && ~isnan(str2double(obj.sigma0))
                sigma0 = str2double(obj.sigma0);
            else
                % Estimate sigma0 using function or derivative evaluations
                sigma0 = obj.sigma_min;
                dim = length(run.x);
                p = run.parameters.p;

                if p == 1
                    [f, der1f] = run.f_handle(run.x);
                    T0 = der1f;
                elseif p == 2
                    [f, der1f, der2f] = run.f_handle(run.x);
                    T0 = der2f;
                elseif p == 3
                    [f, der1f, der2f, der3f] = run.f_handle(run.x);
                    T0 = der3f;
                end

                if obj.sigma0 == "LIPSCHITZ"
                    steps = 3;
                    for i = 1:steps
                        delta_x = randn(dim, 1);
                        % norm(delta_x) ~ sqrt(eps) is optimal for finite differences
                        delta_x = delta_x * sqrt(eps);
                        if p == 1
                            [~, T] = run.f_handle(run.x + delta_x);
                        elseif p == 2
                            [~, ~, T] = run.f_handle(run.x + delta_x);
                        elseif p == 3
                            [~, ~, ~, T] = run.f_handle(run.x + delta_x);
                        end
                        run.total_function_evals = run.total_function_evals + 1;
                        run.total_derivative_evals = run.total_derivative_evals + 1;
                        sigma0 = max(norm(T - T0, 'fro') / norm(delta_x), sigma0);
                    end
                elseif obj.sigma0 == "INVEXIFICATION"
                    steps = 1;
                    delta_x = -der1f;
                    for i = 1:steps
                        norm_x = norm(delta_x);
                        der3f_s = tensor_vec(der3f, delta_x);
                        taylor_der_poly = [
                                           delta_x' * mat_vec(der3f_s, delta_x) / 2
                                           delta_x' * mat_vec(der2f, delta_x)
                                           der1f' * delta_x
                                          ]';
                        sigma_invex = compute_sigma_invex(taylor_der_poly, norm_x);
                        sigma0 = max(sigma0, max(obj.sigma_min, sigma_invex));
                        delta_x = randn(length(der1f), 1);
                        delta_x = -delta_x * sign(delta_x' * der1f);
                    end
                elseif obj.sigma0 == "TAYLOR"
                    steps = 1;
                    for i = 1:steps
                        delta_x = randn(dim, 1);
                        delta_x = delta_x * 10^(-2 * i + 2);

                        f_plus = run.evaluate_function(run.x + delta_x);
                        if abs(f_plus) > (1 / eps) * abs(f)
                            % This function value is unrealistically large, ignore
                            continue
                        end

                        taylor_residual = f_plus - f - der1f' * delta_x;
                        if p >= 2
                            taylor_residual = taylor_residual - (1 / 2) * delta_x' * mat_vec(der2f, delta_x);
                        end
                        if p == 3
                            der3f_s = tensor_vec(der3f, delta_x);
                            taylor_residual = taylor_residual - (1 / 6) * delta_x' * mat_vec(der3f_s, delta_x);
                        end
                        sigma0 = max((p + 1) * abs(taylor_residual) / norm(delta_x)^(p + 1), sigma0);
                    end
                end
            end
        end

    end
end
