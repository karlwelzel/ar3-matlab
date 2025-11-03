classdef MCMR_Parameters < Optimization_Parameters
    % Stores parameters for MCMR and runs the algorithm

    % For a description of MCM see Cartis et al. "Adaptive cubic regularisation
    % methods for unconstrained optimization. Part I: motivation, convergence
    % and numerical results" Algorithm 6.1
    % The MCMR algorithm is an extension of this algorithm to problems of type
    %     f(x) = c' * x + (1/2) * x' * mat * x + (sigma/r) * norm(x)^r
    % for r > 2 and behaves like MCM when r = 3.

    properties
        termination_rule (1, 1) Termination_Rule = Cartis_G_Termination_Rule
    end

    methods (Static)

        function obj = from_struct(params)
            obj = MCMR_Parameters;

            if isfield(params, "verbosity")
                obj.verbosity = params.verbosity;
                params = rmfield(params, "verbosity");
            end

            if isfield(params, "stop_rule")
                [termination_params, params] = extract_params(params, "stop_");
                obj.termination_rule = Termination_Rule.from_struct(termination_params);
            end

            remaining_fieldnames = fieldnames(params);
            if ~isempty(remaining_fieldnames)
                error("Unrecognized field: " + remaining_fieldnames{1});
            end
        end

    end

    methods

        function [status, best_x, iteration] = run(obj, ~, c, mat, sigma, r)
            residual = @(x) c + mat * x;
            n = length(c);
            identity = eye(n, n);
            norm_mat = norm(mat);
            iteration = 0;

            lambda_max = inf;
            lambda_min = eps;

            mat_lambda = mat + lambda_min * identity;
            [R, indefinite] = chol(mat_lambda);

            if sigma < eps
                if indefinite
                    status = Optimization_Status.NOT_LOWER_BOUNDED;
                    best_x = nan(n, 1);
                else
                    status = Optimization_Status.SUCCESS;
                    best_x = -R \ (R' \ c);
                end
                return
            end

            if indefinite
                try
                    if n >= 100
                        min_eigval = eigs(mat_lambda, 1, 'smallestreal');
                    else
                        min_eigval = min(eig(mat_lambda));
                    end
                    lambda_min = max(-min_eigval, 0) + eps;
                catch
                    status = Optimization_Status.ILL_CONDITIONED;
                    best_x = zeros(n, 1);
                    return
                end

                % Find lower bound for initial lambda
                lambda_definite = max(lambda_min, norm_mat * eps);
                mat_lambda = mat + lambda_definite * identity;
                [~, indefinite] = chol(mat_lambda);
                while indefinite
                    lambda_definite = (1 + 1e-5) * lambda_definite;
                    mat_lambda = mat + lambda_definite * identity;
                    [~, indefinite] = chol(mat_lambda);
                end
            else
                lambda_definite = lambda_min;
            end

            % Set initial lambda
            if norm_mat < eps
                lambda = lambda_definite;
            else
                lambda = max(sigma * (norm(c) / norm_mat)^(r - 2), lambda_definite);
            end

            if obj.verbosity >= 1
                fprintf(' %+4s %+23s %+19s %+19s %+19s\n', "it", "lambda", "norm_x", "norm_g", "phi");
            end

            hard_case = true;
            best_norm_g = inf;

            while true
                iteration = iteration + 1;

                if iteration > 2 && ~(lambda_min + eps < lambda && lambda < lambda_max - eps)
                    if hard_case
                        % All evaluations of phi were positive, so we are in the hard case
                        [v1, ~] = eigs(mat_lambda, 1, 'smallestreal');
                        alpha = sqrt((lambda / sigma)^(2 / (r - 2)) - norm_x^2);
                        best_x = x + alpha * v1;
                        status = Optimization_Status.SUCCESS;
                        return
                    else
                        status = Optimization_Status.NUMERICAL_ISSUES;
                        return
                    end
                end

                mat_lambda = mat + lambda * identity;
                [R, indefinite] = chol(mat_lambda);

                if indefinite
                    % lambda is too small
                    lambda_min = max(lambda, lambda_min);

                    % Do bisection in log-space
                    lambda = sqrt(lambda_min * lambda_max);
                    continue
                end

                x = -R \ (R' \ c);
                norm_x = norm(x);
                norm_g = abs((sigma * norm_x^(r - 2) - lambda) * norm_x);
                phi = 1 / norm_x - (sigma / lambda)^(1 / (r - 2));

                if obj.verbosity >= 1
                    fprintf(' %4i %23.16e %19.12e %19.12e %19.12e\n', iteration, lambda, norm_x, norm_g, phi);
                end

                if norm_g < best_norm_g
                    best_x = x;
                    best_norm_g = norm_g;
                end

                run_info = struct(norm_g = best_norm_g, ...
                                  x = best_x, ...
                                  iteration = iteration, ...
                                  sigma = sigma, ...
                                  f_handle = residual, ...
                                  optional = struct());

                [terminate, status] = obj.termination_rule.should_terminate(run_info);

                % Do not terminate when the function value is larger than at x = 0
                if terminate && c' * x + (1 / 2) * x' * mat * x + (1 / r) * sigma * norm_x^r < eps
                    return
                end

                if phi > 0
                    % lambda is too large
                    lambda_max = min(lambda, lambda_max);

                    % Do bisection in log-space
                    lambda = sqrt(lambda_min * lambda_max);
                else
                    % This is not the hard case
                    hard_case = false;

                    % lambda is too small
                    lambda_min = max(lambda, lambda_min);

                    % Compute Newton correction
                    w = R' \ x;
                    derphi = norm(w)^2 / norm_x^3 + (1 / (r - 2)) * (sigma / lambda)^(1 / (r - 2)) * (1 / lambda);
                    delta_lambda = -phi / derphi;

                    lambda = lambda + delta_lambda;
                end

            end
        end

    end
end
