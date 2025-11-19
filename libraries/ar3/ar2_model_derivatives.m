function [m, der1m, der2m] = ar2_model_derivatives(s, f, der1f, der2f, sigma)
    % Computes the function value and derivatives of the regularized Taylor
    % model used in the AR2 algorithm

    arguments (Input)
        s (:, 1) double
        % The step at which to evaluate the model

        f (1, 1) double
        % The function value at 0

        der1f (:, 1) double
        % The gradient at 0

        der2f
        % The Hessian at 0 (matrix or function handle)

        sigma (1, 1) double {mustBeNonnegative}
        % The regularization parameter
    end

    arguments (Output)
        m (1, 1) double
        % The value of the model at s

        der1m (:, 1) double
        % The gradient of the model at s

        der2m
        % The Hessian of the model at s (matrix or function handle)
    end

    n = length(s);
    norm_s = norm(s);

    % Apply Hessian at s
    Hs = mat_vec(der2f, s);

    % Model value
    m = f + der1f' * s + (1 / 2) * (s' * Hs) + (1 / 3) * sigma * norm_s^3;

    % Gradient
    if nargout > 1
        der1m = der1f + Hs + sigma * (norm_s * s);
    end

    % Hessian or Hessian-vector product
    if nargout > 2
        if isa(der2f, 'function_handle')
            % Return Hessian-vector product handle
            if norm_s == 0
                der2m = @(v) der2f(v);
            else
                der2m = @(v) der2f(v) + sigma * (norm_s * v + ((s' * v) / norm_s) * s);
            end
        else
            % Return explicit Hessian matrix
            if norm_s == 0
                der2m = der2f;
            else
                der2m = der2f + sigma * (norm_s * eye(n) + (s * s') / norm_s);
            end
        end
    end
end
