function [m, der1m, der2m] = ar3_model_derivatives(s, f, der1f, der2f, der3f, sigma)
    % Computes the function value and derivatives of the regularized Taylor
    % model used in the AR3 algorithm

    arguments (Input)
        s (:, 1) double
        % The step at which to evaluate the model

        f (1, 1) double
        % The function value at 0

        der1f (:, 1) double
        % The gradient at 0

        der2f
        % The Hessian at 0 (matrix or function handle)

        der3f
        % The third derivative at 0 (tensor or function handle)

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

    % Handle derivative vec multiplications
    der3f_s = tensor_vec(der3f, s);
    Hs = mat_vec(der2f, s);
    der3f_s_s = mat_vec(der3f_s, s);

    % Model value
    m = f + der1f' * s + (1 / 2) * (s' * Hs) + ...
        (1 / 6) * s' * der3f_s_s + (1 / 4) * sigma * norm_s^4;

    % Gradient
    if nargout > 1
        der1m = der1f + Hs + (1 / 2) * der3f_s_s + sigma * (norm_s^2 * s);
    end

    % Hessian or Hessian-vector product
    if nargout > 2
        if isa(der2f, 'function_handle')
            % Return a Hessian-vector product handle
            der2m = @(v) der2f(v) + der3f_s(v) + sigma * (norm_s^2 * v + 2 * (s' * v) * s);
        else
            % Return the explicit Hessian matrix
            der2m = der2f + der3f_s + sigma * (norm_s^2 * eye(n) + 2 * (s * s'));
        end
    end
end
