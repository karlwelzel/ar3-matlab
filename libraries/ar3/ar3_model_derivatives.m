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

        der2f (:, :) double
        % The Hessian at 0

        % der3f (:, :, :) double
        der3f
        % The third derivative at 0

        sigma (1, 1) double {mustBeNonnegative}
        % The regularization parameter
    end

    arguments (Output)
        m (1, 1) double
        % The value of the model at s

        der1m (:, 1) double
        % The gradient of the model at s

        der2m (:, :) double
        % The Hessian of the model at s
    end

    n = length(s);

    if isa(der3f, 'function_handle')
        der3f_s = der3f(s);
    else
        der3f_s = tensorprod(der3f, s, 1);
    end

    m = f + der1f' * s + (1 / 2) * s' * der2f * s + (1 / 6) * s' * der3f_s * s + 1 / 4 * sigma * norm(s)^4;

    if nargout > 1
        der1m = der1f + der2f * s + (1 / 2) * der3f_s * s + sigma * (norm(s)^2 * s);
    end

    if nargout > 2
        der2m = der2f + der3f_s + sigma * (norm(s)^2 * eye(n) + 2 * (s * s'));
    end
end
