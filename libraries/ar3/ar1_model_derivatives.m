function [m, der1m, der2m] = ar1_model_derivatives(s, f, der1f, sigma)
    % Computes the function value and derivatives of the regularized Taylor
    % model used in the AR1 algorithm

    arguments (Input)
        s (:, 1) double
        % The step at which to evaluate the model

        f (1, 1) double
        % The function value at 0

        der1f (:, 1) double
        % The gradient at 0

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

    m = f + der1f' * s + 1 / 2 * sigma * norm(s)^2;

    if nargout > 1
        der1m = der1f + sigma * s;
    end

    if nargout > 2
        der2m = sigma * eye(n);
    end
end
