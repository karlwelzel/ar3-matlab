function [fun, der1f, der2f, der3f] = square(x, lambda)
    % This function evaluates the function value and derivatives for the
    % squared sum over the entries of x
    %     f(x) = lambda/2 (sum x_i)^2

    arguments (Input)
        x (:, 1) double
        % d-dimensional input

        lambda (1, 1) double = 1
        % The initial point
    end

    d = length(x);
    fun = lambda / 2 * (sum(x))^2;

    if nargout > 1
        der1f = lambda * (sum(x)) * ones(d, 1);
    end

    if nargout > 2
        der2f = lambda * ones(d, d);
    end

    if nargout > 3
        der3f = zeros(d, d, d);
    end
end
