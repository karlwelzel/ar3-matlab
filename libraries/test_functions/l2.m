function [fun, der1f, der2f, der3f] = l2(x, lambda)
    % This function evaluates the function value and derivatives for L2
    % norm
    %     f(x) = lambda/2 | x |^2

    arguments (Input)
        x (:, 1) double
        % d-dimensional input

        lambda (1, 1) double = 1
        % The initial point
    end

    d = length(x);
    fun = lambda * (x' * x) / 2;

    if nargout > 1
        der1f = lambda * x;
    end

    if nargout > 2
        der2f = lambda * diag(ones(d, 1));
    end

    if nargout > 3
        der3f = zeros(d, d, d);
    end
end
