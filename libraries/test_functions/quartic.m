function [fun, der1f, der2f, der3f] = quartic(x, lambda)
    % This function evaluates the function value and derivatives for
    % norm
    %     f(x) = lambda/2 sum(x_i^4)

    arguments (Input)
        x (:, 1) double
        % d-dimensional input

        lambda (1, 1) double = 1
        % The initial point
    end

    d = length(x);
    fun = lambda / 2 * sum(x.^4);

    if nargout > 1
        der1f = lambda * 2 * x.^3;
    end

    if nargout > 2
        der2f = lambda * 6 * diag(x.^2);
    end

    if nargout > 3
        der3f = zeros(d, d, d);
        for i = 1:d
            der3f(i, i, i) = x(i);
        end
        der3f = 12 * der3f;
    end
end
