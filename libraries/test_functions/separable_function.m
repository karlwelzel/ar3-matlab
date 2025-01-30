function [f, der1f, der2f, der3f] = separable_function(x, mat, a, p)
    % This function evaluates the function value and derivatives for the
    % following test function
    %     f(x) = 1/2*x' * mat * x - 5 * [1:n] * sin(x) + sum((a.*x).^p)

    n = length(x);
    mat = (mat + mat') / 2; % Symmetrize the matrix

    f = (1 / 2) * x' * mat * x - 5 * [1:n] * sin(x) + sum((a .* x).^p);

    if nargout > 1
        der1f = mat * x - 5 * [1:n]' .* cos(x) + p * a .* (a .* x).^(p - 1);
    end

    if nargout > 2
        der2f = mat + 5 * diag([1:n]' .* sin(x)) + diag(p * (p - 1) * a.^2 .* (a .* x).^(p - 2));
    end

    if nargout > 3
        diagonal = 5 * [1:n]' .* cos(x) + p * (p - 1) * (p - 2) * a.^3 .* (a .* x).^(p - 3);

        der3f = zeros(n, n, n);
        for i = 1:n
            der3f(i, i, i) = diagonal(i);
        end
    end
end
