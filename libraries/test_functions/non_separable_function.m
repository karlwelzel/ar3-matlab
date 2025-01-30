function [f, der1f, der2f, der3f] = non_separable_function(x, mat, p)
    % This function evaluates the function value and derivatives for the
    % following test function
    %     f(x) = 1/2*x' * mat * x - 5 * [1:n] * sin(x) + norm(x)^p

    n = length(x);
    mat = (mat + mat') / 2; % Symmetrize the matrix

    f = (1 / 2) * x' * mat * x - 5 * [1:n] * sin(x) + norm(x)^p;

    if nargout > 1
        der1f = mat * x - 5 * [1:n]' .* cos(x) + p * norm(x)^(p - 2) * x;
    end

    if nargout > 2
        der2f = mat + 5 * diag([1:n]' .* sin(x)) + ...
          (p * (p - 2) * norm(x)^(p - 4) * (x * x') + p * norm(x)^(p - 2) * eye(n));
    end

    if nargout > 3
        der3f = zeros(n, n, n);
        for i = 1:n
            der3f(i, i, i) = 5 * i * cos(x(i));
        end
        uvec = eye(n);
        for i = 1:n
            der3f(:, :, i) = der3f(:, :, i) + ...
              p * (p - 2) * norm(x)^(p - 4) * x * uvec(i, :) + ...
              p * (p - 2) * norm(x)^(p - 4) * uvec(i, :)' * x' + ...
              eye(n) * x(i) * p * (p - 2) * norm(x)^(p - 4) + ...
              x(i) * p * (p - 2) * (p - 4) * norm(x)^(p - 6) * (x * x');
        end
    end
end
