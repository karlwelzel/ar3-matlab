function [fun, der1f, der2f, der3f] = nonlinear_least_squares(mat, b, x)
    [n, ~] = size(mat);
    t = mat * x;
    [phi, der1phi, der2phi, der3phi] = logistic(t);
    fun = sum((phi - b).^2) / n;

    if nargout > 1
        der1f = 2 * mat' * (der1phi .* (phi - b)) / n;
    end

    if nargout > 2
        der2f_diag = 2 * (der1phi.^2 + der2phi .* (phi - b)) / n;
        der2f = mat' * diag(der2f_diag) * mat;
    end

    if nargout > 3
        diagonal = 2 * (3 * der2phi .* der1phi + der3phi .* (phi - b)) / n;

        der3f_diag = zeros(n, n, n);
        for i = 1:n
            der3f_diag(i, i, i) = diagonal(i);
        end
        der3f = tensorprod(der3f_diag, mat, 1);
        der3f = tensorprod(der3f, mat, 1);
        der3f = tensorprod(der3f, mat, 1);
    end
end

function [s, der1f, der2f, der3f] = logistic(t)
    n = length(t);
    M = max(zeros(n, 1), t);
    s = exp(t - M) ./ (exp(-M) + exp(t - M));

    if nargout > 1
        der1f = s .* (1 - s);
    end

    if nargout > 2
        der2f = s .* (1 - s) .* (1 - 2 * s);
    end

    if nargout > 3
        der3f = s .* (1 - s) .* (1 - 6 * s + 6 * s .* s);
    end
end
