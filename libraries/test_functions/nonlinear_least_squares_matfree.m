function [fun, der1f, der2f, der3f] = nonlinear_least_squares_matfree(mat, b, x)
    % Nonlinear least squares objective with optional matrix-free derivatives.
    %
    % Inputs:
    %   mat      : (n x d) data matrix
    %   b        : (n x 1) targets
    %   x        : (d x 1) parameter vector
    %
    % Outputs:
    %   fun  : scalar objective value
    %   der1f: gradient (d x 1)
    %   der2f: Hessian-vector product handle v |-> H v
    %   der3f: tensor-vector-vector product handle s |-> (v |-> T[s,v])

    [n, ~] = size(mat);
    t = mat * x;
    [phi, der1phi, der2phi, der3phi] = logistic(t);
    fun = sum((phi - b).^2) / n;

    % Gradient
    if nargout > 1
        der1f = 2 * mat' * (der1phi .* (phi - b)) / n;
    end

    % Hessian
    if nargout > 2
        der2f_diag = 2 * (der1phi.^2 + der2phi .* (phi - b)) / n;  % (n x 1)

        der2f = @(v) mat' * (der2f_diag .* (mat * v));
    end

    % Third derivative
    if nargout > 3
        diagonal = 2 * (3 * der2phi .* der1phi + der3phi .* (phi - b)) / n;  % (n x 1)

        der3f = @(s) @(v) der3f_s_v_handle(diagonal, mat, s, v);
    end
end

function der3f_s_v = der3f_s_v_handle(diagonal, mat, s, v)
    % Computes T[s,v] for the logistic NLS model, in matrix-free form.
    der3f_s_v = mat' * (diagonal .* (mat * s) .* (mat * v));
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
