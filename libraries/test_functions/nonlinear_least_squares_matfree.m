function [fun, der1f, der2f, der3f] = nonlinear_least_squares_matfree(mat, b, x, mat_free)
    % Nonlinear least squares objective with optional matrix-free derivatives.
    %
    % Inputs:
    %   mat      : (n x d) data matrix
    %   b        : (n x 1) targets
    %   x        : (d x 1) parameter vector
    %   mat_free : logical flag; if true, return mat-free Hessian and 3rd derivative
    %
    % Outputs:
    %   fun  : scalar objective value
    %   der1f: gradient (d x 1)
    %   der2f: either dense Hessian (d x d) or Hv-handle v |-> H v
    %   der3f: either dense 3-tensor (d x d x d) or handle s |-> (v |-> T(0)[s,,] v)

    if nargin < 4 || isempty(mat_free)
        mat_free = true;
    end

    [n, ~] = size(mat);
    t = mat * x;
    [phi, der1phi, der2phi, der3phi] = logistic(t);
    fun = sum((phi - b).^2) / n;

    % Gradient
    if nargout > 1
        der1f = 2 * mat' * (der1phi .* (phi - b)) / n;
    end

    % Hessian (dense or matrix-free)
    if nargout > 2
        der2f_diag = 2 * (der1phi.^2 + der2phi .* (phi - b)) / n;  % (n x 1)

        if mat_free
            % H v = mat' * diag(der2f_diag) * mat * v
            der2f = @(v) mat' * (der2f_diag .* (mat * v));
        else
            der2f = mat' * diag(der2f_diag) * mat;
        end
    end

    % Third derivative (dense tensor or matrix-free oracle)
    if nargout > 3
        diagonal = 2 * (3 * der2phi .* der1phi + der3phi .* (phi - b)) / n;  % (n x 1)

        if mat_free
            % Return a function handle:
            %   der3f(s) returns a function handle v |-> T(0)[s,,] v
            der3f = @(s) @(v) der3f_s_v_handle(diagonal, mat, s, v);
        else
            % Build full 3-tensor as in the original code
            der3f_diag = zeros(n, n, n);
            for i = 1:n
                der3f_diag(i, i, i) = diagonal(i);
            end
            der3f = tensorprod(der3f_diag, mat, 1);
            der3f = tensorprod(der3f, mat, 1);
            der3f = tensorprod(der3f, mat, 1);
        end
    end
end

function der3f_s_v = der3f_s_v_handle(diagonal, mat, s, v)
    % Computes T(0)[s,,] v for the logistic NLS model, in matrix-free form.
    %
    % Formula: T[s,,] v = mat' * ( diagonal .* (mat*s) .* (mat*v) )

    As  = mat * s;   % (n x 1)
    Av  = mat * v;   % (n x 1)
    tmp = diagonal .* As .* Av;  % element-wise
    der3f_s_v = mat' * tmp;      % (d x 1)
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
