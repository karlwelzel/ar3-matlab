function [fun, der1f, der2f, der3f] = rosenbrock_matfree(x)
    % Matrix-free version of the multi-dimensional Rosenbrock function
    %   f(x) = sum_{i=1}^{d-1} 100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2

    d = length(x);
    x_cut_d = x(1:d - 1);
    x_cut_1 = x(2:d);

    % Function value
    fun = sum(100 * (x_cut_d.^2 - x_cut_1).^2 + (x_cut_d - 1).^2);

    % Gradient
    if nargout > 1
        der1f = zeros(d, 1);
        der1f_cut_d = 400 * (x_cut_d.^2 - x_cut_1) .* x_cut_d + 2 * (x_cut_d - 1);
        der1f_cut_1 = -200 * (x_cut_d.^2 - x_cut_1);
        der1f(1:d - 1) = der1f(1:d - 1) + der1f_cut_d;
        der1f(2:d)     = der1f(2:d)     + der1f_cut_1;
    end

    % Hessian-vector product (matrix-free, tridiagonal)
    if nargout > 2
        Hess_diag = ones(d, 1) * 200;
        Hess_diag(1:d - 1) = 400 * (3 * x_cut_d.^2 - x_cut_1) + 202;
        Hess_diag(1) = Hess_diag(1) - 200;

        Hess_off_diag = -400 * x_cut_d;   % length d-1

        der2f = @(v) apply_tridiag(Hess_diag, Hess_off_diag, v);
    end

    % Third derivative directional oracle: s |-> (v |-> T(x)[s,,] v)
    if nargout > 3
        der3f = @(s) @(v) apply_third_dir(x, s, v);
    end
end

function Hv = apply_tridiag(dd, od, v)
    % Apply symmetric tridiagonal matrix H with
    %   diag(H)        = dd   (length d)
    %   diag(H, +/-1)  = od   (length d-1)
    % to vector v.

    % Start with diagonal contribution
    Hv = dd .* v;

    % Upper off-diagonal: H(i,i+1) = od(i)
    % contributes od(i) * v(i+1) to row i.
    Hv(1:end - 1) = Hv(1:end - 1) + od .* v(2:end);

    % Lower off-diagonal: H(i+1,i) = od(i)
    % contributes od(i) * v(i) to row i+1.
    Hv(2:end) = Hv(2:end) + od .* v(1:end - 1);
end

function w = apply_third_dir(x, s, v)
    % Matrix-free application of the 3rd-derivative tensor T(x) of Rosenbrock:
    %   w = T(x)[s,,] v
    %
    % Nonzero pattern in the third derivative:
    %   for i = 1:d-1
    %       if i > 1, T(i-1,i-1,i) = -400; end
    %       T(i,i,i)   = 2400 * x(i);
    %       T(i,i+1,i) = -400;
    %       T(i+1,i,i) = -400;
    %   end
    %   T(d-1,d-1,d) = -400;

    d = length(x);
    w = zeros(d, 1);

    if d == 1
        % Rosenbrock sum is empty, T = 0.
        return
    end

    % Convenience: length d-1 vectors
    vm1 = v(1:d - 1);

    % A: T(i,i,i) = 2400*x(i), contributes to w(i)
    w(1:d - 1) = w(1:d - 1) + 2400 * x(1:d - 1) .* s(1:d - 1) .* vm1;

    % B: T(i, i+1, i) = -400, contributes to w(i)
    w(1:d - 1) = w(1:d - 1) - 400 * s(2:d) .* vm1;

    % C: T(i+1, i, i) = -400, contributes to w(i+1)
    w(2:d) = w(2:d) - 400 * s(1:d - 1) .* vm1;

    % D: T(i-1, i-1, i) = -400 for i = 2,...,d-1, contributes to w(i-1)
    if d > 2
        w(1:d - 2) = w(1:d - 2) - 400 * s(1:d - 2) .* v(2:d - 1);
    end

    % Extra entry: T(d-1, d-1, d) = -400, contributes to w(d-1)
    w(d - 1) = w(d - 1) - 400 * s(d - 1) * v(d);
end
