function [fun, der1f, der2f, der3f] = chebysv_rosenbrock_matfree(x)
    % Matrix-free version of the multi-dimensional Chebyshev-Rosenbrock function
    %     f(x) = (x_1 - 1)^2/4 + sum_{i=1}^{d-1} rho * (x_{i+1} - 2 x_i^2 + 1)^2
    %
    % Outputs:
    %   fun   : scalar value
    %   der1f : gradient (d x 1)
    %   der2f : function handle, v |-> H(x) v   (matrix-free Hessian)
    %   der3f : function handle, s |-> (v |-> T(x)[s,v])  (matrix-free 3rd derivative)

    rho = 1;
    d = length(x);

    % Shorthands
    x_cut_d = x(1:d - 1);
    x_cut_1 = x(2:d);
    nu = x_cut_1 - 2 * x_cut_d.^2 + 1;   % size d-1

    % Function value
    fun = (x(1) - 1)^2 / 4 + rho * sum(nu.^2);

    % Gradient
    if nargout > 1
        der1f = zeros(d, 1);
        der1f(1)        = (x(1) - 1) / 2 - 8 * rho * x(1) * nu(1);
        if d > 2
            der1f(2:d - 1) = 2 * rho * (nu(1:d - 2) - 4 * x(2:d - 1) .* nu(2:d - 1));
        end
        der1f(d)        = 2 * rho * nu(d - 1);
    end

    % Hessian-vector product (matrix-free), using tri-diagonal structure
    if nargout > 2
        Hess_diag = ones(d, 1) * 200;  % placeholder overwritten below
        if d > 2
            Hess_diag(2:d - 1) = 2 * rho * (1 - 4 * nu(2:d - 1) + 16 * x(2:d - 1).^2);
        end
        Hess_diag(1) = 1 / 2 - 8 * rho * nu(1) + 32 * rho * x(1)^2;
        Hess_diag(d) = 2 * rho;

        Hess_off_diag = -8 * rho * x(1:d - 1);  % length d-1

        % Return handle v |-> H v without forming H explicitly
        der2f = @(v) apply_tridiag(Hess_diag, Hess_off_diag, v);
    end

    % Third-derivative directional oracle (matrix-free)
    % Uses exactly the nonzeros from the explicit tensor:
    %   T(1,1,1) = 96*rho*x1;    T(1,1,2)=T(1,2,1)=T(2,1,1) = -8*rho;
    %   For i=2..d-1:
    %     T(i,i,i) = 96*rho*xi;
    %     T(i,i,i+1) = T(i,i+1,i) = T(i+1,i,i) = -8*rho.
    if nargout > 3
        der3f = @(s) @(v) apply_third_dir(x, s, v, rho);
    end
end

function Hv = apply_tridiag(dd, od, v)
    % Apply symmetric tridiagonal matrix H (diag=dd, offdiag=od) to v.
    % dd: (d x 1), od: (d-1 x 1), v: (d x 1)
    d = length(dd);
    Hv = zeros(d, 1);

    if d == 1
        Hv(1) = dd(1) * v(1);
        return
    end

    % First row
    Hv(1) = dd(1) * v(1) + od(1) * v(2);

    % Middle rows
    if d > 2
        i = 2:d - 1;
        Hv(i) = od(i - 1) .* v(i - 1) + dd(i) .* v(i) + od(i) .* v(i + 1);
    end

    % Last row
    Hv(d) = od(d - 1) * v(d - 1) + dd(d) * v(d);
end

function w = apply_third_dir(x, s, v, rho)
    % Compute w = T(x)[s,v] using only the non-zeros specified
    % in the original explicit third derivative code.
    d = length(x);
    w = zeros(d, 1);

    if d == 1
        % Only nonzero is T(1,1,1) = 96*rho*x(1)
        w(1) = 96 * rho * x(1) * s(1) * v(1);
        return
    end

    w(1) = w(1) + 96 * rho * x(1) * s(1) * v(1) - 8 * rho * (s(1) * v(2) + s(2) * v(1));
    w(2) = w(2) - 8 * rho * (s(1) * v(1));

    for i = 2:d - 1
        w(i)   = w(i)   + 96 * rho * x(i) * s(i) * v(i) - 8 * rho * (s(i) * v(i + 1) + s(i + 1) * v(i));
        w(i + 1) = w(i + 1) - 8 * rho * (s(i) * v(i));
    end
end
