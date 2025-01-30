function [fun, der1f, der2f, der3f] = chebysv_rosenbrock(x)
    % This function evaluates the function value and derivatives for the
    % multi-dimensional Chebyshev-Rosenbrock function
    %     f(x) = (x_1^2 - 1)^2/4 + sum_{i=1}^{d-1} \rho (x_{i+1} - x_i^2 + 1)^2
    rho = 1;
    d = length(x);
    x_cut_d = x(1:d - 1);
    x_cut_1 = x(2:d);
    nu = x_cut_1 - 2 * x_cut_d.^2 + 1; % 1:d-1
    fun = (x(1) - 1)^2 / 4 + rho * sum(nu.^2);

    if nargout > 1
        der1f = zeros(d, 1);
        der1f(1) = (x(1) - 1) / 2 - 8 * rho * x(1) * nu(1);
        der1f(2:d - 1) = 2 * rho * (nu(1:d - 2) - 4 * x(2:d - 1) .* nu(2:d - 1));
        der1f(d) = 2 * rho * nu(d - 1);
    end

    if nargout > 2
        Hess_diag = ones(d, 1) * 200;
        Hess_diag(2:d - 1) = 2 * rho * (1 - 4 * nu(2:d - 1) + 16 * x(2:d - 1).^2);
        Hess_diag(1) = 1 / 2 - 8 * rho * nu(1) + 32 * rho * x(1)^2;
        Hess_diag(d) = 2 * rho;
        Hess_off_diag = -8 * rho * x(1:d - 1);
        der2f = diag(Hess_diag) + diag(Hess_off_diag, 1) + diag(Hess_off_diag, -1);
    end

    if nargout > 3
        der3f = zeros(d, d, d);
        der3f(1, 1, 1) = 96 * rho * x(1);
        der3f(1, 1, 2) = -8 * rho;
        der3f(1, 2, 1) = -8 * rho;
        der3f(2, 1, 1) = -8 * rho;
        for i = 2:d - 1
            der3f(i, i, i) = 96 * rho * x(i);
            der3f(i, i, i + 1) = -8 * rho;
            der3f(i, i + 1, i) = -8 * rho;
            der3f(i + 1, i, i) = -8 * rho;
        end
    end
end
