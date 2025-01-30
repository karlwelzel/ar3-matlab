function [fun, der1f, der2f, der3f] = rosenbrock(x)
    % This function evaluates the function value and derivatives for the
    % multi-dimensional Rosenbrock function
    %     f(x) = sum_{i=1}^{d-1} 100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2

    d = length(x);
    x_cut_d = x(1:d - 1);
    x_cut_1 = x(2:d);
    fun = sum(100 * (x_cut_d.^2 - x_cut_1).^2 + (x_cut_d - 1).^2);

    if nargout > 1
        der1f = zeros(d, 1);
        der1f_cut_d = 400 * (x_cut_d.^2 - x_cut_1) .* x_cut_d + 2 * (x_cut_d - 1);
        der1f_cut_1 = -200 * (x_cut_d.^2 - x_cut_1);
        der1f(1:d - 1) = der1f(1:d - 1) + der1f_cut_d;
        der1f(2:d) = der1f(2:d) + der1f_cut_1;
    end

    if nargout > 2
        Hess_diag = ones(d, 1) * 200;
        Hess_diag(1:d - 1) = 400 * (3 * x_cut_d.^2 - x_cut_1) + 202;
        Hess_diag(1) = Hess_diag(1) - 200;
        Hess_off_diag = -400 * x_cut_d;
        der2f = diag(Hess_diag) + diag(Hess_off_diag, 1) + diag(Hess_off_diag, -1);
    end

    if nargout > 3
        der3f = zeros(d, d, d);
        for i = 1:d - 1
            if i > 1
                der3f(i - 1, i - 1, i) = -400;
            end
            der3f(i, i, i) = 2400 * x_cut_d(i);
            der3f(i, i + 1, i) = -400;
            der3f(i + 1, i, i) = -400;
        end
        der3f(d - 1, d - 1, d) = -400;
    end
end
