function [fun, der1f, der2f, der3f] = d1_fun(x)
    % This function evaluates the function value and derivatives for
    % norm
    %     f(x) = k*x^8/336 + d*x^6/120 + c*x^5/60 + b*x^3/2 + a*x

    arguments (Input)
        x (:, 1) double
    end
    k = 0.1344;
    d = -0.9;
    c = -0.5;
    b = 1;
    a = -0.74;
    % Lipschitz 360: f''''(5) = 347.5, f''''(-5) = 357.5

    fun = k * x.^8 / 336 + d * x.^6 / 120 + c * x.^5 / 60 + b * x.^3 / 6 + a * x + 6;

    if nargout > 1
        der1f = k * x.^7 / 42 + d * x.^5 / 20 + c * x.^4 / 12 + b * x.^2 / 2 + a;
    end

    if nargout > 2
        der2f = k * x.^6 / 6 + d * x.^4 / 4 + c * x.^3 / 3 + b * x;
    end

    if nargout > 3
        der3f = k * x.^5 + d * x.^3 + c * x.^2 + b;
    end
end
