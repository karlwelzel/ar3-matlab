function [m, der1m, der2m] = d1_model(s, f, der1f, der2f, der3f, sigma)

    n = length(s);
    m = f * ones(1, n) + der1f * s + (1 / 2) * der2f * s.^2 + (1 / 6) * der3f * s.^3 + 1 / 4 * sigma * s.^4;

    if nargout > 1
        der1m = der1f * ones(1, n) + der2f * s + (1 / 2) * der3f * s.^2 + sigma * s.^3;
    end

    if nargout > 2
        der2m = der2f * ones(1, n) + der3f * s + 3 * sigma * s.^2;
    end
end
