function persistent_alpha = analyze_persistent_min(taylor_poly, tolerance)
    % Takes a one-dimensional polynomial and analyzes it persistent minimizers.
    %
    % This function assumes the input polynomial to be a pth-order Taylor
    % expansion t of a one-dimensional function. There is a unique persistent
    % minimizer curve gamma: I -> R. The image of this curve is an interval
    % where one endpoint is zero. If the interval contains positive values,
    % this function returns the positive endpoint (possibly +inf). Otherwise,
    % the function returns -1. The tolerance argument relaxes the requirements
    % for a point to count as persistent as described in the paper.

    arguments (Input)
        taylor_poly (1, :) double
        tolerance (1, 1) double {mustBeNonnegative} = 0
    end

    p = length(taylor_poly) - 1;

    % The step is transient if it is not a descent direction
    if taylor_poly(end - 1) >= 0
        persistent_alpha = -1;
    else
        taylor_der_poly = padded_polyder(taylor_poly);
        pos_sigma_poly = [zeros(1, p - 1), tolerance] - taylor_der_poly;
        local_min_poly = [padded_polyder(taylor_der_poly), 0] + p * pos_sigma_poly;

        alpha_options1 = real_roots(pos_sigma_poly);
        alpha_options1 = alpha_options1(polyval(local_min_poly, alpha_options1) >= -10 * eps);
        alpha_options2 = real_roots(local_min_poly);
        alpha_options2 = alpha_options2(polyval(pos_sigma_poly, alpha_options2) >= -10 * eps);

        alpha_options = [alpha_options1; alpha_options2];
        alpha_options = alpha_options(alpha_options > 0);
        persistent_alpha = min([alpha_options; inf]);
    end
end
