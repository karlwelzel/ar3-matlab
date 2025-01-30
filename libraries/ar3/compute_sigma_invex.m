function sigma_invex = compute_sigma_invex(taylor_der_poly, norm_step)
    % Computes the sigma that makes the one-dimensional AR3 model invex

    % This function assumes that taylor_poly represents a third-order
    % polynomial and computes a value of sigma such that
    %     polyval(taylor_poly, x) + (sigma / 4) * norm_step * x^4
    % is an invex function in x. This means that every stationary point is a
    % global minimum. If the function is invex for all sigma, this subroutine
    % returns NaN.
    % Note: Using derivative of the taylor_poly as input rather to use
    % polyder(taylor_poly), as it gives a bug when it contains zero

    assert(length(taylor_der_poly) == 3); % Third-order polynomial

    a = norm_step^4;
    b = taylor_der_poly(1);
    c = taylor_der_poly(2);
    d = taylor_der_poly(3);
    sigma_poly = [
                  27 * a^2 * d^2
                  (4 * c^3 - 18 * b * c * d) * a
                  (4 * b * d - c^2) * b^2
                 ];
    sigma_invex = max([real_roots(sigma_poly); nan]);
end
