% This script implements the derivative test described in the technical report
% E.G. Birgin, J. L. Gardenghi, J. M. Martinez, S. A. Santos "Third-order
% derivatives of the More, Garbow and Hillstrom test set problems", 2018,
% originally devised in V. Z. Averbukh, S. Figueroa, T. Schlick "Remark on
% algorithm 566", 1994.

% The numbers produced by this script will converge to 2 if no derivatives
% are correct, to 4 if the gradients are correct, to 8 if gradients and
% Hessians are correct and to 16 if gradients, Hessians and third derivatives
% are correctly computed. It can be used to check the correctness of the
% implementation of derivatives.

% The check does not work if the Taylor expansion is exact. This is the
% case for the "l2" and "square" function as well as MGH 32, 33 and 34.

% Choose function to test

% [x0, f_handle] = params2problem('{"name": "l2", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "square", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "quartic", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "rosenbrock", "dim": 5}');
[x0, f_handle] = params2problem('{"name": "chebysv_rosenbrock", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "separable_function", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "non_separable_function", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "nonlinear_least_squares", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "ill_cond_bm", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "ill_cond_H", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "ill_cond_T", "dim": 5}');
% [x0, f_handle] = params2problem('{"name": "ill_cond_HT", "dim": 5}');
% [x0, f_handle] = params2problem(1);

step = randn(length(x0), 1);

% Define Taylor expansion
[f, der1f, der2f, der3f_v] = f_handle(x0);
if isa(der3f_v, "function_handle")
    taylor = @(s) f + der1f' * s + (1 / 2) * s' * der2f * s + (1 / 6) * s' * (der3f_v(s) * s);
else
    taylor = @(s) f + der1f' * s + (1 / 2) * s' * der2f * s + (1 / 6) * s' * (tensorprod(der3f_v, s, 1) * s);
end

% Compute approximation errors
evaluations = 11;
errors = ones(1, evaluations);
for i = 1:evaluations
    factor = 1 / (2^i);
    errors(i) = abs(f_handle(x0 + factor * step) - taylor(factor * step));
end

% Display ratios between errors
for i = 1:length(errors) - 1
    disp("" + i + " " + (errors(i) / errors(i + 1)));
end
