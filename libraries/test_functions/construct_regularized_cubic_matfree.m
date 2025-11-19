function f_handle = construct_regularized_cubic_matfree(n, mat_cond, tensor_cond)
    p = 8;

    if mat_cond == 0
        mat_diag = linspace(1, 10, n);
    else
        mat_diag = logspace(-mat_cond, mat_cond, n);
        sign_flip_indices = randperm(n, ceil(n / 3));
        mat_diag(sign_flip_indices) = -mat_diag(sign_flip_indices);
        mat_diag = mat_diag(randperm(n));
    end

    if tensor_cond == 0
        tensor_diag = linspace(1, 10, n);
    else
        tensor_diag = logspace(-tensor_cond, 2, n);
        sign_flip_indices = randperm(n, ceil(n / 2));
        tensor_diag(sign_flip_indices) = -tensor_diag(sign_flip_indices);
        tensor_diag = tensor_diag(randperm(n));
    end

    vec = rand(n, 1);
    [mat_unitary, ~, ~]    = svd(randn(n, n));
    [tensor_unitary, ~, ~] = svd(randn(n, n));

    % Matrix-free objective/derivatives handle
    f_handle = @(x) regularized_cubic_matfree(x, vec, mat_unitary, mat_diag, tensor_diag, tensor_unitary, p);
end

function [f, der1f, der2f, der3f] = regularized_cubic_matfree(x, vec, mat_unitary, mat_diag, tensor_diag, ...
                                                              tensor_unitary, p)
    % Matrix-free version of the regularized cubic test function:
    %   f(x) = vec' * x + 1/2 * x' * H * x + 1/6 * T[x]^3 + ||x||^p
    % where:
    %   H = mat_unitary * diag(mat_diag) * mat_unitary'
    %   T has CP form with (tensor_unitary, tensor_diag)
    %
    % Returns:
    %   der2f as v |-> H(x) v
    %   der3f as s |-> (v |-> T(x)[s,v])

    md = mat_diag(:);       % ensure column
    td = tensor_diag(:);    % ensure column

    % Interaction with H (without forming H)
    mat_unitary_x = mat_unitary' * x;
    mat_x         = mat_unitary * (md .* mat_unitary_x);

    % Interaction with T via CP factors (without forming the 3-tensor)
    tensor_unitary_x = tensor_unitary' * x;
    tensor_x2        = tensor_unitary * (td .* (tensor_unitary_x.^2));
    tensor_x3        = sum(td .* (tensor_unitary_x.^3));

    f = vec' * x + (1 / 2) * x' * mat_x + (1 / 6) * tensor_x3 + norm(x)^p;

    if nargout > 1
        der1f = vec + mat_x + (1 / 2) * tensor_x2 + p * norm(x)^(p - 2) * x;
    end

    norm_x = norm(x);
    % Matrix-free Hessian: v |-> H(x)v
    ut_x   = tensor_unitary' * x;
    if nargout > 2
        der2f = @(v) hess_vec_matfree(v, mat_unitary, md, tensor_unitary, ut_x, ...
                                      td, x, norm_x, p);
    end

    % Matrix-free third derivative: s |-> (v |-> T(x)[s,v])
    if nargout > 3
        der3f = @(s) @(v) tensor_vec_vec_matfree(s, v, tensor_unitary, td, x, norm_x, p);
    end
end

function Hv = hess_vec_matfree(v, mat_unitary, md, tensor_unitary, ut_x, ...
                               td, x, norm_x, p)
    % H(x) v =
    %   + mat_unitary * (md .* (mat_unitary' * v))                    (quadratic)
    %   + tensor_unitary * ( (td .* (tensor_unitary' * x)) .* (tensor_unitary' * v) )   (cubic)
    %   + p||x||^(p-2) v + p(p-2)||x||^(p-4) x (x' v)                 (regularizer)

    % Quadratic contribution
    Hv_mat = mat_unitary * (md .* (mat_unitary' * v));

    % Cubic contribution
    Ut_v   = tensor_unitary' * v;
    Hv_cub = tensor_unitary * ((td .* ut_x) .* Ut_v);

    % Regularization contribution
    if norm_x == 0
        Hv_reg = zeros(size(v));
    else
        Hv_reg = p * norm_x^(p - 2) * v + p * (p - 2) * norm_x^(p - 4) * x * (x' * v);
    end

    Hv = Hv_mat + Hv_cub + Hv_reg;
end

function w = tensor_vec_vec_matfree(s, v, tensor_unitary, td, x, norm_x, p)
    % Apply the third derivative in direction s to vector v:
    %   w = T(x)[s,v]
    %
    % Cubic CP term:
    %   T_cubic[s] v = tensor_unitary * ( (td .* (tensor_unitary' * s)) .* (tensor_unitary' * v) )
    %
    % Regularizer r(x) = ||x||^p:
    %   Let r = ||x||, c1 = p(p-2) r^(p-4), c2 = p(p-2)(p-4) r^(p-6).
    %   T_reg[s] v = c2 (x' s) x (x' v) + c1 ( s (x' v) + x (s' v) ) + c1 (x' s) v.

    % Cubic term
    Us  = tensor_unitary' * s;
    Uv  = tensor_unitary' * v;
    w_c = tensor_unitary * ((td .* Us) .* Uv);

    % Regularization term
    if norm_x == 0
        w_r = zeros(size(v));
    else
        c1  = p * (p - 2) * norm_x^(p - 4);
        c2  = p * (p - 2) * (p - 4) * norm_x^(p - 6);
        xTv = x' * v;
        xTs = x' * s;
        sTv = s' * v;
        w_r = c2 * (xTs) * x * xTv + c1 * (s * xTv + x * sTv) + c1 * (xTs) * v;
    end

    w = w_c + w_r;
end
