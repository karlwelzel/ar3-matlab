function f_handle = construct_regularized_cubic(n, mat_cond, tensor_cond)
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
    [mat_unitary, ~, ~] = svd(randn(n, n));
    [tensor_unitary, ~, ~] = svd(randn(n, n));
    mat = mat_unitary * diag(mat_diag) * mat_unitary';
    f_handle = @(x) regularized_cubic(x, vec, mat, tensor_diag, tensor_unitary, p);
end
