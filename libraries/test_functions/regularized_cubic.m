function [f, der1f, der2f, der3f_v] = regularized_cubic(x, vec, mat, tensor_diag, tensor_unitary, p)
    % This function evaluates the function value and derivatives for the
    % following test function
    %     f(x) = vec' * x + 1/2 * x' * mat * x + 1/6 * tensor[x]^3 + norm(x)^p
    % where tensor is given by its CP decomposition
    %     tensor = tensor_diag[tensor_unitary]^3
    n = length(x);

    % Interaction with H
    mat_x = mat * x;

    % Interaction with T
    tensor_unitary_x = tensor_unitary' * x;
    tensor_x = tensor_unitary * diag(tensor_diag' .* tensor_unitary_x) * tensor_unitary';
    tensor_x2 = tensor_unitary * (tensor_diag' .* tensor_unitary_x.^2);
    tensor_x3 = sum(tensor_diag' .* tensor_unitary_x.^3);

    f = vec' * x + (1 / 2) * x' * mat_x + (1 / 6) * tensor_x3 + norm(x)^p;

    if nargout > 1
        der1f = vec + mat_x + (1 / 2) * tensor_x2 + p * norm(x)^(p - 2) * x;
    end

    if nargout > 2
        % mat = mat_unitary * diag(mat_diag) * mat_unitary';
        der2f = mat + tensor_x + ...
          (p * (p - 2) * norm(x)^(p - 4) * (x * x') + p * norm(x)^(p - 2) * eye(n));
    end

    if nargout > 3
        der3f_v = @(v) tensor_unitary * diag(tensor_diag' .* (tensor_unitary' * v)) * tensor_unitary' + ...
          der3regularization_v(x, p, n, v);
    end
end

function tv = der3regularization_v(x, p, n, v)
    tv = p * (p - 2) * (p - 4) * norm(x)^(p - 6) * (x' * v) * (x * x') + ...
      p * (p - 2) * norm(x)^(p - 4) * ((v * x') + (x * v')) + ...
      p * (p - 2) * norm(x)^(p - 4) * (x' * v) * eye(n);
end
