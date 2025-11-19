function [matvec] = mat_vec(mat, vec)
    if isa(mat, 'function_handle')
        matvec = mat(vec);
    else
        matvec = mat * vec;
    end
end
