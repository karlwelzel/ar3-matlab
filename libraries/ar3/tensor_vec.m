function [tesorvec] = tensor_vec(tesor, vec)
    if isa(tesor, 'function_handle')
        tesorvec = tesor(vec);
    else
        tesorvec = tensorprod(tesor, vec, 1);
    end
end
