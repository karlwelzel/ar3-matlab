function [tesorvec] = tensor_vec(tesor, vec)
    if isa(tesor, 'function_handle')
        tesorvec = tesor(vec);
    else
        tesorvec = tensorprod(der3f, vec, 1);
    end
end
