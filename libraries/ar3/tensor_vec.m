function [tensorvec] = tensor_vec(tensor, vec)
    if isa(tensor, 'function_handle')
        tensorvec = tensor(vec);
    else
        tensorvec = tensorprod(tensor, vec, 1);
    end
end
