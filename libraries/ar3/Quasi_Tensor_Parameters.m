classdef Quasi_Tensor_Parameters < Parameters
    properties
        p (1, 1) double {mustBeInteger, mustBeGreaterThan(p, 1), mustBeLessThan(p, 4)} = 3
        type Quasi_Tensor_Type = Quasi_Tensor_Type.POWELL_SYMMETRIC_BROYDEN
    end

    methods (Static)

        function obj = from_struct(params)
            obj = Quasi_Tensor_Parameters;
            obj = obj.update(params);
        end

    end
end
