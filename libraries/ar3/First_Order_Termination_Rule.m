classdef First_Order_Termination_Rule < Termination_Rule
    % Implements the absolute subproblem termination condition (TC.a)

    properties
        theta (1, 1) double {mustBePositive} = 100 % keep to avoid bugs
        tolerance_g (1, 1) double {mustBePositive} = 1e-9
    end

    methods

        function [terminate, status] = should_terminate(obj, run)
            if run.norm_g < obj.tolerance_g
                terminate = true;
                status = Optimization_Status.SUCCESS;
            else
                [terminate, status] = should_terminate@Termination_Rule(obj, run);
            end
        end

    end
end
