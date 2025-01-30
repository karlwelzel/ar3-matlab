classdef Function_Value_Termination_Rule < Termination_Rule
    % Implements a subproblem termination condition based on the function value

    properties
        f_threshold (1, 1) double = 0
    end

    methods

        function [terminate, status] = should_terminate(obj, run)
            if run.f < obj.f_threshold
                terminate = true;
                status = Optimization_Status.SUCCESS;
            else
                [terminate, status] = should_terminate@Termination_Rule(obj, run);
            end
        end

    end
end
