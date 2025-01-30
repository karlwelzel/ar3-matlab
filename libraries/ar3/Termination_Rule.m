classdef (Abstract) Termination_Rule < Parameters
    % Base class for termination conditions

    % This class implements tests for when the optimization method should
    % terminate unsuccessfully, which are shared among all methods.

    properties
        outer_run (1, 1) = struct() % Optimization_Run or struct
        max_iterations (1, 1) double {mustBeInteger, mustBePositive} = 1000
    end

    methods (Static)

        function obj = from_struct(params)
            if params.rule == "First_Order"
                obj = First_Order_Termination_Rule;
            elseif params.rule == "Cartis_G"
                obj = Cartis_G_Termination_Rule;
            elseif params.rule == "Cartis_S"
                obj = Cartis_S_Termination_Rule;
            elseif params.rule == "ARP_Theory"
                obj = ARP_Theory_Termination_Rule;
            elseif params.rule == "General_Norm"
                obj = General_Norm_Termination_Rule;
            elseif params.rule == "Function_Value"
                obj = Function_Value_Termination_Rule;
            else
                error("Unknown termination rule: " + params.rule);
            end
            params = rmfield(params, "rule");
            obj = obj.update(params);
        end

    end

    methods

        function [terminate, status] = should_terminate(obj, run)

            arguments (Input)
                obj
                run (1, 1) % Optimization_Run or struct
            end

            arguments (Output)
                terminate (1, 1) logical
                status (1, 1) Optimization_Status
            end

            if isa(run, "Optimization_Run") && run.f < -1 / eps && run.norm_g > 1 / eps && ...
              run.norm_step ~= 0
                terminate = true;
                status = Optimization_Status.NOT_LOWER_BOUNDED;
            elseif run.iteration >= obj.max_iterations
                terminate = true;
                status = Optimization_Status.MAX_ITERATIONS_EXCEEDED;
            elseif isfield(run.optional, "monitor") && run.optional.monitor.Stop
                terminate = true;
                status = Optimization_Status.USER_TERMINATED;
            else
                terminate = false;
                status = Optimization_Status.RUNNING;
            end
        end

    end
end
