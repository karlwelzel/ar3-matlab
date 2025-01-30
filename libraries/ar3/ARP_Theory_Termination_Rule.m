classdef ARP_Theory_Termination_Rule < Termination_Rule
    % Implements the relative subproblem termination condition (TC.r)

    properties
        theta (1, 1) double {mustBePositive} = 100
        tolerance_g (1, 1) double {mustBePositive} = 1e-9 % keep to avoid bugs
    end

    methods

        function [terminate, status] = should_terminate(obj, run)
            % See Birgin et al. "Worst-case evaluation complexity for
            % unconstrained nonlinear optimization using high-order regularized
            % models" equation (2.13)
            if run.norm_g < obj.theta * norm(run.x)^(obj.outer_run.parameters.p)
                terminate = true;
                status = Optimization_Status.SUCCESS;
            else
                [terminate, status] = should_terminate@Termination_Rule(obj, run);
            end
        end

    end
end
