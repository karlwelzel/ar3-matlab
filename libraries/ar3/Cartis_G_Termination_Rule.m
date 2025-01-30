classdef Cartis_G_Termination_Rule < Termination_Rule
    % Implements the g rule termination condition from the ARC literature

    properties
        relative_tolerance_g (1, 1) double {mustBePositive} = 1e-4
    end

    methods

        function [terminate, status] = should_terminate(obj, run)
            % See Cartis et al. "Adaptive cubic regularisation methods for
            % unconstrained optimization. Part I: motivation, convergence and
            % numerical results" equation (7.1)
            if run.norm_g < min(obj.relative_tolerance_g, obj.outer_run.norm_g^(1 / 2)) * obj.outer_run.norm_g
                terminate = true;
                status = Optimization_Status.SUCCESS;
            else
                [terminate, status] = should_terminate@Termination_Rule(obj, run);
            end
        end

    end
end
