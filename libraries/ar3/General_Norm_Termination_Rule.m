classdef General_Norm_Termination_Rule < Termination_Rule
    % Implements the generalized norm subproblem termination condition

    properties
        theta (1, 1) double {mustBePositive} = 1
        tolerance_g (1, 1) double {mustBePositive} = 1e-9 % keep to avoid bugs
    end

    methods

        function [terminate, status] = should_terminate(obj, run)
            % See Gratton and Toint. "Adaptive regularization minimization
            % algorithms with nonsmooth norms" equation (2.6)
            norm_step = norm(run.x);
            if obj.outer_run.parameters.p == 3
                [~, der1m] = run.f_handle(run.x);
                norm_der1t = norm(der1m - run.sigma * ( ...
                                                       norm_step^obj.outer_run.parameters.p * run.x));
            elseif obj.outer_run.parameters.p == 2
                norm_der1t = norm(run.f_handle(run.x));
            else
                error("Invalid order p");
            end

            if norm_der1t < obj.theta * run.sigma * norm_step^( ...
                                                               obj.outer_run.parameters.p)
                terminate = true;
                status = Optimization_Status.SUCCESS;
            else
                [terminate, status] = should_terminate@Termination_Rule(obj, run);
            end
        end

    end
end
