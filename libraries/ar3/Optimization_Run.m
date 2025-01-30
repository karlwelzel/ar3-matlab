classdef (Abstract) Optimization_Run < handle
    % Base class for ARP_Run, Fminunc_Run, etc.

    properties
        parameters (1, 1) Optimization_Parameters = ARP_Parameters
        f_handle (1, 1) function_handle = @(x) 0
        x (:, 1) double = nan
        iteration (1, 1) double {mustBeInteger, mustBeNonnegative} = 1
        history (1, :) struct
        current_history_row (1, 1) struct
        default_history_row (1, 1) struct
        optional struct
    end

    methods

        function log_metrics(obj)
            if isfield(obj.optional, "monitor")
                recordMetrics(obj.optional.monitor, obj.iteration, ...
                              f = obj.current_history_row.f, ...
                              norm_g = obj.current_history_row.norm_g);
                max_iterations = obj.parameters.termination_rule.max_iterations;
                obj.optional.monitor.Progress = 100 * (obj.iteration / max_iterations);
            end

            if isfield(obj.optional, "wandb")
                obj.optional.wandb.log(py.dict(obj.current_history_row));
            end

            obj.history(obj.iteration) = obj.current_history_row;
            obj.current_history_row = obj.default_history_row;
            obj.iteration = obj.iteration + 1;
        end

    end
end
