classdef Fminunc_Run < Optimization_Run
    % Executes fminunc using the given parameters

    properties
        f (1, 1) double = nan
        norm_g (1, 1) double = nan
        norm_step (1, 1) double = nan
        start_time (1, 1) uint64 = nan
        status (1, 1) Optimization_Status = Optimization_Status.RUNNING
    end

    methods

        function obj = Fminunc_Run(f_handle, x0, parameters, optional)
            obj.f_handle = f_handle;
            obj.x = x0;
            obj.parameters = parameters;
            obj.optional = optional;

            obj.default_history_row = struct(f = nan, ...
                                             norm_g = nan, ...
                                             norm_step = nan, ...
                                             total_fun = nan, ...
                                             total_der = nan, ...
                                             time = nan);
            obj.history = obj.default_history_row;
            obj.current_history_row = obj.default_history_row;
        end

        function terminate = callback(obj, x, optim_values, state)
            if state == "init" || state == "interrupt"
                terminate = false;
            elseif state == "iter"
                obj.x = x;
                obj.iteration = optim_values.iteration + 1;
                obj.f = optim_values.fval;
                obj.norm_g = optim_values.firstorderopt;
                if obj.parameters.algorithm == "trust-region"
                    obj.norm_step = optim_values.stepsize;
                end

                % Log metrics
                obj.current_history_row.f = obj.f;
                obj.current_history_row.norm_g = obj.norm_g;
                obj.current_history_row.norm_step = obj.norm_step;
                obj.current_history_row.total_fun = optim_values.funccount;
                obj.current_history_row.total_der = optim_values.iteration;
                obj.current_history_row.time = toc(obj.start_time);
                obj.log_metrics();

                [terminate, obj.status] = obj.parameters.termination_rule.should_terminate(obj);
            elseif state == "done"
                % fminunc stopped even though the tolerances are set to zero
                terminate = false;
                obj.status = Optimization_Status.NUMERICAL_ISSUES;
            else
                error("Invalid state %s returned by fminunc", state);
            end
        end

        function run(obj)
            obj.start_time = tic;

            options = obj.parameters.to_optimoptions();
            options.OutputFcn = @obj.callback;
            try
                [~, ~, exitflag, ~] = fminunc(obj.f_handle, obj.x, options);
            catch error
                switch error.identifier
                    case 'optim:lineSearch:FPrimeInitialNeg'
                        obj.status = Optimization_Status.NUMERICAL_ISSUES;
                        return
                    otherwise
                        rethrow(error);
                end
            end

            % Check that optimization was terminated by the callback function
            % or because the change in x was too small (numerical issues)
            assert(exitflag == -1 || exitflag == 2 || exitflag == 5);
        end

    end

end
