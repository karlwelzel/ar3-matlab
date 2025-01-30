classdef Fminunc_Parameters < Optimization_Parameters
    % Stores parameters for optimizing a function using fminunc

    properties
        algorithm (1, 1) string = "trust-region"
        termination_rule (1, 1) Termination_Rule = First_Order_Termination_Rule
    end

    methods (Static)

        function obj = from_struct(params)
            obj = Fminunc_Parameters;

            if isfield(params, "verbosity")
                obj.verbosity = params.verbosity;
                params = rmfield(params, "verbosity");
            end

            if isfield(params, "x0_shift")
                obj.x0_shift = params.x0_shift;
                params = rmfield(params, "x0_shift");
            end

            % The options for "algorithm" are
            % - "trust-region" (uses gradient & Hessian)
            % - "bfgs" (uses gradient)
            % - "lbfgs" (uses gradient)
            % - "dfp" (uses gradient)
            % - "steepdesc" (uses gradient)

            if isfield(params, "algorithm")
                obj.algorithm = params.algorithm;
                params = rmfield(params, "algorithm");
            end

            if isfield(params, "stop_rule")
                [termination_params, params] = extract_params(params, "stop_");
                obj.termination_rule = Termination_Rule.from_struct(termination_params);
            end

            remaining_fieldnames = fieldnames(params);
            if ~isempty(remaining_fieldnames)
                error("Unrecognized field: " + remaining_fieldnames{1});
            end
        end

    end

    methods

        function options = to_optimoptions(obj)
            options = optimset("fminunc");
            options.MaxIter = inf;
            options.MaxFunEvals = inf;
            options.TolFun = 0;
            options.TolX = 10 * eps;
            options.ObjectiveLimit = -inf;
            options.GradObj = "on";

            if obj.algorithm == "trust-region"
                options.Algorithm = obj.algorithm;
                options.Hessian = "on";
                options.SubproblemAlgorithm = "factorization";
            elseif obj.algorithm == "trust-region-cg"
                options.Algorithm = "trust-region";
                options.Hessian = "on";
                options.SubproblemAlgorithm = "cg";
            else
                options.Algorithm = "quasi-newton";
                options.HessUpdate = obj.algorithm;
            end

            if obj.verbosity > 1
                options.Display = "iter-detailed";
            elseif obj.verbosity > 0
                options.Display = "iter";
            else
                options.Display = "off";
            end
        end

        function [status, x, history] = run(obj, f_handle, x0, optional)
            % Minimizes a given nonlinear function using fminunc

            arguments (Input)
                obj

                f_handle (1, 1) function_handle
                % A handle to evaluate the function value and derivatives of
                % the objective function, up to 2 depending on the number of
                % output arguments

                x0 (:, 1) double
                % The initial point

                optional.monitor experiment.shared.Monitor
                % An optional Monitor object to track the progress of the
                % optimization

                optional.wandb py.module
                % A reference to the python module `wandb` to track metrics
                % during the run
            end

            arguments (Output)
                status (1, 1) Optimization_Status
                % Informs the user whether the optimization terminated
                % successfully or which reason prevented the function from
                % doing so. See Optimization_Status for details.

                x (:, 1) double
                % The final iterate, an approximate minimizer

                history (1, :) struct
                % A table recording various metrics as the algorithm progresses
            end

            run = Fminunc_Run(f_handle, x0 + obj.x0_shift, obj, optional);
            run.run();
            status = run.status;
            x = run.x;
            history = run.history;
        end

    end
end
