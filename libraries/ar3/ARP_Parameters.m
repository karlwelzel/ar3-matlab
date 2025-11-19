classdef ARP_Parameters < Optimization_Parameters
    % Stores parameters for a run of the ARp (p=2/3) optimization algorithm

    % This parameter class groups parameters and stores them in their own
    % parameter objects. This makes it possible to nest multiple instances of
    % this class. For example parameters that describe an AR3 run which uses
    % AR2 as the subproblem solver, would lead to an ARP_Parameters object
    % whose subproblem_parameters attribute is also an ARP_Parameters object.

    properties
        p (1, 1) double {mustBeInteger, mustBeGreaterThan(p, 0), mustBeLessThan(p, 4)} = 2
        sigma_update_parameters (1, 1) Sigma_Update_Parameters = Simple_Update_Parameters
        subproblem_parameters (1, 1) Optimization_Parameters = MCMR_Parameters
        termination_rule (1, 1) Termination_Rule = First_Order_Termination_Rule
    end

    methods (Static)

        function obj = from_struct(params)
            obj = ARP_Parameters;

            if isfield(params, "p")
                obj.p = params.p;
                params = rmfield(params, "p");
            end

            if isfield(params, "verbosity")
                obj.verbosity = params.verbosity;
                params = rmfield(params, "verbosity");
            end

            if isfield(params, "x0_shift")
                obj.x0_shift = params.x0_shift;
                params = rmfield(params, "x0_shift");
            end

            if isfield(params, "update_type")
                [sigma_update_params, params] = extract_params(params, "update_");
                obj.sigma_update_parameters = Sigma_Update_Parameters.from_struct(sigma_update_params);
            end

            if obj.p < 3 && ~isa(obj.sigma_update_parameters, "BGMS_Update_Parameters")
                % Do not use prerejection for p < 3, except for BGMS
                obj.sigma_update_parameters.use_prerejection = false;
            end

            if isfield(params, "inner_solver")
                [subproblem_params, params] = extract_params(params, "inner_");
                obj.subproblem_parameters = Optimization_Parameters.from_struct(subproblem_params);
                if isa(obj.subproblem_parameters, "ARP_Parameters")
                    % Improve numerical stability by enabling assume_decrease for the subproblem solver
                    obj.subproblem_parameters.sigma_update_parameters.assume_decrease = true;
                end
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

        function [status, x, history] = run(obj, f_handle, x0, optional)
            % Minimizes a given nonlinear function using the ARp algorithm

            arguments (Input)
                obj

                f_handle (1, 1) function_handle
                % A handle to evaluate the function value and derivatives of
                % the objective function, up to 3 depending on the number of
                % output arguments

                x0 (:, 1) double
                % The initial point

                optional.monitor experiment.shared.Monitor
                % An optional Monitor object to track the progress of the
                % optimization

                optional.wandb py.module
                % A reference to the python module `wandb` to track metrics
                % during the run, also optional
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

            run = ARP_Run(f_handle, x0 + obj.x0_shift, obj, optional);
            run.run();
            status = run.status;
            x = run.x;
            history = run.history;
        end

    end
end
