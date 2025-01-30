classdef QQR_Parameters < Optimization_Parameters
    % Stores parameters for a run of the QQR optimization algorithm

    properties
        lambda_c (1, 1) {mustBePositive} = 1e-2
        successful_cutoff (1, 1) double {mustBePositive, lt(successful_cutoff, 1)} = 0.1 % rho_1
        hessian_increase (1, 1) double {mustBePositive, gt(hessian_increase, 1)} = 2 % eta_1
        regularization_increase (1, 1) double {gt(regularization_increase, 1)} = 2 % eta_2
        hessian_perturbation (1, 1) {mustBePositive} = 1e-2  % \tilde{p} if perturbation necessary
        subproblem_parameters (1, 1) Optimization_Parameters = MCMR_Parameters
        termination_rule (1, 1) Termination_Rule = First_Order_Termination_Rule
    end

    properties (Constant)
        p (1, 1) double {mustBeInteger} = 2 % constant for ARP_Theory termination condition
    end

    methods (Static)

        function obj = from_struct(params)
            obj = QQR_Parameters;

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

            if isfield(params, "inner_solver")
                [subproblem_params, params] = extract_params(params, "inner_");
                obj.subproblem_parameters = Optimization_Parameters.from_struct(subproblem_params);
            end

            if isfield(params, "stop_rule")
                [termination_params, params] = extract_params(params, "stop_");
                obj.termination_rule = Termination_Rule.from_struct(termination_params);
            end

            obj.update(params);
        end

    end

    methods

        function [status, x, history] = run(obj, f_handle, sigma, x0, optional)
            % Minimizes a given AR3 model function using QQR

            arguments (Input)
                obj

                f_handle (1, 1) function_handle
                % A handle to evaluate the function value and derivatives of
                % the AR3 model function, up to 2 depending on the number of
                % output arguments

                sigma (1, 1) double
                % The regularization parameter of the AR3 model

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

            run = QQR_Run(f_handle, sigma, x0 + obj.x0_shift, obj, optional);
            run.run();
            status = run.status;
            x = run.x;
            history = run.history;
        end

    end
end
