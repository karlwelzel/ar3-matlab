classdef (Abstract) Optimization_Parameters < Parameters
    % Stores parameters for a run of an optimization algorithm

    properties
        verbosity (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
        x0_shift (:, 1) double = 0
    end

    methods (Static)

        function obj = from_struct(params)
            solver = params.solver;
            params = rmfield(params, "solver");
            if solver == "ARP"
                obj = ARP_Parameters.from_struct(params);
            elseif any(solver == ["AR2", "ARC"])
                obj = ARP_Parameters.from_struct(params);
                obj.p = 2;
            elseif solver == "AR3"
                obj = ARP_Parameters.from_struct(params);
                obj.p = 3;
            elseif solver == "MCMR"
                obj = MCMR_Parameters.from_struct(params);
            elseif solver == "QQR"
                obj = QQR_Parameters.from_struct(params);
            elseif startsWith(solver, "fminunc_")
                obj = Fminunc_Parameters.from_struct(params);
                obj.algorithm = regexprep(solver, "^fminunc_", "");
            else
                error("Unknown subproblem solver: " + solver);
            end
        end

    end
end
