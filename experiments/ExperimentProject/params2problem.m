function [x0, f_handle] = params2problem(problem)
    % Returns the initial point and a function handle for the requested
    % optimization problem. If problem is an integer (a), it uses the
    % corresponding problem from the MGH problem set with the default initial
    % point. Otherwise it expects problem to be a string encoding the desired
    % optimization problem using JSON. The options then are (b) to construct a
    % default initial point given the problem dimension or (c) to specify the
    % initial point explicitly.
    %
    % Example inputs for the three different cases:
    %     (a) 5
    %     (b) {"name": "rosenbrock", "dim": 10}
    %     (c) {"name": "rosenbrock", "x0": [2, 3, 4, 5]}

    arguments (Input)
        problem
    end

    arguments (Output)
        x0 (:, 1) double
        f_handle function_handle
    end

    if isa(problem, "string") && ~isnan(str2double(problem))
        problem = str2double(problem);
    end

    if isnumeric(problem)
        [~, ~, ~, x0, f_handle] = mgh_function(problem);
    else
        objective = jsondecode(problem);

        if isfield(objective, "x0")
            x0 = objective.x0;
        elseif isfield(objective, "dim") % default x0
            x0 = zeros(objective.dim, 1);
        else
            error("No way of constructing x0 given");
        end
        n = length(x0);

        if isnumeric(objective.name)
            if isfield(objective, "x0")
                [~, ~, ~, ~, f_handle] = mgh_function(objective.name, length(x0));
            elseif isfield(objective, "dim")
                [~, ~, ~, x0, f_handle] = mgh_function(objective.name, objective.dim);
            end
        elseif objective.name == "chebysv_rosenbrock_matfree"
            f_handle = @(x) chebysv_rosenbrock_matfree(x);
            x0 = ones(objective.dim, 1);
            x0(1) = -1;
        elseif objective.name == "chebysv_rosenbrock"
            f_handle = @(x) chebysv_rosenbrock(x);
            x0 = ones(objective.dim, 1);
            x0(1) = -1;
        elseif objective.name == "nonlinear_least_squares_matfree"
            m = n;
            mat = rand(m, n);
            b = zeros(n, 1);
            b(1:floor(n / 2)) = 1;
            f_handle = @(x) nonlinear_least_squares_matfree(mat, b, x);
            x0 = zeros(objective.dim, 1);
        elseif objective.name == "nonlinear_least_squares"
            m = n;
            mat = rand(m, n);
            b = zeros(n, 1);
            b(1:floor(n / 2)) = 1;
            f_handle = @(x) nonlinear_least_squares(mat, b, x);
            x0 = zeros(objective.dim, 1);
        elseif startsWith(objective.name, "ill_cond")
            x0 = ones(objective.dim, 1); % T not well-defined at 0
            if objective.name == "ill_cond_bm_matfree"
                rng(0);
                f_handle = construct_regularized_cubic_matfree(n, 0, 0);
            elseif objective.name == "ill_cond_H_matfree"
                rng(1);
                f_handle = construct_regularized_cubic_matfree(n, 6, 0);
            elseif objective.name == "ill_cond_T_matfree"
                rng(2);
                f_handle = construct_regularized_cubic_matfree(n, 0, 6);
            elseif objective.name == "ill_cond_HT_matfree"
                rng(3);
                f_handle = construct_regularized_cubic_matfree(n, 6, 6);
            elseif objective.name == "ill_cond_bm"
                rng(0);
                f_handle = construct_regularized_cubic(n, 0, 0);
            elseif objective.name == "ill_cond_H"
                rng(1);
                f_handle = construct_regularized_cubic(n, 6, 0);
            elseif objective.name == "ill_cond_T"
                rng(2);
                f_handle = construct_regularized_cubic(n, 0, 6);
            elseif objective.name == "ill_cond_HT"
                rng(3);
                f_handle = construct_regularized_cubic(n, 6, 6);
            end
        else
            f_handle = str2func(objective.name);
        end
    end
end
