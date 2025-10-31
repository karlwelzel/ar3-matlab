function [n, m, name, x0, f_handle] = mgh_function(problem_number, dimension)
    % Retrieve function from the More, Garbow, and Hillstrom test set

    arguments (Input)
        problem_number (1, 1) {mustBeInteger, mustBePositive}
        % The problem number between 1 and 35

        dimension (1, 1) {mustBeInteger} = -1
        % The problem dimension
    end

    arguments (Output)
        n (1, 1) {mustBeInteger}
        % The number of variables of the problem

        m (1, 1) {mustBeInteger}
        % The number of terms for nonlinear least-squares problems

        name (1, 1) string
        % The nme of the problem

        x0 (:, 1) double
        % The default starting point of the optimization problem

        f_handle function_handle
        % A function handle to evaluate the function and its derivatives
    end

    if dimension < 0
        [n, m, name, x0, status] = mgh_wrapper('initial', problem_number);
    else
        [n, m, name, x0, status] = mgh_wrapper('initial', problem_number, dimension);
    end

    throw_mgh_error(status);

    f_handle = @derivatives;
end

function [f, der1f, der2f, der3f] = derivatives(x)
    [f, status] = mgh_wrapper('eval_f', x);
    throw_mgh_error(status);

    if nargout > 1
        [der1f, status] = mgh_wrapper('eval_g', x);
        throw_mgh_error(status);
    end

    if nargout > 2
        [der2f, status] = mgh_wrapper('eval_h', x);
        throw_mgh_error(status);
    end

    if nargout > 3
        [der3f, status] = mgh_wrapper('eval_t', x);
        throw_mgh_error(status);
    end
end

function throw_mgh_error(status)
    switch status
        case 0
            % Success, don't do anything
        case -1
            throw(MException('MGH:InvalidProblemNumber', ...
                             'Problem number out of range [1, 35].'));
        case -2
            throw(MException('MGH:InputArrayMissing', ...
                             'Input array not set.'));
        case -3
            throw(MException('MGH:EvaluationError', ...
                             'Error while evaluating derivatives.'));
        case -4
            throw(MException('MGH:SpaceAllocationError', ...
                             'Error while allocating space for derivatives.'));
        case -5
            throw(MException('MGH:MissingInitialization', ...
                             "Missing call mgh_wrapper('initial', problem)."));
        otherwise
            throw(MException('MGH:GenericError', ...
                             'Something has gone wrong.'));
    end
end
