function output = training(params, monitor)
    % Ensure reproducibility of randomly generated problems
    rng(123456789);

    % Save all params
    all_params = params;

    % Extract wandb parameters
    [wandb_params, params] = extract_params(params, "wandb_");

    % Determine objective function
    problem = params.problem;
    params = rmfield(params, "problem");
    [default_x0, f_handle] = params2problem(problem);
    dim = length(default_x0);
    if isfield(params, "x0_type")
        if isnumeric(params.x0_type)
            x0 = params.x0_type;
        elseif strcmp(params.x0_type, "default")
            x0 = default_x0;
        elseif strcmp(params.x0_type, "randn")
            x0 = randn(dim, 1);
        elseif strcmp(params.x0_type, "rand")
            x0 = rand(dim, 1);
        elseif strcmp(params.x0_type, "uniform")
            x0 = ones(dim, 1) / dim;
        else
            error("Unknown x0 type " + params.x0_type);
        end
        params = rmfield(params, "x0_type");
    end

    n = size(x0, 1);
    params.inner_stop_theta = 10^(log2(n)/4);
    if params.p == 3
        params.inner_inner_stop_theta = 10^(log2(n)/2);
    end
    
    % Determine subproblem solver and AR3 parameters
    arp_parameters = ARP_Parameters.from_struct(params);

    % Run ARp algorithm
    if nargin > 1
        [status, x, history] = arp_parameters.run(f_handle, x0, monitor = monitor);
    else
        [status, x, history] = arp_parameters.run(f_handle, x0);
    end

    disp("Experiment terminated with status " + string(status));

    if nargin > 1
        output = {all_params, status, x, history};
    else
        output = history;
    end
end
