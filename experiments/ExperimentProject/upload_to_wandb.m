function upload_to_wandb(results_folder)
    wandb = py.importlib.import_module("wandb");

    files = dir(results_folder);
    for i = 3:length(files)
        if files(i).isdir && startsWith(files(i).name, "Trial")
            output_file = fullfile(files(i).folder, files(i).name, "output.mat");
            disp(output_file);

            try
                [params, status, ~, history] = load(output_file).outputs{:};
            catch exception
                switch exception.identifier
                    case 'MATLAB:load:couldNotReadFile'
                        continue
                    otherwise
                        rethrow(exception);
                end
            end

            wandb.init(project = params.wandb_project, ...
                       group = params.wandb_group, ...
                       config = py.dict(params));

            for j = 1:length(history)
                wandb.log(py.dict(history(j)));
            end

            if status == Optimization_Status.SUCCESS
                wandb.finish(quiet = true);
            else
                wandb.finish(int32(status), quiet = true);
            end
        end
    end
end
