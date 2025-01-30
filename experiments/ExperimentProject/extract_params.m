function [selected_params, remaining_params] = extract_params(params, prefix)
    % Extracts all fields that start with prefix from the struct params and
    % saves them with the prefix stripped to selected_params. All remaining
    % fields are copied as-is to remaining_params.

    arguments (Input)
        params struct
        prefix string
    end

    arguments (Output)
        selected_params struct
        remaining_params struct
    end

    selected_params = struct;
    remaining_params = struct;

    for name = fieldnames(params)'
        if startsWith(name{1}, prefix)
            shortened_name = regexprep(name{1}, "^" + prefix, "");
            selected_params.(shortened_name) = params.(name{1});
        else
            remaining_params.(name{1}) = params.(name{1});
        end
    end
end
