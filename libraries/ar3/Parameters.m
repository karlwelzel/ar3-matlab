classdef (Abstract) Parameters
    % Base class of all parameter classes

    methods

        function obj = update(obj, params)
            arguments
                obj
                params struct
            end

            % Copy all parameters from struct to the object
            if ~isempty(fieldnames(params))
                for name = fieldnames(params)'
                    obj.(name{1}) = params.(name{1});
                end
            end
        end

    end
end
