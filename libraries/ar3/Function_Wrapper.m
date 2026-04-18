classdef Function_Wrapper < handle
    properties
        f_handle (1, 1) function_handle = @(x) 0
        eval_counter (1, 4) double = [0 0 0 0]
        last_iterate (:, 1) double = nan
        last_iterate_increase (1, 4) double = [0 0 0 0]
        last_tentative_iterate (:, 1) double = nan
        last_tentative_iterate_increase (1, 4) double = [0 0 0 0]
    end

    methods

        function obj = Function_Wrapper(f_handle)
            obj.f_handle = f_handle;
        end

        function [fun, der1f, der2f, der3f] = evaluate(obj, x, access_type)
            arguments (Input)
                obj

                x (:, 1) double
                access_type (1, 1) Function_Access_Type
            end

            if nargout == 1
                [fun] = obj.f_handle(x);
            elseif nargout == 2
                [fun, der1f] = obj.f_handle(x);
            elseif nargout == 3
                [fun, der1f, der2f] = obj.f_handle(x);
            elseif nargout == 4
                [fun, der1f, der2f, der3f] = obj.f_handle(x);
            end

            eval_counter_increase = 1:4 <= nargout;

            if access_type == Function_Access_Type.RAW
                return
            end

            if x == obj.last_iterate
                obj.eval_counter = obj.eval_counter - obj.last_iterate_increase;
                eval_counter_increase = max(eval_counter_increase, obj.last_iterate_increase);
            elseif x == obj.last_tentative_iterate
                obj.eval_counter = obj.eval_counter - obj.last_tentative_iterate_increase;
                eval_counter_increase = max(eval_counter_increase, obj.last_tentative_iterate_increase);
            end
            obj.eval_counter = obj.eval_counter + eval_counter_increase;

            switch access_type
                case Function_Access_Type.TENTATIVE_ITERATE
                    obj.last_tentative_iterate = x;
                    obj.last_tentative_iterate_increase = eval_counter_increase;
                case Function_Access_Type.NEW_ITERATE
                    obj.last_iterate = x;
                    obj.last_iterate_increase = eval_counter_increase;
            end
        end

    end
end
