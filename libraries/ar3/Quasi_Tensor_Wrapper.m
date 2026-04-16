classdef Quasi_Tensor_Wrapper < handle
    properties
        f_handle (1, 1) function_handle = @(x) 0
        parameters (1, 1) Quasi_Tensor_Parameters
        approximation double = nan
        last_iterate (:, 1) double = nan
        last_derivative double = nan
        last_gradient double = nan
    end

    methods

        function obj = Quasi_Tensor_Wrapper(f_handle, parameters)
            obj.f_handle = f_handle;
            obj.parameters = parameters;
        end

        function update_approximation(obj, current_iterate, current_derivative, current_gradient)
            n = length(current_iterate);
            p = obj.parameters.p;

            for k = 1
                if isscalar(obj.approximation) && isnan(obj.approximation)
                    obj.approximation = zeros(repmat(n, 1, p));
                    break
                end

                if obj.parameters.type == Quasi_Tensor_Type.CONSTANT
                    break
                end

                step = current_iterate - obj.last_iterate;
                step_norm = norm(step);
                normed_step = step / step_norm;

                predicted_diff = tensorprod(obj.approximation, normed_step, 1);
                exact_diff = (current_derivative - obj.last_derivative) / step_norm;
                correction_term = predicted_diff - exact_diff;

                diff_error_bound = sqrt(2) * eps * ...
                  (norm(current_derivative, "fro") + norm(obj.last_derivative, "fro")) / step_norm;
                if diff_error_bound > obj.parameters.numerical_error_threshold * norm(exact_diff, "fro")
                    % skip update if update is dominated by numerical errors
                    printf("skip quasi-tensor update: %e errors vs %e diff norm", diff_error_bound, norm(exact_diff, "fro"));
                    break
                end

                switch obj.parameters.type
                    case Quasi_Tensor_Type.POWELL_SYMMETRIC_BROYDEN
                        weighted_step = normed_step;
                    case Quasi_Tensor_Type.DAVIDON_FLETCHER_POWELL
                        weighted_step = (current_gradient - obj.last_gradient) / step_norm;
                    case Quasi_Tensor_Type.SYMMETRIC_RANK_ONE_LIKE
                        if p == 2
                            weighted_step = correction_term / norm(correction_term);
                        elseif p == 3
                            [weighted_step, ~] = eigs(correction_term, 1, "largestabs");
                        end
                end

                if abs(weighted_step' * normed_step) < obj.parameters.orthogonality_threshold
                    % skip update if weighted_step is almost orthogonal to normed_step
                    printf("skip quasi-tensor update: %e orthogonality", abs(weighted_step' * normed_step));
                    break
                end

                update = zeros(size(obj.approximation));
                for j = 1:p
                    weighted_step_outer = tensor_outer_product(weighted_step, j);
                    correction_inner = tensor_inner_product(correction_term, normed_step, j - 1);
                    update = update + (-1)^j * nchoosek(p, j) * (weighted_step' * normed_step)^(-j) * ...
                      tensorprod(weighted_step_outer, correction_inner, NumDimensionsA = j);
                end

                % Symmetrize update using approach from tensor toolbox:
                % https://gitlab.com/tensors/tensor_toolbox/-/blob/v3.6/@tensor/symmetrize.m
                sz = size(update);
                if p == 2
                    [index1, index2] = ind2sub(sz, 1:numel(update));
                    index = sort([index1; index2]);
                    linindex2symindex = sub2ind(sz, index(1, :), index(2, :));
                elseif p == 3
                    [index1, index2, index3] = ind2sub(sz, 1:numel(update));
                    index = sort([index1; index2; index3]);
                    linindex2symindex = sub2ind(sz, index(1, :), index(2, :), index(3, :));
                end
                average = accumarray(linindex2symindex', update(:)) ./ accumarray(linindex2symindex', 1);
                update = reshape(average(linindex2symindex), size(update));

                obj.approximation = obj.approximation + update;

                assert(all(ismembertol(tensorprod(obj.approximation, normed_step, 1), exact_diff), "all"));
            end

            obj.last_iterate = current_iterate;
            obj.last_derivative = current_derivative;
            obj.last_gradient = current_gradient;
        end

        function [fun, der1f, der2f, der3f, exact_der3f] = evaluate(obj, x)
            arguments (Input)
                obj

                x (:, 1) double
            end

            if nargout == 1
                [fun] = obj.f_handle(x);
            elseif nargout == 2
                [fun, der1f] = obj.f_handle(x);
            elseif nargout == 3
                if obj.parameters.p == 2
                    [fun, der1f] = obj.f_handle(x);
                    obj.update_approximation(x, der1f, der1f);
                    der2f = obj.approximation;
                else
                    [fun, der1f, der2f] = obj.f_handle(x);
                end
            elseif nargout == 4
                if obj.parameters.p == 3
                    [fun, der1f, der2f] = obj.f_handle(x);
                    obj.update_approximation(x, der2f, der1f);
                    der3f = obj.approximation;
                else
                    [fun, der1f, der2f, der3f] = obj.f_handle(x);
                end
            elseif nargout == 5 && obj.parameters.p == 3
                [fun, der1f, der2f, exact_der3f] = obj.f_handle(x);
                obj.update_approximation(x, der2f, der1f);
                der3f = obj.approximation;
                fprintf("rel. error            = %f\n", norm(der3f - exact_der3f, "fro") / norm(exact_der3f, "fro"));
            end
        end

    end
end

function prod = tensor_outer_product(v, repeat)
    arguments (Input)
        v (:, 1) double
        repeat (1, 1) double {mustBeInteger, mustBePositive}
    end

    arguments (Output)
        prod double
    end

    prod = v;
    for i = 2:repeat
        % Compute outer product between prod and v and assign to prod
        prod = tensorprod(prod, v);
    end

    % Remove singleton dimensions, they interact weirdly with tensorprod
    prod = squeeze(prod);
end

function prod = tensor_inner_product(t, v, repeat)
    arguments (Input)
        t double
        v (:, 1) double
        repeat (1, 1) double {mustBeInteger, mustBeNonnegative}
    end

    arguments (Output)
        prod double
    end

    prod = t;
    for i = 1:repeat
        prod = tensorprod(prod, v, 1);
    end
end
