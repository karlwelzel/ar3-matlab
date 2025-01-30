for p = 2:3
    disp("Slalom Experiment with p = " + int2str(p));
    disp("============================");
    disp("");

    history = training(struct( ...
                              p = p, ...
                              problem = '{"name": "slalom", "x0": [0.5, 0]}', ...
                              x0_type = "default", ...
                              update_sigma0 = 50, ...
                              update_type = "Simple", ...
                              update_use_prerejection = false, ...
                              update_sigma_decrease = 1, ...
                              stop_rule = "Function_Value", ...
                              stop_f_threshold = -3.45, ...
                              inner_solver = "fminunc_steepdesc", ...
                              inner_x0_shift = [-0.1; 0], ...
                              inner_stop_rule = "First_Order", ...
                              inner_stop_tolerance_g = 1e-9, ...
                              inner_stop_max_iterations = 1000 ...
                             ));

    disp(struct2table(history));
end

for p = 2:3
    disp("Hairpin Turn Experiment with p = " + int2str(p));
    disp("==================================");
    disp("");

    history = training(struct( ...
                              p = p, ...
                              problem = '{"name": "hairpin_turn", "x0": [0.5, 0]}', ...
                              x0_type = "default", ...
                              update_sigma0 = 50, ...
                              update_type = "Simple", ...
                              update_use_prerejection = false, ...
                              update_sigma_decrease = 1, ...
                              stop_rule = "First_Order", ...
                              stop_tolerance_g = 1e-7, ...
                              inner_solver = "fminunc_steepdesc", ...
                              inner_x0_shift = [-0.1; 0], ...
                              inner_stop_rule = "First_Order", ...
                              inner_stop_tolerance_g = 1e-9, ...
                              inner_stop_max_iterations = 1000 ...
                             ));

    disp(struct2table(history));
end
