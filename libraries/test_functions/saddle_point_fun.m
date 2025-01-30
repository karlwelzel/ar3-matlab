function [fun, der1f, der2f, der3f] = saddle_point_fun(x)
    assert(length(x) == 2);

    q = floor(x(1) / (2 * pi));
    r = mod(x(1), 2 * pi);

    added_slope = 1e-1;

    [fun, der1f, der2f, der3f] = inner([r, x(2) + q * pi]);
    fun = fun - q * pi - added_slope * x(1);
    der1f(1) = der1f(1) - added_slope;
end

function [fun, der1f, der2f, der3f] = inner(x)
    fun_part1 = (sin(x(1)) - x(1));
    der1_part1 = cos(x(1)) - 1;
    der2_part1 = -sin(x(1));
    der3_part1 = -cos(x(1));

    fun_part2 = (2 * sin(x(2)) + pi) / (2 * pi);
    der1_part2 = cos(x(2)) / pi;
    der2_part2 = -sin(x(2)) / pi;
    der3_part2 = -cos(x(2)) / pi;

    fun = sin(x(2)) + fun_part1 * fun_part2;

    der1f = zeros(2, 1);
    der1f(1) = der1_part1 * fun_part2;
    der1f(2) = fun_part1 * der1_part2 + cos(x(2));

    der2f = zeros(2, 2);
    der2f(1, 1) = der2_part1 * fun_part2;
    der2f(1, 2) = der1_part1 * der1_part2;
    der2f(2, 1) = der1_part1 * der1_part2;
    der2f(2, 2) = fun_part1 * der2_part2 - sin(x(2));

    der3f = zeros(2, 2, 2);
    der3f(1, 1, 1) = der3_part1 * fun_part2;
    der3f(1, 1, 2) = der2_part1 * der1_part2;
    der3f(1, 2, 1) = der2_part1 * der1_part2;
    der3f(1, 2, 2) = der1_part1 * der2_part2;
    der3f(2, 1, 1) = der2_part1 * der1_part2;
    der3f(2, 1, 2) = der1_part1 * der2_part2;
    der3f(2, 2, 1) = der1_part1 * der2_part2;
    der3f(2, 2, 2) = fun_part1 * der3_part2 - cos(x(2));
end
