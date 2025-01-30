function [fun, der1f, der2f, der3f] = hairpin_turn(x)
    [fun, der1f, der2f, der3f] = step(x(1), x(2));

    barrier_mult = 50;
    barrier_box = [-0.4, 0.5, -5, 0]; % x1, x2, y1, y2
    if x(1) <= barrier_box(1)
        fun = fun + barrier_mult * (x(1) - barrier_box(1))^4;
        der1f(1) = der1f(1) + barrier_mult * 4 * (x(1) - barrier_box(1))^3;
        der2f(1, 1) = der2f(1, 1) + barrier_mult * 12 * (x(1) - barrier_box(1))^2;
        der3f(1, 1, 1) = der3f(1, 1, 1) + barrier_mult * 24 * (x(1) - barrier_box(1));
    elseif x(1) >= barrier_box(2)
        fun = fun + barrier_mult * (x(1) - barrier_box(2))^4;
        der1f(1) = der1f(1) + barrier_mult * 4 * (x(1) - barrier_box(2))^3;
        der2f(1, 1) = der2f(1, 1) + barrier_mult * 12 * (x(1) - barrier_box(2))^2;
        der3f(1, 1, 1) = der3f(1, 1, 1) + barrier_mult * 24 * (x(1) - barrier_box(2));
    end

    if x(2) <= barrier_box(3)
        fun = fun + barrier_mult * (x(2) - barrier_box(3))^4;
        der1f(2) = der1f(2) + barrier_mult * 4 * (x(2) - barrier_box(3))^3;
        der2f(2, 2) = der2f(2, 2) + barrier_mult * 12 * (x(2) - barrier_box(3))^2;
        der3f(2, 2, 2) = der3f(2, 2, 2) + barrier_mult * 24 * (x(2) - barrier_box(3));
    elseif x(2) >= barrier_box(4)
        fun = fun + barrier_mult * (x(2) - barrier_box(4))^4;
        der1f(2) = der1f(2) + barrier_mult * 4 * (x(2) - barrier_box(4))^3;
        der2f(2, 2) = der2f(2, 2) + barrier_mult * 12 * (x(2) - barrier_box(4))^2;
        der3f(2, 2, 2) = der3f(2, 2, 2) + barrier_mult * 24 * (x(2) - barrier_box(4));
    end

    fun = fun + 3e-4 * x(1);
    der1f(1) = der1f(1) + 3e-4;
end

function [fun, der1f, der2f, der3f] = step(x, y)
    [sig, der1sig, der2sig, der3sig] = sigmoid_fun(y);
    [s, der1s, der2s, der3s] = scaling(x, 2 * sig);
    [w, der1w, der2w, der3w] = wiggly(x);

    fun = w * s;

    der1f = zeros(2, 1);
    der1f(1) = der1w * s + w * der1s(1);
    der1f(2) = w * der1s(2) * 2 * der1sig;

    der2f = zeros(2, 2);
    der2f(1, 1) = der2w * s + 2 * der1w * der1s(1) + w * der2s(1, 1);
    der2f(1, 2) = (der1w * der1s(2) + w * der2s(1, 2)) * 2 * der1sig;
    der2f(2, 1) = (der1w * der1s(2) + w * der2s(1, 2)) * 2 * der1sig;
    der2f(2, 2) = w * (der2s(2, 2) * 2^2 * der1sig^2 + der1s(2) * 2 * der2sig);

    der3f = zeros(2, 2, 2);
    der3f(1, 1, 1) = der3w * s + 3 * der2w * der1s(1) + 3 * der1w * der2s(1, 1) + w * der3s(1, 1, 1);
    der3f(1, 1, 2) = (der2w * der1s(2) + 2 * der1w * der2s(1, 2) + w * der3s(1, 1, 2)) * 2 * der1sig;
    der3f(1, 2, 1) = der3f(1, 1, 2);
    der3f(1, 2, 2) = der1w * (der2s(2, 2) * 2^2 * der1sig^2 + der1s(2) * 2 * der2sig) + ...
      w * (der3s(1, 2, 2) * 2^2 * der1sig^2 + der2s(1, 2) * 2 * der2sig);
    der3f(2, 1, 1) = der3f(1, 1, 2);
    der3f(2, 1, 2) = der3f(1, 2, 2);
    der3f(2, 2, 1) = der3f(1, 2, 2);
    der3f(2, 2, 2) = w * (der3s(2, 2, 2) * 2^3 * der1sig^3 + 3 * der2s(2, 2) * 2^2 * der1sig * der2sig + ...
                          der1s(2) * 2 * der3sig);
end

function [fun, der1f, der2f, der3f] = scaling(x, y)
    if x < -1
        fun = 1;
        der1f = zeros(2, 1);
        der2f = zeros(2, 2);
        der3f = zeros(2, 2, 2);
        return
    elseif x <= -0.5
        xtilde = 2 * x + 2;
    elseif x < 0.5
        fun = y;
        der1f = [0; 1];
        der2f = zeros(2, 2);
        der3f = zeros(2, 2, 2);
        return
    elseif x <= 1
        xtilde = 2 * x - 1;
    else
        fun = 1;
        der1f = zeros(2, 1);
        der2f = zeros(2, 2);
        der3f = zeros(2, 2, 2);
        return
    end

    [h, der1h, der2h, der3h] = scaling_helper(xtilde);
    if x <= -0.5
        fun = -h * (1 - y) + 1;

        der1f = zeros(2, 1);
        der1f(1) = -der1h * 2 * (1 - y);
        der1f(2) = h;

        der2f = zeros(2, 2);
        der2f(1, 1) = -der2h * 4 * (1 - y);
        der2f(1, 2) = der1h * 2;
        der2f(2, 1) = der1h * 2;
        der2f(2, 2) = 0;

        der3f = zeros(2, 2, 2);
        der3f(1, 1, 1) = -der3h * 8 * (1 - y);
        der3f(1, 1, 2) = der2h * 4;
        der3f(1, 2, 1) = der2h * 4;
        der3f(1, 2, 2) = 0;
        der3f(2, 1, 1) = der2h * 4;
        der3f(2, 1, 2) = 0;
        der3f(2, 2, 1) = 0;
        der3f(2, 2, 2) = 0;
    else
        fun = h * (1 - y) + y;

        der1f = zeros(2, 1);
        der1f(1) = der1h * 2 * (1 - y);
        der1f(2) = -h + 1;

        der2f = zeros(2, 2);
        der2f(1, 1) = der2h * 4 * (1 - y);
        der2f(1, 2) = -der1h * 2;
        der2f(2, 1) = -der1h * 2;
        der2f(2, 2) = 0;

        der3f = zeros(2, 2, 2);
        der3f(1, 1, 1) = der3h * 8 * (1 - y);
        der3f(1, 1, 2) = -der2h * 4;
        der3f(1, 2, 1) = -der2h * 4;
        der3f(1, 2, 2) = 0;
        der3f(2, 1, 1) = -der2h * 4;
        der3f(2, 1, 2) = 0;
        der3f(2, 2, 1) = 0;
        der3f(2, 2, 2) = 0;
    end
end

function [fun, der1f, der2f, der3f] = wiggly(x)
    [fun, der1f, der2f, der3f] = wiggly_helper(mod(x + 0.5, 1));
    fun = fun + floor(x + 0.5) - 0.5;
end

function [fun, der1f, der2f, der3f] = sigmoid_fun(x)
    expminusx = exp(-x);
    fun = 1 / (1 + expminusx);
    der1f = expminusx / (1 + expminusx)^2;
    der2f = (-expminusx * (1 - expminusx)) / (1 + expminusx)^3;
    der3f = (expminusx - 4 * expminusx^2 + expminusx^3) / (1 + expminusx)^4;
end

function [fun, der1f, der2f, der3f] = scaling_helper(x)
    poly = [-20, 70, -84, 35, 0, 0, 0, 0];
    fun = polyval(poly, x);
    poly = polyder(poly);
    der1f = polyval(poly, x);
    poly = polyder(poly);
    der2f = polyval(poly, x);
    poly = polyder(poly);
    der3f = polyval(poly, x);
end

function [fun, der1f, der2f, der3f] = wiggly_helper(x)
    poly = [6, -15, 10, 0, 0, 0];
    fun = polyval(poly, x);
    poly = polyder(poly);
    der1f = polyval(poly, x);
    poly = polyder(poly);
    der2f = polyval(poly, x);
    poly = polyder(poly);
    der3f = polyval(poly, x);
end
