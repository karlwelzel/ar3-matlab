function der = padded_polyder(p)
    % same as polyder but always of length width(p)-1
    der = polyder(p);
    der = [zeros(1, length(p) - 1 - length(der)), der];
end
