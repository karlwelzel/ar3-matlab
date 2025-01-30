function r = real_roots(p)
    r = roots(p);
    r = r(imag(r) == 0);
end
