prob = 21;
[n, m, name, x_0, status] = mgh('initial', prob);
[f, status] = mgh('eval_f', x_0);
[g, status] = mgh('eval_g', x_0);
[h, status] = mgh('eval_h', x_0);
[t, status] = mgh('eval_t', x_0);
[status] = mgh('final');
