classdef GLRT_Parameters < Optimization_Parameters
    % GLRT_Parameters, GLRT subproblem in a Krylov/Lanczos subspace
    % Alg. 9.2.1 in [Cartis, Gould and Toint, 2022] Evaluation Complexity of
    % Algorithms for Nonconvex Optimization, Theory, Computation and Perspectives
    %   Build the Lanczos tridiagonal T at each subspace size t.
    %   For a given lambda, try L*D*L' factorization of (T + lambda*I):
    %       if a nonpositive pivot appears, DO NOT solve; instead
    %         raise a safe lower bound lambda_lower and retry.
    %   When SPD: solve (T + lambda*I) y = beta1*e1, form w via L u = y, w = u./sqrt(d).
    %   Define phi(lambda) = sigma*norm(y) - lambda.
    %       Use safeguarded Newton steps inside the current bracket
    %       [lambda_lower, lambda_upper], with fallback bracket steps.
    %   Lambda is kept inside the current bracket (no forced monotonicity).
    %
    % Notes:
    %   No "hard case" augmentation here.

    properties
        termination_rule (1, 1) Termination_Rule = Cartis_G_Termination_Rule
    end

    methods (Static)

        function obj = from_struct(params)
            obj = GLRT_Parameters;
            if isfield(params, "stop_rule")
                [termination_params, params] = extract_params(params, "stop_");
                obj.termination_rule = Termination_Rule.from_struct(termination_params);
            end
            remaining = fieldnames(params);
            if ~isempty(remaining)
                error("Unrecognized field: " + remaining{1});
            end
        end

    end

    methods

        function [status, best_x, t, sigma] = run(obj, ~, g, H, sigma)
            arguments
                obj
                ~
                g (:, 1) double
                H
                sigma (1, 1) double {mustBeNonnegative}
            end

            n = numel(g);

            % Right-hand side and its norm
            b = -g;
            beta1 = norm(b);
            betan = beta1;
            t = 0;

            % Lanczos process
            v_prev    = zeros(n, 1);
            beta_prev = beta1;
            v         = b / beta1;      % first Lanczos vector
            V         = v;              % reorthogonalized basis we build
            alpha_diag = [];
            beta_diag  = [];

            % Bracket and controls for lambda (CGT-GLRT 9.2.1 style)
            lam_lower     = 0.0;
            lam_upper     = Inf;
            clip_margin   = 1e-10;    % tiny interior margin when clipping in [L,U]
            max_escalate  = 30;       % cap doubling attempts if NPC persists
            theta         = 0.5;      % linear split parameter (0<theta<1)
            max_lifting   = 2;

            % Lambda-region status:
            % 'N' (indefinite), 'L' (less), 'G' (greater), 'E' (Escape)
            lambda_status = 'N';

            norm_g       = Inf;
            best_y       = zeros(1, 1);
            tol_lanczos = eps * n * max(1, beta1);

            while true
                % Lanczos step: only while Krylov subspace dimension <= n
                % (you can tighten this if you want: e.g., t < n && betan ~= 0)
                if t < n && betan ~= 0
                    % Lanczos step (before reorth)
                    [v_next, alfa, betan_raw] = GLRT_Parameters.lanczos(H, v, v_prev, beta_prev);
                    t = t + 1; % mat-vec multiplications

                    % Full reorthogonalization of v_next and normalization
                    if t == 1
                        v_next = v_next - (v' * v_next) * v;
                        % Initialize lambda (exact scalar 1D solution)
                        lamda = (sqrt(alfa^2 + 4 * beta1 * sigma) - alfa) / 2;
                        lamda = max(lamda, lam_lower);
                    else
                        v_next = v_next - V * (V' * v_next);
                    end

                    betan = norm(v_next);
                    if betan > tol_lanczos
                        v_next = v_next / betan;
                    else
                        % Subspace breakdown, stop expanding
                        betan  = 0;
                        v_next = zeros(n, 1);
                    end

                    % Grow basis and tridiagonal data
                    V           = [V, v_next];
                    alpha_diag = [alpha_diag; alfa];
                    beta_diag  = [beta_diag;  betan];

                    % Advance Lanczos state
                    v_prev    = v;
                    v         = v_next;
                    beta_prev = betan;

                    % New T (larger Krylov subspace): reset lambda-iteration counter
                    lambda_iters = 0;
                end

                lamda_lifting = 0;
                lambda_status = 'N';

                while lambda_status ~= 'E' % E for escape
                    % Try LDL at current t
                    [has_nc, l, d, lam_min_saf] = ...
                        GLRT_Parameters.ldl_tridiag(alpha_diag, beta_diag(1:end - 1), lamda);

                    if has_nc
                        % if NPC, raise lam_lower and retry
                        lambda_status = 'N';
                        lam_lower = max(lam_lower, lam_min_saf);

                        % Change 1: Reset lam_upper if lower bound contradicts it
                        % If the safe lower bound from LDL exceeds our current upper bound,
                        % the upper bound was likely invalid or we are in a noisy region.
                        if isfinite(lam_upper) && lam_lower >= lam_upper
                            lam_upper = Inf;
                        end

                        lamda     = max(lamda, lam_lower);
                        lamda_lifting    = lamda_lifting + 1;

                        if lamda_lifting >= max_lifting % try lifted lambda twice, doubling if fails
                            % bracket-aware escalation: double until SPD,
                            % but stay inside (lam_lower, lam_upper) if finite
                            success = false;
                            lam_try = max(lamda, max(1e-4, lam_lower));
                            for k = 1:max_escalate
                                if isfinite(lam_upper)
                                    % keep strictly inside the bracket
                                    left  = lam_lower + clip_margin * max(1, lam_upper - lam_lower);
                                    right = lam_upper - clip_margin * max(1, lam_upper - lam_lower);

                                    % propose a doubling lift; cap by right edge
                                    lam_try = min(max(2 * lam_try, left), right);

                                    % if no increase possible after capping, fall back to bracket step
                                    if lam_try < lamda + eps
                                        lam_geom = sqrt(lam_lower * lam_upper);
                                        lam_lin  = lam_lower + theta * (lam_upper - lam_lower);
                                        lam_try  = max(lam_geom, lam_lin);  % strictly inside by construction
                                    end
                                else
                                    % no finite upper bound yet, pure doubling is fine
                                    lam_try = max(2 * lam_try, lam_lower * (2.0^k));
                                end

                                [has_nc2, l2, d2, lam_min2] = ...
                                    GLRT_Parameters.ldl_tridiag(alpha_diag, beta_diag(1:end - 1), lam_try);

                                if ~has_nc2
                                    % T + lambda I is PSD, lambda in L or G
                                    l = l2;
                                    d = d2;
                                    lamda = lam_try;
                                    success = true;
                                    break
                                else
                                    lam_lower = max(lam_lower, lam_min2);
                                    % Ensure we don't break the upper bound during escalation
                                    if isfinite(lam_upper) && lam_lower >= lam_upper
                                        lam_upper = Inf;
                                    end
                                    lam_try   = max(lam_try, lam_lower);
                                end
                            end

                            if ~success
                                status = Optimization_Status.NUMERICAL_ISSUES;
                                % fall back to current y in this subspace
                                best_x = V(:, 1:numel(best_y)) * best_y;
                                return
                            end
                            % Found SPD via escalation, continue to SPD path below with l,d,lamda
                        else
                            % Try LDL again at the lifted lambda
                            continue
                        end
                    end
                    % Escaped with PSD T.

                    % -----------------------------------------------------------------
                    % SPD at current lambda: solve (T+lambda*I) y = beta1 e1, and build w
                    % -----------------------------------------------------------------
                    % Forward: L q = beta1 e1
                    q      = zeros(t, 1);
                    q(1) = beta1;
                    for i = 2:t
                        q(i) = -l(i - 1) * q(i - 1);
                    end
                    % Diagonal: D z = q
                    z      = q ./ d;
                    % Backward: L' y = z
                    y      = zeros(t, 1);
                    y(t) = z(t);
                    for i = t - 1:-1:1
                        y(i) = z(i) - l(i) * y(i + 1);
                    end

                    % Build w implicitly, via u, and get ||w||^2 = sum(u.^2 ./ d)
                    u      = zeros(t, 1);
                    u(1) = y(1);
                    for i = 2:t
                        u(i) = y(i) - l(i - 1) * u(i - 1);
                    end
                    wn2    = sum(u.^2 ./ max(d, eps)); % norm(w)^2

                    yn     = norm(y);
                    phi    = sigma * yn - lamda;    % secular residual in this subspace

                    % Residual measure passed to your termination rule (kept unchanged)
                    grad_m = phi * y;
                    grad_m(end) = grad_m(end) + betan * yn;
                    norm_gl = norm_g;
                    norm_g  = norm(grad_m);
                    if norm_g < norm_gl
                        best_y = y;
                    end

                    % Package run_info for generic termination rule
                    run_info = struct( ...
                                      "norm_g",   norm_g, ...
                                      "x",        y, ... % ||x|| = ||y||
                                      "iteration", t, ...
                                      "sigma",    sigma, ...
                                      "optional", struct() ...
                                     );

                    [terminate, status] = obj.termination_rule.should_terminate(run_info);
                    if terminate
                        best_x = V(:, 1:numel(best_y)) * best_y;
                        status = Optimization_Status.SUCCESS;
                        return
                    end

                    % Hard cap on lambda iterations
                    if lambda_iters >= max_escalate || abs(phi) <= 1e-10 + 1e-10 * max(1, lamda) || ...
                            lam_upper - lam_lower < 1e-10
                        if betan < eps
                            status = Optimization_Status.NUMERICAL_ISSUES;
                            best_x = V(:, 1:numel(y)) * y;
                            return
                        else
                            lambda_status = 'E';
                            break
                        end
                    end

                    % Update the bracket for lambda_t
                    if phi < 0
                        lambda_status = 'G';  % lambda too large (in G)
                        lam_upper = min(lam_upper, lamda);   % lambda too large, lower upper bound
                        % Change 2: Check for crossover
                    else
                        lambda_status = 'L';
                        lam_lower = max(lam_lower, lamda);   % lambda too small, lift lower bound
                        % Change 2: Check for crossover
                        % if lam_lower >= lam_upper
                        %     lam_upper = lam_lower + eps;
                        % end
                    end

                    if lam_upper <= lam_lower
                        % lam_lower = max(0, lam_upper - eps);
                        best_x = V(:, 1:numel(best_y)) * best_y;
                        status = Optimization_Status.NUMERICAL_ISSUES;
                        return
                    end

                    % -----------------------------------------------------------------
                    % One lambda update per t:
                    %   Newton step in both L and G, safeguarded by the bracket.
                    %   If Newton step not acceptable, fall back to bracket step.
                    % -----------------------------------------------------------------

                    % Count this as one lambda search iteration for the current T
                    lambda_iters = lambda_iters + 1;
                    derphi = -sigma * wn2 / max(yn, eps) - 1;
                    lambda_newton = lamda - phi / derphi;

                    % Bracket-based candidate (geometric/linear split)
                    if isfinite(lam_upper)
                        lam_geom = sqrt(lam_lower * lam_upper);
                        lam_lin  = lam_lower + theta * (lam_upper - lam_lower);
                        lam_br   = max(lam_geom, lam_lin);
                    else
                        % Robust expansion if no upper bound
                        lam_br = lam_lower + max(eps, clip_margin * max([1, abs(lamda), abs(lam_lower)]));
                    end

                    % Decide if Newton candidate is acceptable (strictly inside bracket)
                    if isfinite(lam_upper)
                        scale = max(eps, lam_upper - lam_lower); % Change 3: Ensure non-zero scale
                    else
                        scale = max(1, abs(lam_lower));
                    end
                    delta = clip_margin * scale;

                    if isfinite(lam_upper)
                        newton_inside = isfinite(lambda_newton) && ...
                                        (lambda_newton > lam_lower + delta) && ...
                                        (lambda_newton < lam_upper - delta);
                    else
                        newton_inside = isfinite(lambda_newton) && ...
                                        (lambda_newton > lam_lower + delta);
                    end

                    if newton_inside
                        lam_cand = lambda_newton;
                    else
                        lam_cand = lam_br;
                    end

                    % Final bracket clipping: keep lambda strictly inside (lam_lower, lam_upper)
                    if isfinite(lam_upper)
                        left  = lam_lower + clip_margin * max(eps, lam_upper - lam_lower);
                        right = lam_upper - clip_margin * max(eps, lam_upper - lam_lower);
                        % Ensure right > left in case of extreme closeness
                        if right <= left
                            % right = lam_upper;
                            % left = lam_lower;
                            best_x = V(:, 1:numel(best_y)) * best_y;
                            status = Optimization_Status.NUMERICAL_ISSUES;
                            return
                        end
                        lam_cand = min(max(lam_cand, left), right);
                    else
                        lam_cand = max(lam_cand, lam_lower + ...
                                       clip_margin * max(1, abs(lamda) - abs(lam_lower)));
                    end

                    lamda = lam_cand;

                    % One lambda update per t, proceed (either new Lanczos step if t<n,
                    % or another lambda-refinement step if t>=n).
                    break
                end
            end
        end

    end

    methods (Static)

        function [vn, alfa, betan] = lanczos(H, v, vp, beta)
            % One Lanczos step: vn (pre-reorth residual direction), alfa = v'*H*v, betan_raw = ||vn||.
            Av    = mat_vec(H, v);
            alfa = v' * Av;
            vn    = Av - v * alfa - vp * beta;
            betan = norm(vn);
        end

        function [has_nc, l, d, lam_min_saf] = ldl_tridiag(alpha, beta, lamda)
            % LDL^T detection for (T + lamda*I). Do NOT repair lambda here.
            % If any pivot small, return has_nc=true and a safe lower bound lam_min_saf.
            % T = diag(alpha) + diag(beta,1) + diag(beta,-1)
            t  = numel(alpha);
            l  = zeros(max(0, t - 1), 1);
            d  = zeros(t, 1);
            has_nc      = false;
            lam_min_saf = -Inf;

            sqe   = sqrt(eps);
            tau_d = 4 * sqe;  % relative threshold on pivots w.r.t. some local scale

            for i = 1:t
                if i == 1
                    d(1) = alpha(1) + lamda;
                    scale1 = max(1, abs(alpha(1)) + abs(lamda));
                    if d(1) <= tau_d * scale1 % numerical stability
                        lam_min       = -alpha(1);
                        lam_min_saf   = max(lam_min_saf, lam_min + 4 * sqe * scale1);
                        has_nc = true;
                        return
                    end
                else
                    % Guarded recurrence to avoid 1/d overflow during detection
                    scale_prev = max([1, abs(alpha(i - 1) + lamda), abs(beta(i - 1))]);
                    denom      = max(d(i - 1), 4 * sqe * scale_prev);
                    l(i - 1)   = beta(i - 1) / denom;
                    ref        = (beta(i - 1)^2) / denom;
                    d(i)       = alpha(i) + lamda - ref;

                    scale_i = max([1, abs(alpha(i) + lamda), ref]);
                    if d(i) <= 4 * sqe * scale_i % numerical stability
                        lam_min       = ref - alpha(i);
                        lam_min_saf   = max(lam_min_saf, lam_min + 4 * sqe * scale_i);
                        has_nc = true;
                        return
                    end
                end
            end
        end

        % Build explicit (T + lambda*I) and RHS beta1*e1 from tridiagonal data (for checks)
        function [Tlam, rhs] = build_tridiag_system(alpha, beta, lambda, beta1)
            t = numel(alpha);
            if t == 0
                Tlam = [];
                rhs = [];
                return
            end
            if ~isempty(beta) && numel(beta) ~= t - 1
                error('beta must have length t-1 where t = numel(alpha).');
            end
            Tlam = diag(alpha + lambda);
            if t >= 2
                Tlam = Tlam + diag(beta, +1) + diag(beta, -1);
            end
            rhs    = zeros(t, 1);
            rhs(1) = beta1;
        end

    end
end
