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
    %       If phi > 0 (lambda too small, in L): take ONE safeguarded Newton step
    %         and clip inside [lambda_lower, lambda_upper].
    %       If phi < 0 (in G): take a bracket step only (no Newton).
    %   Lambda is kept nondecreasing and clipped strictly inside the current bracket.
    %
    % Notes:
    %   No "hard case" augmentation here.
    %
    % Public knobs (kept minimal to match your version):
    %   update_strike : how many lambda raises before we escalate by doubling

    properties
        termination_rule (1, 1) Termination_Rule = Cartis_G_Termination_Rule
        update_strike   (1, 1) double {mustBeInteger, mustBePositive} = 10
    end

    methods (Static)

        function obj = from_struct(params)
            obj = GLRT_Parameters;
            if isfield(params, "update_strike")
                obj.update_strike = params.update_strike;
                params = rmfield(params, "update_strike");
            end
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

        function [status, best_x, iterations, sigma] = run(obj, ~, g, H, sigma)
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
            if beta1 == 0
                status = Optimization_Status.SUCCESS;
                best_x = zeros(n, 1);
                iterations = 0;
                return
            end

            % Lanczos process
            v_prev    = zeros(n, 1);
            beta_prev = beta1;
            v         = b / beta1;     % first Lanczos vector
            V         = v;             % reorthogonalized basis we build
            alpha_diag = [];
            beta_diag  = [];

            iterations = 0;

            % Bracket and controls for lambda (CGT-GLRT 9.2.1 style)
            lam_lower     = 0.0;
            lam_upper     = Inf;
            clip_margin   = 1e-10;   % tiny interior margin when clipping in [L,U]
            max_escalate  = 30;      % cap doubling attempts if NPC persists
            theta         = 0.5;     % linear split parameter (0<theta<1)

            while true
                % Lanczos step (before reorth)
                [v_next, alfa, betan_raw] = GLRT_Parameters.lanczos(H, v, v_prev, beta_prev);
                iterations = iterations + 1; % Mat-vec multiplications

                % Full reorthogonalization of v_next and normalization
                if iterations == 1
                    v_next = v_next - (v' * v_next) * v;
                    % Initialize lambda (classic cubic heuristic)
                    lamda = (sqrt(alfa^2 + 4 * beta1 * sigma) - alfa) / 2;
                    lamda = max(lamda, lam_lower);
                else
                    v_next = v_next - V * (V' * v_next);
                end

                betan = norm(v_next);
                if betan > eps
                    v_next = v_next / betan;
                else
                    % Subspace breakdown: stop expanding
                    betan  = 0;
                    v_next = zeros(n, 1);
                end

                % Grow basis and tridiagonal data
                V          = [V, v_next];
                alpha_diag = [alpha_diag; alfa];
                beta_diag  = [beta_diag;  betan];

                % Advance Lanczos state
                v_prev   = v;
                v        = v_next;
                beta_prev = betan;

                % Try LDL at current t; if NPC, raise lam_lower and retry (bounded by update_strike)
                t        = numel(alpha_diag);
                strikes  = 0;

                while true
                    [has_nc, l, d, lam_min_saf] = ...
                        GLRT_Parameters.ldl_tridiag(alpha_diag, beta_diag(1:end - 1), lamda);

                    if has_nc
                        % lambda is in the indefinite region N: lift the lower bound and retry
                        lam_lower = max(lam_lower, lam_min_saf);
                        lamda     = max(lamda, lam_lower);
                        strikes   = strikes + 1;

                        if strikes >= obj.update_strike
                            % racket-aware escalation: double until SPD,
                            % but stay inside (lam_lower, lam_upper) if finite
                            success = false;
                            lam_try = max(lamda, max(1.0, lam_lower));
                            for k = 1:max_escalate
                                if isfinite(lam_upper)
                                    % keep strictly inside the bracket
                                    left  = lam_lower + clip_margin * max(1, lam_upper - lam_lower);
                                    right = lam_upper - clip_margin * max(1, lam_upper - lam_lower);

                                    % propose a monotone lift; cap by right edge
                                    lam_try = min(max(2.0 * lam_try, left), right);

                                    % if no increase possible after capping, fall back to bracket step
                                    if lam_try <= lamda + eps
                                        lam_geom = sqrt(lam_lower * lam_upper);
                                        lam_lin  = lam_lower + theta * (lam_upper - lam_lower);
                                        lam_try  = max(lam_geom, lam_lin);  % strictly inside by construction
                                    end
                                else
                                    % no finite upper bound yet, pure doubling is fine
                                    lam_try = max(2.0 * lam_try, lam_lower * (2.0^k));
                                end

                                [has_nc2, l2, d2, lam_min2] = ...
                                    GLRT_Parameters.ldl_tridiag(alpha_diag, beta_diag(1:end - 1), lam_try);

                                if ~has_nc2
                                    l = l2;
                                    d = d2;
                                    lamda = lam_try;
                                    success = true;
                                    break
                                else
                                    lam_lower = max(lam_lower, lam_min2);
                                    lam_try   = max(lam_try, lam_lower); % next trial respects lifted lower bound
                                end
                            end
                            if ~success
                                status  = Optimization_Status.NUMERICAL_ISSUES;
                                best_x = V(:, 1:t - 1) * y;
                                return
                            end
                            % Found SPD via escalation, continue to SPD path below with l,d,lamda
                        else
                            % Try LDL again at the lifted lambda
                            continue
                        end
                    end

                    % SPD at current lambda: solve (T+lambda*I) y = beta1 e1, and build w
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

                    % Build w by L u = y, then w = u ./ sqrt(d)
                    sd     = sqrt(max(d, eps));
                    u      = zeros(t, 1);
                    u(1) = y(1);
                    for i = 2:t
                        u(i) = y(i) - l(i - 1) * u(i - 1);
                    end
                    w      = u ./ sd;

                    yn     = norm(y);
                    wn     = norm(w);
                    phi    = sigma * yn - lamda;   % secular residual in this subspace

                    % Update the bracket
                    if phi > 0
                        lam_lower = max(lam_lower, lamda);   % lambda too small, lift lower bound
                    else
                        lam_upper = min(lam_upper, lamda);   % lambda too large, lower upper bound
                    end

                    % Residual measure passed to your termination rule (kept unchanged)
                    norm_g = abs(phi) * yn;
                    rnorm  = norm_g + betan * abs(y(end));

                    run_info = struct( ...
                                      norm_g = rnorm, ... % return || nabla m ||
                                      x = y, ... % should be V(:,1:t) * y, but ||x|| = ||y||
                                      iteration = iterations, ...
                                      sigma = sigma, ...
                                      optional = struct() ...
                                     );
                    [terminate, status] = obj.termination_rule.should_terminate(run_info);
                    if terminate
                        best_x = V(:, 1:t) * y;
                        status = Optimization_Status.SUCCESS;
                        return
                    end

                    % One lambda update per t (CGT-GLRT 9.2.1)
                    if phi > 0
                        % In L (too small): ONE safeguarded Newton step, clipped to the bracket
                        derphi    = -sigma * (wn^2) / max(yn, eps) - 1;   % strictly negative in theory
                        delta_lam = -phi / max(derphi, -eps);
                        lam_cand  = lamda + delta_lam;

                        % If Newton leaves the bracket or is not finite, fall back to bracket step
                        out_of_bracket = ~isfinite(lam_cand) || ...
                                         (isfinite(lam_upper) && (lam_cand >= lam_upper - ...
                                                                  clip_margin * max(1, lam_upper - lam_lower))) || ...
                                         (lam_cand <= lam_lower + clip_margin * max(1, lam_upper - lam_lower));
                        if out_of_bracket
                            if isfinite(lam_upper)
                                lam_geom = sqrt(lam_lower * lam_upper);
                                lam_lin  = lam_lower + theta * (lam_upper - lam_lower);
                                lam_cand = max(lam_geom, lam_lin);
                            else
                                lam_cand = max(lamda, lamda + max(1e-18, 1e-6 * max([1, lamda, lam_lower])));
                            end
                        end
                    else
                        % In G (too large): bracket step only (no Newton)
                        if isfinite(lam_upper)
                            lam_geom = sqrt(lam_lower * lam_upper);
                            lam_lin  = lam_lower + theta * (lam_upper - lam_lower);
                            lam_cand = max(lam_geom, lam_lin);
                        else
                            lam_cand = max(lamda, lamda + max(eps, 1e-6 * max([1, lamda, lam_lower])));
                        end
                    end

                    % Clip strictly inside the bracket and keep lambda nondecreasing
                    if isfinite(lam_upper)
                        left  = lam_lower + clip_margin * max(1, lam_upper - lam_lower);
                        right = lam_upper - clip_margin * max(1, lam_upper - lam_lower);
                        % Guard: if the right edge has collapsed to lamda (or below), do not decrease lambda.
                        right = max(right, lamda);
                        lam_cand = min(max(lam_cand, max(lamda, left)), right);
                    else
                        lam_cand = max(lam_cand, lamda);
                    end

                    lamda = lam_cand;

                    % One lambda update per t, proceed to the next Lanczos step
                    break
                end
            end
        end

    end

    methods (Static)

        function [vn, alfa, betan] = lanczos(H, v, vp, beta)
            % One Lanczos step: vn (pre-reorth residual direction), alfa = v'*H*v, betan_raw = ||vn||.
            Av   = mat_vec(H, v);
            alfa = v' * Av;
            vn   = Av - v * alfa - vp * beta;
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
                        lam_min      = -alpha(1);
                        lam_min_saf  = max(lam_min_saf, lam_min + 4 * sqe * scale1);
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
                        lam_min      = ref - alpha(i);
                        lam_min_saf  = max(lam_min_saf, lam_min + 4 * sqe * scale_i);
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
