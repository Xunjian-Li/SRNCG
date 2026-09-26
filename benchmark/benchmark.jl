# ============================================================================
# benchmark_all_solvers.jl
#
# Formal benchmark using:
#   ARCqK, SRN-CG, GD, NAG, ANCG, ARNCG
#
# Design:
#   * Uses the validated current problem definitions from the SRN-CG regression.
#   * Uses the existing competitor implementations from compare_all_solvers.jl.
#   * Uses the installed SRNCG package; no embedded SRN-CG implementation.
#   * One full warm-up run per method/problem, then 10 timed repetitions.
#   * Fresh deterministic problem/caches for every run.
#   * Reports median runtime and IQR, plus deterministic algorithmic counters.
# ============================================================================

using AdaptiveRegularization
using NLPModels
using LinearAlgebra
using Random
using Printf
using Statistics
using BenchmarkTools
using SRNCG

import NLPModels: reset!, obj, grad


mutable struct OptResult
    name::String
    time::Float64
    f_final::Float64
    g_final::Float64
    iter::Int
    hvp::Int
    nobj::Int
    ngrad::Int
    x::Vector{Float64}
end


# ============================================================================
# 1. GD
# ============================================================================
function gd_optimize(nlp; initial_theta, max_iter = 1000, tol = 1e-5,
                     L0 = 10000.0, gamma = 0.5, eta = 1, max_ls = 30)
    reset!(nlp)
    x = copy(initial_theta)
    L = L0
    start = time()

    hvp_c = 0; nobj_c = 0; ngrad_c = 0

    f = obj(nlp, x); nobj_c += 1
    g = grad(nlp, x); ngrad_c += 1
    g_norm = norm(g)

    local it = 0
    for k in 1:max_iter
        it = k
        g_norm < tol && break
        d = -g
        dTg = dot(d, g)

        accepted = false
        for ls in 1:max_ls
            step = 1.0 / L
            x_new = x .+ step .* d
            f_new = obj(nlp, x_new); nobj_c += 1
            if f_new <= f + eta * step * dTg
                x = x_new; f = f_new
                L = max(L * gamma, 1e-12)
                accepted = true
                break
            else
                L = L / gamma
            end
        end
        if !accepted
            step = 1.0 / L
            x = x .+ step .* d
            f = obj(nlp, x); nobj_c += 1
            L = max(L * 2.0, 1e-12)
        end

        g = grad(nlp, x); ngrad_c += 1
        g_norm = norm(g)
    end

    return OptResult("GD", time() - start, f, g_norm, it, hvp_c, nobj_c, ngrad_c, x)
end


# ============================================================================
# 2. NAG
# ============================================================================
function nag_optimize(nlp; initial_theta, max_iter = 1000, tol = 1e-5,
                      L0 = 1e0, gamma = 0.5, eta = 1, max_ls = 30)
    reset!(nlp)
    x = copy(initial_theta)
    y = copy(x)
    A_prev = 0.0
    L = L0
    start = time()

    hvp_c = 0; nobj_c = 0; ngrad_c = 0

    f = obj(nlp, x); nobj_c += 1
    g = grad(nlp, x); ngrad_c += 1
    g_norm = norm(g)

    local it = 0
    for k in 1:max_iter
        it = k
        g_theta = grad(nlp, y); ngrad_c += 1
        g_at_theta = grad(nlp, x); ngrad_c += 1
        g_norm_theta = norm(g_at_theta)
        g_norm_theta < tol && break

        d = -g_theta
        dTg = dot(d, g_theta)
        f_y = obj(nlp, y); nobj_c += 1

        x_new = y
        accepted = false
        for ls in 1:max_ls
            step = 1.0 / L
            x_try = y .+ step .* d
            f_try = obj(nlp, x_try); nobj_c += 1
            if f_try <= f_y + eta * step * dTg
                x_new = x_try
                accepted = true
                break
            else
                L = L / gamma
            end
        end
        if !accepted
            step = 1.0 / L
            x_new = y .+ step .* d
            L = L * 2.0
        else
            L = L * gamma
        end
        L = max(L, 1e-12)

        a_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * A_prev^2))
        tau = (A_prev - 1.0) / a_new
        y_new = x_new .+ tau .* (x_new .- x)

        momentum = x_new .- x
        if dot(momentum, -g_at_theta) < 0.0
            A_prev = 0.0
            y_new = copy(x_new)
        else
            A_prev = a_new
        end

        x = x_new
        y = y_new
        f = obj(nlp, x); nobj_c += 1
        g = grad(nlp, x); ngrad_c += 1
        g_norm = norm(g)
    end

    return OptResult("NAG", time() - start, f, g_norm, it, hvp_c, nobj_c, ngrad_c, x)
end

function ancg_optimize(nlp; initial_theta, max_iter = 100, tol = 1e-5,
                       gamma_0 = 10.0, theta = 0.5, eta = 0.01,
                       max_cg = 200, max_ls = 40)
    reset!(nlp)
    x = copy(initial_theta)
    gamma_k = gamma_0
    c_sol = eta * (1 - eta) * theta / 400.0
    start = time()

    hvp_c = 0; nobj_c = 0; ngrad_c = 0

    function capped_cg(x_np, g_np, eps, zeta)
        n = length(g_np)
        sigma = max(eps, 1e-12)
        gn = norm(g_np)
        gn < 1e-15 && return zeros(n), :SOL

        y = zeros(n)
        r = copy(g_np)
        p = -r
        r0_norm = norm(r)
        r0_norm < 1e-15 && return zeros(n), :SOL

        Hp = hprod(nlp, x_np, p); hvp_c += 1
        if dot(p, Hp) < sigma * dot(p, p)
            return copy(p), :NC
        end
        U = norm(Hp) / max(norm(p), 1e-15)

        for j in 1:max_cg
            Hp = hprod(nlp, x_np, p); hvp_c += 1
            p_norm = max(norm(p), 1e-15)
            U = max(U, norm(Hp) / p_norm)

            H_bar_p = Hp .+ 2 * sigma * p
            pHp = dot(p, H_bar_p)
            pHp <= 0 && return p ./ p_norm, :NC

            r_norm_sq = dot(r, r)
            alpha = r_norm_sq / pHp
            y_new = y .+ alpha .* p
            r_new = r .+ alpha .* H_bar_p

            kappa = max((U + 2*sigma) / max(sigma, 1e-15), 1.0)
            zeta_cur = zeta / (3 * kappa)

            Hy_new = hprod(nlp, x_np, y_new); hvp_c += 1
            y_norm_sq = dot(y_new, y_new)
            y_H_y = dot(y_new, Hy_new) + 2*sigma*y_norm_sq
            if y_H_y < sigma * y_norm_sq
                return y_new ./ max(sqrt(y_norm_sq), 1e-15), :NC
            end

            r_new_norm = norm(r_new)
            r_new_norm <= zeta_cur * r0_norm && return copy(y_new), :SOL

            if j >= 1
                tau_cg = sqrt(kappa) / (sqrt(kappa) + 1)
                T_val = 4 * kappa^4 / max((1 - sqrt(tau_cg))^2, 1e-15)
                if r_new_norm > sqrt(T_val) * r0_norm
                    alpha_next = dot(r_new, r_new) / max(pHp, 1e-30)
                    y_next = y_new .+ alpha_next .* p
                    Hy_next = hprod(nlp, x_np, y_next); hvp_c += 1
                    yn_sq = dot(y_next, y_next)
                    yn_H_yn = dot(y_next, Hy_next) + 2*sigma*yn_sq
                    if yn_H_yn < sigma * yn_sq
                        return y_next ./ max(sqrt(yn_sq), 1e-15), :NC
                    end
                end
            end

            r_new_norm_sq = dot(r_new, r_new)
            r_new_norm_sq < 1e-30 && return copy(y_new), :SOL
            beta = r_new_norm_sq / r_norm_sq
            p = -r_new .+ beta .* p
            y = y_new
            r = r_new
        end
        return copy(y), :SOL
    end

    function linesearch_nc(x_np, d_np, f_k, g_k_norm, gamma_k)
        d_norm3 = norm(d_np)^3
        for j in 0:(max_ls-1)
            alpha = theta^j
            x_try = x_np .+ alpha .* d_np
            f_try = obj(nlp, x_try); nobj_c += 1
            (isnan(f_try) || isinf(f_try)) && continue
            if f_try <= f_k - (eta/2) * alpha^2 * d_norm3
                g_try = grad(nlp, x_try); ngrad_c += 1
                g_try_norm = norm(g_try)
                need_update = (g_try_norm > g_k_norm / 2) && (alpha < theta / gamma_k)
                return alpha, x_try, true, need_update
            end
        end
        return 0.0, x_np, false, true
    end

    function linesearch_sol(x_np, d_np, f_k, g_k_norm, gamma_k, eps_k)
        d_norm = norm(d_np)

        x_full = x_np .+ d_np
        f_full = obj(nlp, x_full); nobj_c += 1
        g_full = grad(nlp, x_full); ngrad_c += 1
        g_full_norm = norm(g_full)

        if !isnan(f_full) && !isinf(f_full)
            if f_full <= f_k && g_full_norm <= g_k_norm / 2
                return 1.0, x_full, true, false
            end
        end

        for j in 0:(max_ls-1)
            alpha = theta^j
            x_try = x_np .+ alpha .* d_np
            f_try = obj(nlp, x_try); nobj_c += 1
            (isnan(f_try) || isinf(f_try)) && continue
            if f_try <= f_k - eta * eps_k * alpha * d_norm^2
                g_try = grad(nlp, x_try); ngrad_c += 1
                g_try_norm = norm(g_try)
                descent = f_k - f_try
                threshold = c_sol / sqrt(max(gamma_k, 1e-15)) * g_k_norm^1.5
                need_update = (g_try_norm > g_k_norm / 2) && (descent < threshold)
                return alpha, x_try, true, need_update
            end
        end
        return 0.0, x_np, false, true
    end

    f_k = obj(nlp, x); nobj_c += 1
    g_k = grad(nlp, x); ngrad_c += 1
    g_k_norm = norm(g_k)

    local it = 0
    for k in 1:max_iter
        it = k
        g_k_norm < tol && break

        if isnan(f_k) || isinf(f_k) || g_k_norm > 1e10
            x = x .- 1e-8 .* g_k ./ max(g_k_norm, 1e-15)
            gamma_k = max(gamma_k * 10, 1e6)
            g_k = grad(nlp, x); ngrad_c += 1
            g_k_norm = norm(g_k)
            f_k = obj(nlp, x); nobj_c += 1
            continue
        end

        eps_k = sqrt(max(gamma_k * g_k_norm, 1e-15))
        zeta_k = min(0.5, sqrt(g_k_norm))

        local d, d_type
        try
            d, d_type = capped_cg(x, g_k, eps_k, zeta_k)
        catch
            x = x .- 1e-6 .* g_k ./ max(g_k_norm, 1e-15)
            g_k = grad(nlp, x); ngrad_c += 1
            g_k_norm = norm(g_k)
            f_k = obj(nlp, x); nobj_c += 1
            continue
        end

        d_norm = norm(d)
        if d_norm < 1e-15
            continue
        end

        if d_type == :NC
            Hd = hprod(nlp, x, d); hvp_c += 1
            d_T_H_d = dot(d, Hd)
            d_T_g = dot(d, g_k)
            sign = d_T_g < 0 ? -1.0 : 1.0
            d_k = -sign * abs(d_T_H_d) / max(d_norm^3, 1e-15) .* d

            alpha_k, x_new, accepted, need_update =
                linesearch_nc(x, d_k, f_k, g_k_norm, gamma_k)

            if accepted
                x = x_new
            end
            need_update && (gamma_k *= 2.0)
        else
            d_k = d
            alpha_k, x_new, accepted, need_update =
                linesearch_sol(x, d_k, f_k, g_k_norm, gamma_k, eps_k)

            if accepted
                x = x_new
                need_update && (gamma_k *= 2.0)
            else
                gamma_k *= 2.0
                if alpha_k < 1e-8
                    x = x .- 1e-8 .* g_k ./ max(g_k_norm, 1e-15)
                end
            end
        end

        g_k = grad(nlp, x); ngrad_c += 1
        g_k_norm = norm(g_k)
        f_k = obj(nlp, x); nobj_c += 1
    end

    return OptResult("ANCG", time() - start, f_k, g_k_norm, it, hvp_c, nobj_c, ngrad_c, x)
end


# ============================================================================
# 3. ARNCG — direct Julia translation of authors' MATLAB implementation
# ============================================================================

# Julia translation of the authors' MATLAB ARNCG implementation.
# Source: ARNCG-master/matlab/AdapNewtonCG.m and CappedCG.m
# Default configuration matches TestCUTEst.m index=2: ARNCG_g, lambda=0, theta=1.
# This is a benchmark function, not a package.

function arncg_official_optimize(nlp; initial_theta,
        max_iter::Int=100_000, tol::Float64=1e-5,
        cg_reltol::Float64=0.01, cg_abstol::Float64=0.01,
        cg_maxiter::Int=length(initial_theta)+2,
        max_omega::Float64=Inf,
        beta::Float64=0.5, mu::Float64=0.3, min_alpha::Float64=0.3,
        gamma::Float64=5.0, tau_minus::Float64=0.3, tau_plus::Float64=1.0,
        theta::Float64=1.0,
        regularization_policy::Symbol=:gradient,
        acceleration_policy::Symbol=:gradient,
        fixed_omega::Float64=sqrt(1e-5),
        fallback_enabled::Bool=false,
        fallback_growth_threshold::Float64=100.0,
        fallback_shrink_threshold::Float64=0.01,
        minimal_norm_d::Union{Nothing,Float64}=2e-16,
        exit_for_many_unchanged_f_and_g::Int=20,
        max_time::Float64=5*3600.0,
        verbose::Bool=false)

    reset!(nlp)
    start = time()
    hvp_c = 0
    grad_evals = 1
    func_evals = 1
    hess_evals = 0
    rejected_step_flag = 0

    # Count actual Hessian-vector products while keeping the official CG logic.
    function hv(x, v)
        hvp_c += 1
        return hprod(nlp, x, v)
    end

    function capped_cg(x, G, reltol, abstol, rho, max_cg)
        Y = zeros(size(G)); Hy = zeros(size(G))
        R = copy(G)
        Hr = hv(x, R)
        P = -R
        Hp = hv(x, P)
        norm_rr0 = dot(R, R)

        U = 0.0
        kappa = (U + 2*rho) / rho
        tau_cg = sqrt(kappa) / (sqrt(kappa) + 1)
        T = 4*kappa^4 / (1 - sqrt(kappa))^2
        dtype = :EMPTY
        norm_rr = norm_rr0
        norm_pbarHp = NaN
        barHp = similar(P)
        it_out = 0

        for it in 0:max_cg
            it_out = it
            if it > 0
                alpha = norm_rr / norm_pbarHp
                Y = Y + alpha * P
                Hy = Hy + alpha * Hp
                R = alpha * barHp + R
                Hr = hv(x, R)
                beta_cg = dot(R,R) / norm_rr
                P = beta_cg * P - R
                Hp = beta_cg * Hp - Hr
            end

            norm_Hp = dot(Hp,Hp); norm_pp = dot(P,P)
            barHp = Hp + 2*rho*P
            norm_pbarHp = dot(P,barHp)
            if sqrt(norm_Hp) > U*sqrt(norm_pp)
                U = sqrt(norm_Hp)/sqrt(norm_pp)
            end
            norm_Hr = dot(Hr,Hr); norm_rr = dot(R,R)
            if sqrt(norm_Hr) > U*sqrt(norm_rr)
                U = sqrt(norm_Hr)/sqrt(norm_rr)
            end
            norm_Hy = dot(Hy,Hy); norm_yy = dot(Y,Y)
            if sqrt(norm_Hy) > U*sqrt(norm_yy)
                U = sqrt(norm_Hy)/sqrt(norm_yy)
            end

            kappa = (U + 2*rho)/rho
            tau_cg = sqrt(kappa)/(sqrt(kappa)+1)
            T = 4*kappa^4/(1-sqrt(kappa))^2
            hat_xi = reltol/(3*kappa)

            norm_ybarHy = dot(Y, Hy + 2*rho*Y)
            if norm_ybarHy < rho*norm_yy
                return copy(Y), sqrt(norm_rr), it_out, :NC
            elseif sqrt(norm_rr) < hat_xi*sqrt(norm_rr0) && sqrt(norm_rr) < abstol
                return copy(Y), sqrt(norm_rr), it_out, :SOL
            elseif norm_pbarHp < rho*norm_pp
                return copy(P), sqrt(norm_rr), it_out, :NC
            elseif sqrt(norm_rr) > sqrt(T)*(tau_cg^(it/2))*sqrt(norm_rr0)
                # Official TestCUTEst uses cg_policy='recompute'.
                alpha = norm_rr/norm_pbarHp
                Y_extra = Y + alpha*P
                HY_extra = Hy + alpha*Hp

                Y = zeros(size(G)); Hy = zeros(size(G))
                R = copy(G); Hr = hv(x,R)
                P = -R; Hp = hv(x,P)
                dtype = :ERR

                for i in 1:(it+1)
                    diff = Y_extra - Y
                    sqrnorm_diff = dot(diff,diff)
                    diff_H_diff = dot(diff, HY_extra-Hy)
                    if diff_H_diff < rho*sqrnorm_diff
                        return copy(diff), sqrt(norm_rr), it_out, :NC
                    end

                    # Preserve the MATLAB source literally: condition is `it > 0`.
                    if it > 0
                        alpha = norm_rr/norm_pbarHp
                        Y = Y + alpha*P
                        Hy = Hy + alpha*Hp
                        R = alpha*barHp + R
                        Hr = hv(x,R)
                        beta_cg = dot(R,R)/norm_rr
                        P = beta_cg*P - R
                        Hp = beta_cg*Hp - Hr
                    end

                    norm_Hp = dot(Hp,Hp); norm_pp = dot(P,P)
                    barHp = Hp + 2*rho*P
                    norm_pbarHp = dot(P,barHp)
                    norm_Hr = dot(Hr,Hr); norm_rr = dot(R,R)
                    norm_Hy = dot(Hy,Hy); norm_yy = dot(Y,Y)
                end
                break
            end
        end

        return copy(Y), sqrt(norm_rr), it_out, (dtype == :EMPTY ? :SOL : dtype)
    end

    M = 1.0
    x = copy(initial_theta)
    g = grad(nlp,x)
    f = obj(nlp,x)
    norm_g = norm(g)
    norm_g_min = norm_g
    prev_norm_g = norm_g
    prev_norm_g_min = norm_g_min
    fallback_flag = false

    # Only values needed for the official unchanged-f/g stopping test.
    f_hist = Float64[]
    g_hist = Float64[]
    it_done = 0

    for it in 1:max_iter
        it_done = it
        norm_g < tol && break
        hess_evals += 1 - rejected_step_flag
        rejected_step_flag = 0

        omega_base = regularization_policy == :gradient ? sqrt(norm_g) :
                     regularization_policy == :minimum_gradient ? sqrt(norm_g_min) :
                     regularization_policy == :fixed ? fixed_omega : error("Unknown regularization policy")

        if !fallback_flag
            omega = acceleration_policy == :gradient ? omega_base*min(1.0,(norm_g/prev_norm_g)^theta) :
                    acceleration_policy == :minimum_gradient ? omega_base*(norm_g_min/prev_norm_g_min)^theta :
                    acceleration_policy == :fixed ? fixed_omega : error("Unknown acceleration policy")
        else
            omega = omega_base
            fallback_flag = false
        end
        omega = min(omega,max_omega)
        cgtol = min(cg_reltol,sqrt(M)*omega)

        tilde_d,cg_norm_rr,cg_it,cg_dtype = capped_cg(x,g,cgtol,cg_abstol,sqrt(M)*omega,cg_maxiter)
        dot_g_tilde_d = dot(g,tilde_d)
        linesearch_failure_flag = false
        smaller_stepsize_flag = false

        if cg_dtype == :SOL
            d = tilde_d; norm_d = norm(d); dot_g_d = dot_g_tilde_d
            alpha = 1.0
            x_new = x; f_new = f
            while alpha > min_alpha
                x_new = x + alpha*d
                f_new = obj(nlp,x_new); func_evals += 1
                f_new <= f + mu*alpha*dot_g_d && break
                alpha *= beta
            end
            hat_alpha = sqrt(omega/sqrt(M)/norm_d)
            if hat_alpha < 1 && alpha <= min_alpha
                alpha = 1.0; smaller_stepsize_flag = true
                while alpha > min_alpha
                    x_new = x + hat_alpha*alpha*d
                    f_new = obj(nlp,x_new); func_evals += 1
                    f_new <= f + mu*hat_alpha*alpha*dot_g_d && break
                    alpha *= beta
                end
                alpha <= min_alpha && (linesearch_failure_flag = true)
                alpha = hat_alpha*alpha
            elseif alpha <= min_alpha
                linesearch_failure_flag = true
            end
        else
            norm_tilde_d = norm(tilde_d)
            normalized_tilde_d = tilde_d/norm_tilde_d
            L_d = abs(dot(hv(x,normalized_tilde_d),normalized_tilde_d))/M
            d = -L_d*sign(dot(g,normalized_tilde_d))*normalized_tilde_d
            norm_d = norm(d); dot_g_d = dot(g,d)
            alpha = 1.0; x_new=x; f_new=f
            while alpha > min_alpha
                x_new = x + alpha*d
                f_new = obj(nlp,x_new); func_evals += 1
                f_new <= f - M*mu*alpha^2*norm_d^3 && break
                alpha *= beta
            end
            alpha <= min_alpha && (linesearch_failure_flag = true)
        end

        M_new = M
        predict_lipcoeff = tau_plus/sqrt(M)
        M_dec_coeff = mu*tau_minus
        gradient_computed = false
        g_new = g

        if cg_dtype == :SOL
            if alpha < 1
                predict_descent = predict_lipcoeff*omega^3
                f < beta*mu*predict_descent + f_new && (M_new=M*gamma)
            else
                g_new = grad(nlp,x_new); grad_evals += 1; gradient_computed=true
                predict_descent = predict_lipcoeff*min(omega^3,norm(g_new)^2/omega)
                M_dec_coeff *= 4/33
                f < (4/33)*mu*predict_descent + f_new && (M_new=M*gamma)
            end
        else
            predict_descent = predict_lipcoeff*omega^3
            f < ((1-2*mu)*beta)^2*mu*predict_descent + f_new && (M_new=M*gamma)
        end

        if f > M_dec_coeff*predict_lipcoeff*omega_base^3 + f_new
            M_new = M/gamma
        end
        Delta = f-f_new

        if linesearch_failure_flag
            M_new=M*gamma; x_new=x; f_new=f; g_new=g
            rejected_step_flag=1; gradient_computed=true
        end

        if fallback_enabled && theta > 0
            if norm_g < prev_norm_g*fallback_shrink_threshold
                if !gradient_computed
                    g_new=grad(nlp,x_new); grad_evals+=1; gradient_computed=true
                end
                next_norm_g=norm(g_new)
                if next_norm_g > norm_g*fallback_growth_threshold
                    fallback_flag=true; rejected_step_flag=1
                    x_new=x; g_new=g; f_new=f; M_new=M
                end
            end
        end

        push!(f_hist,f); push!(g_hist,norm_g)
        if exit_for_many_unchanged_f_and_g > 0 && length(f_hist) > exit_for_many_unchanged_f_and_g
            j=length(f_hist)-exit_for_many_unchanged_f_and_g
            if f_hist[end] == f_hist[j] && g_hist[end] == g_hist[j]
                break
            end
        end

        x=x_new; M=M_new
        if !gradient_computed
            grad_evals += 1; g=grad(nlp,x)
        else
            g=g_new
        end
        prev_norm_g=norm_g; prev_norm_g_min=norm_g_min
        norm_g=norm(g); norm_g_min=min(norm_g,norm_g_min)
        f=f_new

        isnan(f) && break
        M > 1e40 && break
        time()-start > max_time && break
        minimal_norm_d !== nothing && norm_d < minimal_norm_d && break
    end

    return OptResult("ARNCG",time()-start,f,norm_g,it_done,hvp_c,func_evals,grad_evals,copy(x))
end

# ============================================================================
# Benchmark wrappers
# ============================================================================

const BENCH_TOL = 1e-7
const BENCH_MAX_ITER = 10_000
const BENCH_SECONDS = 5.0
const ARC_MAX_TIME = 120.0

function make_problem(name::String)
    if name == "Rosenbrock"
        x0, o, g, hp = make_rosenbrock(1000; x0=zeros(1000))
        label = "Rosenbrock (dim=1000)"
    elseif name == "LogSumExp"
        x0, o, g, hp = make_logsumexp(200, 1000, 0.05; x0=zeros(200))
        label = "Log-Sum-Exp (dim=200, n=1000, rho=0.05)"
    elseif name == "Polytope"
        x0, o, g, hp = make_polytope(400, 500, 4; x0=zeros(400))
        label = "Polytope (dim=400, m=500, p=4)"
    elseif name == "TRegression"
        x0, o, g, hp = make_tregression(200, 500, 0.001)
        label = "T-Regression (dim=200, m=500, nu=0.001)"
    else
        error("Unknown problem: $name")
    end

    nlp = SRNCG.ClosureNLPModel(copy(x0), o, g, hp; name=name)
    return label, nlp, copy(x0)
end

function benchmark_arcqk(nlp; max_time=ARC_MAX_TIME, tol=BENCH_TOL)
    reset!(nlp)
    t = @elapsed begin
        stats = ARCqKOp(
            nlp;
            max_time=max_time,
            max_iter=typemax(Int64),
            max_eval=typemax(Int64),
            atol=tol,
            rtol=0.0,
        )
    end

    return OptResult(
        "ARCqK", t, stats.objective, stats.dual_feas, stats.iter,
        neval_hprod(nlp), neval_obj(nlp), neval_grad(nlp),
        copy(stats.solution),
    )
end

function benchmark_srncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    reset!(nlp)

    solver = SRNCG.SRN_CG(max_outer=max_outer)

    t = @elapsed begin
        x_sol = SRNCG.optimize!(
            solver, nlp;
            initial_theta=copy(x0),
            max_iter=max_outer,
            tol=tol,
        )
    end

    # Capture counters before final diagnostic evaluations.
    nobj = nlp.counters.neval_obj
    ngrad = nlp.counters.neval_grad

    f_sol = obj(nlp, x_sol)
    g_sol = norm(grad(nlp, x_sol))

    return OptResult(
        "SRN-CG", t, f_sol, g_sol,
        length(solver.loss_history) - 1,
        solver.hvp_count, nobj, ngrad, copy(x_sol),
    )
end

function run_method(method::String, problem_name::String)
    _, nlp, x0 = make_problem(problem_name)

    if method == "ARCqK"
        return benchmark_arcqk(nlp; tol=BENCH_TOL)
    elseif method == "SRN-CG"
        return benchmark_srncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    elseif method == "GD"
        return gd_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "NAG"
        return nag_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ANCG"
        return ancg_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ARNCG"
        return arncg_official_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
            cg_maxiter=length(x0)+2,
        )
    else
        error("Unknown method: $method")
    end
end

function deterministic_run(method::String, problem_name::String)
    return run_method(method, problem_name)
end

function benchmark_method(method::String, problem_name::String; seconds=BENCH_SECONDS)
    # Fresh deterministic problem and mutable caches are constructed in setup.
    # BenchmarkTools excludes setup time from the measured expression.
    #
    # Do not interpolate `seconds` with `$seconds` here. BenchmarkTools parses
    # keyword parameters such as `seconds` outside the benchmark expression.
    params = BenchmarkTools.Parameters(seconds=seconds, evals=1)
    trial = @benchmarkable begin
        run_method_on_instance($method, nlp, x0)
    end setup=begin
        problem_label, nlp, x0 = make_problem($problem_name)
    end
    trial.params = params
    result = run(trial)

    med_ns = median(result).time
    q25_ns = quantile(result.times, 0.25)
    q75_ns = quantile(result.times, 0.75)

    return (
        trial=result,
        median=med_ns / 1e9,
        q25=q25_ns / 1e9,
        q75=q75_ns / 1e9,
        iqr=(q75_ns - q25_ns) / 1e9,
        samples=length(result.times),
    )
end

function run_method_on_instance(method::String, nlp, x0)
    if method == "ARCqK"
        return benchmark_arcqk(nlp; tol=BENCH_TOL)
    elseif method == "SRN-CG"
        return benchmark_srncg(nlp, x0; tol=BENCH_TOL, max_outer=BENCH_MAX_ITER)
    elseif method == "GD"
        return gd_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "NAG"
        return nag_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ANCG"
        return ancg_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
        )
    elseif method == "ARNCG"
        return arncg_official_optimize(
            nlp; initial_theta=x0,
            max_iter=BENCH_MAX_ITER, tol=BENCH_TOL,
            cg_maxiter=length(x0)+2,
        )
    else
        error("Unknown method: $method")
    end
end

function print_summary(rows)
    println()
    println("="^178)
    println("Formal benchmark summary (BenchmarkTools, evals=1, seconds=$(BENCH_SECONDS) per method/problem)")
    println("Timing excludes deterministic problem/cache construction in setup.")
    println("Algorithmic counters and final values come from one separate deterministic run.")
    println("="^178)

    @printf(
        "%-13s %-9s %11s %11s %11s %8s %9s %8s %8s %8s %13s %12s %-14s\n",
        "Problem", "Method", "Median(s)", "Q25(s)", "Q75(s)", "Samples",
        "Iter", "HVP", "Obj", "Grad", "f(x*)", "|g(x*)|", "Status"
    )
    println("-"^178)

    current_problem = ""
    for r in rows
        if current_problem != "" && r.problem != current_problem
            println("-"^178)
        end
        current_problem = r.problem

        @printf(
            "%-13s %-9s %11.5f %11.5f %11.5f %8d %9d %8d %8d %8d %13.5e %12.4e %-14s\n",
            r.problem, r.method, r.median, r.q25, r.q75, r.samples,
            r.iter, r.hvp, r.nobj, r.ngrad, r.f, r.g, r.status
        )
    end
    println("="^178)
end

function main()
    problem_names = ["Rosenbrock", "LogSumExp", "Polytope", "TRegression"]
    methods = ["ARCqK", "SRN-CG", "GD", "NAG", "ANCG", "ARNCG"]

    println("="^108)
    println("Formal optimizer benchmark")
    println("Methods: ", join(methods, ", "))
    println("BenchmarkTools protocol: seconds=$(BENCH_SECONDS), evals=1")
    println("Tolerance: $(BENCH_TOL)")
    println("="^108)

    rows = NamedTuple[]

    for problem_name in problem_names
        label, _, _ = make_problem(problem_name)

        println()
        println("="^108)
        println("Problem: ", label)
        println("="^108)

        for method in methods
            # Run once outside BenchmarkTools to record deterministic algorithmic metrics.
            print(@sprintf("  %-8s deterministic run ... ", method))
            r = deterministic_run(method, problem_name)
            status = r.g_final <= BENCH_TOL ? "CONVERGED" : "NOT CONVERGED"
            @printf(
                "iter=%d HVP=%d f=%.5e |g|=%.4e [%s]\n",
                r.iter, r.hvp, r.f_final, r.g_final, status
            )

            # BenchmarkTools performs compilation warm-up and repeated sampling.
            print(@sprintf("  %-8s benchmark ... ", method))
            b = benchmark_method(method, problem_name)
            @printf(
                "median=%.5f s  Q25=%.5f  Q75=%.5f  samples=%d\n",
                b.median, b.q25, b.q75, b.samples
            )

            push!(rows, (
                problem=problem_name,
                method=method,
                median=b.median,
                q25=b.q25,
                q75=b.q75,
                iqr=b.iqr,
                samples=b.samples,
                iter=r.iter,
                hvp=r.hvp,
                nobj=r.nobj,
                ngrad=r.ngrad,
                f=r.f_final,
                g=r.g_final,
                status=status,
            ))
        end
    end

    print_summary(rows)
    return rows
end

rows = main()