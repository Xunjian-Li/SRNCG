# module SRNCG

using LinearAlgebra
using NLPModels

import NLPModels: obj, grad!, hprod!

export SRN_CG, ClosureNLPModel, optimize!

"""
    ClosureNLPModel(x0, obj_fn, grad_fn, hprod_fn; name="Closure")

Minimal `NLPModels.jl` wrapper for an objective, in-place gradient, and
in-place Hessian-vector product. The callback signatures are

    obj_fn(x)
    grad_fn(g, x)
    hprod_fn(Hv, x, v)
"""
mutable struct ClosureNLPModel{OF,GF,HF} <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    obj_fn::OF
    grad_fn::GF
    hprod_fn::HF
end

function ClosureNLPModel(x0::Vector{Float64}, obj_fn, grad_fn, hprod_fn;
                         name::String = "Closure")
    meta = NLPModelMeta(length(x0); x0=x0, name=name)
    counters = NLPModels.Counters()
    return ClosureNLPModel(meta, counters, obj_fn, grad_fn, hprod_fn)
end

function NLPModels.obj(nlp::ClosureNLPModel, x)
    nlp.counters.neval_obj += 1
    return nlp.obj_fn(x)
end

function NLPModels.grad!(nlp::ClosureNLPModel, x::AbstractVector, g::AbstractVector)
    nlp.counters.neval_grad += 1
    nlp.grad_fn(g, x)
    return g
end

function NLPModels.hprod!(nlp::ClosureNLPModel, x::AbstractVector,
                          v::AbstractVector, Hv::AbstractVector; kwargs...)
    nlp.counters.neval_hprod += 1
    nlp.hprod_fn(Hv, x, v)
    return Hv
end

"""
    SRN_CG(; kwargs...)

Sparse regularized Newton-CG solver using a Lanczos/Krylov reduced model and
a scalar target-ratio equation.

`optimize!(solver, nlp; initial_theta, max_iter, tol)` returns the final iterate.
Run statistics and histories are stored in `solver`.
"""
mutable struct SRN_CG
    sigma_0_ref::Float64
    sigma_min::Float64
    rho::Float64
    beta::Float64
    gamma::Float64
    tau::Float64
    max_outer::Int
    max_krylov::Int
    max_newton::Int
    tol_newton::Float64
    verbose::Bool

    hvp_count::Int
    grad_count::Int
    func_count::Int
    krylov_extensions::Int
    ratio_increases::Int
    newton_iters_total::Int
    accepted_kappa::Vector{Float64}
    attained_dims::Vector{Int}

    loss_history::Vector{Float64}
    grad_norm_history::Vector{Float64}
    time_history::Vector{Float64}
    total_time::Float64
end

function SRN_CG(;
    sigma_0_ref::Float64 = 1.0,
    sigma_min::Float64 = 1e-8,
    rho::Float64 = 2.0,
    beta::Float64 = 0.25,
    gamma::Float64 = 1e-3,
    tau::Float64 = 0.1,
    max_outer::Int = 500,
    max_krylov::Int = 0,
    max_newton::Int = 10,
    tol_newton::Float64 = 1e-3,
    verbose::Bool = false,
)
    @assert 0 < beta < 0.5
    @assert 0 < gamma < tau < 1
    @assert rho > 1
    @assert sigma_0_ref >= sigma_min > 0

    return SRN_CG(
        sigma_0_ref, sigma_min, rho, beta, gamma, tau,
        max_outer, max_krylov, max_newton, tol_newton, verbose,
        0, 0, 0, 0, 0, 0, Float64[], Int[],
        Float64[], Float64[], Float64[], 0.0,
    )
end

function _reset_stats!(s::SRN_CG)
    s.hvp_count = 0
    s.grad_count = 0
    s.func_count = 0
    s.krylov_extensions = 0
    s.ratio_increases = 0
    s.newton_iters_total = 0
    s.accepted_kappa = Float64[]
    s.attained_dims = Int[]
    s.loss_history = Float64[]
    s.grad_norm_history = Float64[]
    s.time_history = Float64[]
    s.total_time = 0.0
end

function _lanczos_extension_qdirect!(s::SRN_CG, nlp::N, theta,
                                     q_prev::AbstractVector{Float64},
                                     q_curr::AbstractVector{Float64},
                                     delta_prev::Float64,
                                     Hq_work::Vector{Float64},
                                     w_work::Vector{Float64},
                                     q_next::AbstractVector{Float64}) where {N<:ClosureNLPModel}
    begin
        begin
            hprod!(nlp, theta, q_curr, Hq_work)
        end
        s.hvp_count += 1
    end

    begin
        @inbounds @simd for i in eachindex(w_work, Hq_work, q_prev)
            w_work[i] = Hq_work[i] - delta_prev * q_prev[i]
        end
    end

    theta_j = dot(q_curr, w_work)

    begin
        @inbounds @simd for i in eachindex(w_work, q_curr)
            w_work[i] -= theta_j * q_curr[i]
        end
    end

    coeff = dot(w_work, q_curr)

    begin
        @inbounds @simd for i in eachindex(w_work, q_curr)
            w_work[i] -= coeff * q_curr[i]
        end
    end

    delta_j = norm(w_work)

    if delta_j < 1e-14
        return theta_j, 0.0, true
    end

    begin
        @inbounds @simd for i in eachindex(q_next, w_work)
            q_next[i] = w_work[i] / delta_j
        end
    end

    return theta_j, delta_j, false
end

# In-place Thomas solver used only by the SRN-CG target-ratio calculation.
# Arithmetic/order is the same as _solve_tridiagonal; only storage is reused.
function _solve_tridiagonal_ws!(y::AbstractVector{Float64},
                                diag::AbstractVector{Float64},
                                offdiag::AbstractVector{Float64},
                                rhs::AbstractVector{Float64},
                                c_prime::AbstractVector{Float64},
                                d_prime::AbstractVector{Float64})
    k = length(rhs)
    k == 0 && return y

    denom = abs(diag[1]) > 1e-15 ? diag[1] : 1e-15
    c_prime[1] = k > 1 ? offdiag[1] / denom : 0.0
    d_prime[1] = rhs[1] / denom

    for i in 2:k
        denom = diag[i] - offdiag[i-1] * c_prime[i-1]
        abs(denom) < 1e-15 && (denom = 1e-15)
        c_prime[i] = i < k ? offdiag[i] / denom : 0.0
        d_prime[i] = (rhs[i] - offdiag[i-1] * d_prime[i-1]) / denom
    end

    y[k] = d_prime[k]
    for i in (k-1):-1:1
        y[i] = d_prime[i] - c_prime[i] * y[i+1]
    end
    return y
end


function _min_eig_tridiag(diag::Vector{Float64}, offdiag::Vector{Float64})
    k = length(diag)
    k == 0 && return 0.0
    k == 1 && return diag[1]
    T = SymTridiagonal(diag, offdiag[1:k-1])
    try
        return minimum(eigvals(T))
    catch
        return 0.0
    end
end

# Workspace version of _eval_F_only.
# It performs the same shifted-diagonal construction and Thomas solve,
# but writes into caller-owned buffers.
function _eval_F_only_ws!(diag_shifted::AbstractVector{Float64},
                          u::AbstractVector{Float64},
                          c_prime::AbstractVector{Float64},
                          d_prime::AbstractVector{Float64},
                          diag::AbstractVector{Float64},
                          offdiag::AbstractVector{Float64},
                          e1::AbstractVector{Float64},
                          lam::Float64,
                          sigma::Float64,
                          g_norm::Float64)
    j = length(diag)

    @inbounds for i in 1:j
        di = diag[i] + lam
        diag_shifted[i] = di
        if di <= 1e-15
            return nothing, nothing
        end
    end

    try
        _solve_tridiagonal_ws!(u, diag_shifted, offdiag, e1, c_prime, d_prime)
    catch
        return nothing, nothing
    end

    u_norm = norm(u)
    if !isfinite(u_norm)
        return nothing, nothing
    end
    F_val = lam - sigma * g_norm * u_norm
    return F_val, u_norm
end

function _solve_target_ratio_ws!(s::SRN_CG,
                                 diag::Vector{Float64},
                                 offdiag::Vector{Float64},
                                 sigma::Float64,
                                 g_norm::Float64,
                                 a_j::Float64,
                                 diag_shifted_buf::Vector{Float64},
                                 e1_buf::Vector{Float64},
                                 u_buf::Vector{Float64},
                                 w_buf::Vector{Float64},
                                 c_prime_buf::Vector{Float64},
                                 d_prime_buf::Vector{Float64},
                                 y_result_buf::Vector{Float64})
    j = length(diag)
    j == 0 && return nothing, nothing, false

    # The original routine trims/pads offdiag to j-1.  In this algorithm
    # offdiag can contain the saved next Lanczos coefficient, so use exactly
    # the first j-1 entries without modifying the stored vector.
    off = @view offdiag[1:max(j-1, 0)]

    diag_shifted = @view diag_shifted_buf[1:j]
    e1 = @view e1_buf[1:j]
    u = @view u_buf[1:j]
    w = @view w_buf[1:j]
    c_prime = @view c_prime_buf[1:j]
    d_prime = @view d_prime_buf[1:j]
    y_result = @view y_result_buf[1:j]

    fill!(e1, 0.0)
    e1[1] = 1.0

    # Newton must start strictly to the right of the spectral lower bound
    # a_j, but on the left side of the scalar root: F(lam) < 0.
    #
    # IMPORTANT: shrink the GAP lam - a_j, not lam itself.  Multiplying lam
    # directly by 0.1 can cross below a_j when a_j > 0 and destroy positive
    # definiteness of T_j + lam*I.
    # Use the large-lambda asymptotic root scale
    #
    #     lambda_* ~ sqrt(sigma * ||g||),
    #
    # as a candidate scale instead of always starting extremely close to
    # a_j.  We still REQUIRE a certified left start F(lambda) < 0 before
    # Newton.  If the candidate lies to the right of the root, contract
    # only the gap lambda-a_j until it lies on the left.
    asymptotic_scale = sqrt(max(sigma, 0.0)) * sqrt(max(g_norm, 0.0))
    base_gap = max(1e-6, 0.01 * max(1.0, abs(a_j)))
    gap = max(base_gap, asymptotic_scale)
    lam = a_j + gap
    found_left_start = false

    for _ in 1:32
        F_val, u_norm = _eval_F_only_ws!(
            diag_shifted, u, c_prime, d_prime,
            diag, off, e1, lam, sigma, g_norm
        )

        if F_val !== nothing && isfinite(F_val) && F_val < 0
            found_left_start = true
            break
        end

        gap *= 0.5
        lam_new = a_j + gap

        if !(lam_new > a_j)
            break
        end
        lam = lam_new
    end

    found_left_start || return nothing, nothing, false

    for it in 1:s.max_newton
        F_val, u_norm = _eval_F_only_ws!(
            diag_shifted, u, c_prime, d_prime,
            diag, off, e1, lam, sigma, g_norm
        )
        F_val === nothing && return nothing, nothing, false

        # Original code rebuilds diag .+ lam and solves for u again here.
        @inbounds for i in 1:j
            diag_shifted[i] = diag[i] + lam
        end

        try
            _solve_tridiagonal_ws!(u, diag_shifted, off, e1, c_prime, d_prime)
            _solve_tridiagonal_ws!(w, diag_shifted, off, u, c_prime, d_prime)
        catch
            return nothing, nothing, false
        end

        u_norm_safe = max(u_norm, 1e-30)
        F_prime = 1.0 + sigma * g_norm * dot(u, w) / u_norm_safe
        F_prime < 1e-15 && return nothing, nothing, false

        step = F_val / F_prime
        lam_new = lam - step

        # Spectral safeguard: every Newton iterate must remain strictly above
        # a_j so that T_j + lam*I stays on the admissible side.
        lam_new <= a_j + 1e-15 && (lam_new = a_j + 0.5 * (lam - a_j))

        if abs(step) < s.tol_newton * max(1.0, abs(lam))
            s.newton_iters_total += it

            @inbounds for i in 1:j
                diag_shifted[i] = diag[i] + lam_new
            end
            try
                _solve_tridiagonal_ws!(u, diag_shifted, off, e1, c_prime, d_prime)
            catch
                return nothing, nothing, false
            end

            # This is the one output vector that must survive the workspace.
            @inbounds for i in 1:j
                y_result[i] = g_norm * u[i]
            end
            return lam_new, copy(y_result), true
        end
        lam = lam_new
    end

    s.newton_iters_total += s.max_newton
    @inbounds for i in 1:j
        diag_shifted[i] = diag[i] + lam
    end
    try
        _solve_tridiagonal_ws!(u, diag_shifted, off, e1, c_prime, d_prime)
    catch
        return nothing, nothing, false
    end

    @inbounds for i in 1:j
        y_result[i] = g_norm * u[i]
    end
    return lam, copy(y_result), true
end

function _projection_ratio_from_mat(Q_mat::AbstractMatrix{Float64},
                                     g_new::Vector{Float64})
    g_norm_sq = dot(g_new, g_new)
    g_norm_sq < 1e-30 && return 1.0
    proj = Q_mat' * g_new
    proj_sq = dot(proj, proj)
    proj_sq > g_norm_sq && (proj_sq = g_norm_sq)
    return min(sqrt(max(0.0, proj_sq) / g_norm_sq), 1.0)
end


# Extend the Lanczos basis directly in the preallocated basis matrix.
function _extend_krylov_qdirect!(s::SRN_CG, nlp::N, x_n,
                                  Q::Matrix{Float64},
                                  j::Int,
                                  diag_list::Vector{Float64},
                                  offdiag_list::Vector{Float64},
                                  delta_j::Float64,
                                  Hq_work::Vector{Float64},
                                  w_work::Vector{Float64}) where {N<:ClosureNLPModel}
    if delta_j <= 0
        return 0.0, j, true
    end

    j_new = j + 1
    q_prev = @view Q[:, j_new - 1]
    q_curr = @view Q[:, j_new]
    q_next = @view Q[:, j_new + 1]

    theta_new, delta_new, broke_new =
        _lanczos_extension_qdirect!(s, nlp, x_n, q_prev, q_curr, delta_j,
                                    Hq_work, w_work, q_next)

    push!(diag_list, theta_new)

    if !broke_new
        push!(offdiag_list, delta_new)
        return delta_new, j_new, false
    else
        return 0.0, j_new, true
    end
end

"""
    optimize!(solver, nlp; initial_theta=nothing, max_iter=nothing, tol=1e-5)

Run SRN-CG on an `NLPModels.jl` model. If `initial_theta` is omitted, a zero
vector is used. Set `solver.max_krylov == 0` to allow up to `nvar` Krylov
directions.
"""
function optimize!(s::SRN_CG, nlp;
                   initial_theta::Union{Nothing, Vector{Float64}} = nothing,
                   max_iter::Union{Nothing, Int} = nothing,
                   tol::Float64 = 1e-5)
    _reset_stats!(s)

    dim = nlp.meta.nvar
    max_outer = max_iter === nothing ? s.max_outer : max_iter
    eps_stop = tol

    x_n = initial_theta === nothing ? zeros(dim) : copy(initial_theta)
    max_krylov = s.max_krylov == 0 ? dim : min(s.max_krylov, dim)

    # Krylov storage reused across outer iterations.
    Q = Matrix{Float64}(undef, dim, max_krylov + 1)

    # Lanczos workspaces reused across all outer/inner iterations.
    lanczos_Hq    = Vector{Float64}(undef, dim)
    lanczos_w     = Vector{Float64}(undef, dim)

    # Workspaces for the reduced target-ratio Newton solve.
    tr_diag_shifted = Vector{Float64}(undef, max_krylov)
    tr_e1           = zeros(Float64, max_krylov)
    tr_u            = Vector{Float64}(undef, max_krylov)
    tr_w            = Vector{Float64}(undef, max_krylov)
    tr_cprime       = Vector{Float64}(undef, max_krylov)
    tr_dprime       = Vector{Float64}(undef, max_krylov)
    tr_yresult      = Vector{Float64}(undef, max_krylov)

    # Trial-point workspaces. These remove one x-vector and one gradient-vector
    # allocation from every trial evaluation.
    x_trial_work = Vector{Float64}(undef, dim)
    g_trial_work = Vector{Float64}(undef, dim)

    # Reusable Krylov-initialization vectors.
    # Keep q_curr arithmetic as scalar division, matching:
    #     -g_n ./ max(g_norm, 1e-15)
    # Do NOT replace division by multiplication with a reciprocal.
    q_prev_work = zeros(dim)
    q_curr_work = Vector{Float64}(undef, dim)

    # Reusable output for the reduced BLAS matrix-vector product Q_mat * y_j.
    # The multiplication itself remains BLAS mul!; only the result storage is reused.
    reduced_matvec_work = Vector{Float64}(undef, dim)

    start_time = time()


    f_n = begin
        val = obj(nlp, x_n)
        s.func_count += 1
        val
    end

    g_n = Vector{Float64}(undef, dim)
    begin
        grad!(nlp, x_n, g_n)
        s.grad_count += 1
    end

    g_norm = norm(g_n)
    g_norm_sq = g_norm * g_norm

    push!(s.loss_history, f_n)
    push!(s.grad_norm_history, g_norm)
    push!(s.time_history, time() - start_time)

    sigma_ref = s.sigma_0_ref

    n = 0
    for n in 1:max_outer
        g_norm_sq <= eps_stop^2 && break

        # Preserve the original initialization arithmetic while reusing storage.
        q_prev = q_prev_work
        q_curr = q_curr_work
        delta_prev = 0.0

        begin
            fill!(q_prev, 0.0)
            denom = max(g_norm, 1e-15)
            @inbounds for i in eachindex(q_curr, g_n)
                q_curr[i] = -g_n[i] / denom
            end
            copyto!(@view(Q[:, 1]), q_curr)
        end

        diag_list = Float64[]
        offdiag_list = Float64[]

        # Exact original Lanczos recurrence, unchanged.
        q_next_col = @view Q[:, 2]
        theta_1, delta_1, broke_1 = begin
                _lanczos_extension_qdirect!(
                    s, nlp, x_n, q_prev, q_curr, delta_prev,
                    lanczos_Hq, lanczos_w, q_next_col
                )
            end
        push!(diag_list, theta_1)

        local delta_j::Float64
        if broke_1
            delta_j = 0.0
        else
            push!(offdiag_list, delta_1)
            delta_j = delta_1
        end

        j = 1
        Q_mat = @view Q[:, 1:j]
        a_j = begin
            max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
        end

        sigma = sigma_ref
        accepted = false

        local last_lambda::Float64 = 0.0
        local last_v::Vector{Float64} = zeros(dim)
        local last_f_trial::Float64 = 0.0
        local last_g_trial::Vector{Float64} = zeros(dim)
        local last_g_trial_norm::Float64 = 0.0

        max_inner = max_krylov * 3

        for inner_iter in 1:max_inner
            isempty(diag_list) && break

            lam, y_j, ok = begin
                _solve_target_ratio_ws!(
                    s, diag_list, offdiag_list, sigma, g_norm, a_j,
                    tr_diag_shifted, tr_e1, tr_u, tr_w,
                    tr_cprime, tr_dprime, tr_yresult
                )
            end

            if !ok || lam === nothing || y_j === nothing
                if delta_j > 0 && j < max_krylov

                    delta_j, j, broke = begin
                            _extend_krylov_qdirect!(
                                s, nlp, x_n, Q, j, diag_list, offdiag_list,
                                delta_j, lanczos_Hq, lanczos_w
                            )
                        end
                    s.krylov_extensions += 1

                    Q_mat = @view Q[:, 1:j]
                    a_j = begin
                        max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                    end
                    continue
                else

                    sigma *= s.rho
                    s.ratio_increases += 1
                    continue
                end
            end

            # Preserve the BLAS matrix-vector arithmetic, but reuse the output storage.
            # Q_mat remains the same column view and y_j is unchanged.
            v = reduced_matvec_work
            begin
                mul!(v, Q_mat, y_j)
            end
            v_norm = norm(v)

            if v_norm < 1e-15
                if delta_j > 0 && j < max_krylov

                    delta_j, j, broke = begin
                            _extend_krylov_qdirect!(
                                s, nlp, x_n, Q, j, diag_list, offdiag_list,
                                delta_j, lanczos_Hq, lanczos_w
                            )
                        end
                    s.krylov_extensions += 1

                    Q_mat = @view Q[:, 1:j]
                    a_j = begin
                        max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                    end
                    continue
                else

                    sigma *= s.rho
                    s.ratio_increases += 1
                    continue
                end
            end

            kappa_cur = lam / max(v_norm, 1e-15)
            if kappa_cur > 10.0 * sigma

                sigma *= s.rho
                s.ratio_increases += 1
                continue
            end

            # In-place trial point: x_trial_work = x_n + v.
            begin
                @inbounds @simd for i in eachindex(x_n, v)
                    x_trial_work[i] = x_n[i] + v[i]
                end
            end

            f_trial = begin
                val = obj(nlp, x_trial_work)
                s.func_count += 1
                val
            end

            begin
                grad!(nlp, x_trial_work, g_trial_work)
                s.grad_count += 1
            end

            g_trial_norm = norm(g_trial_work)

            if !isfinite(f_trial) || !isfinite(g_trial_norm)

                sigma *= s.rho
                s.ratio_increases += 1
                continue
            end

            rho_proj = begin
                _projection_ratio_from_mat(Q_mat, g_trial_work)
            end

            proj_ok, descent_ok, grad_ok = begin
                    p = (rho_proj >= s.tau) || (g_trial_norm < 1e-30)
                    d = f_trial <= f_n - s.beta * lam * v_norm^2
                    g = lam * v_norm >= s.gamma * g_trial_norm
                    (p, d, g)
                end

            if proj_ok && descent_ok && grad_ok && kappa_cur <= sigma

                accepted = true
                last_lambda = lam
                last_v = v
                last_f_trial = f_trial
                # The workspace will be reused, so preserve the accepted
                # gradient exactly as the original allocating grad() did.
                copyto!(last_g_trial, g_trial_work)
                last_g_trial_norm = g_trial_norm
                break
            end

            if proj_ok && descent_ok && grad_ok && kappa_cur > sigma

                sigma *= s.rho
                s.ratio_increases += 1
                continue
            end

            if !proj_ok && delta_j > 0 && j < max_krylov
                # Projection-batch variant:
                # after a projection failure, extend an adaptive number (1, 3, or 5)
                # of Krylov directions consecutively.  Do NOT recompute lambda, Q*y, objective,
                # gradient, or the projection test between these extensions.
                #
                # This intentionally changes the algorithmic trajectory relative
                # to the one-extension-at-a-time baseline and is meant to test
                # whether intermediate projection checks are unnecessarily costly.

                # Adaptive projection batch.  A near-pass receives only one
                # extra direction; a moderate miss receives three; a severe
                # miss retains the original five-direction batch.
                projection_fraction = rho_proj / max(s.tau, eps(Float64))
                projection_batch =
                    projection_fraction >= 0.75 ? 1 :
                    projection_fraction >= 0.25 ? 3 : 5

                for _ in 1:projection_batch
                    (j >= max_krylov || delta_j <= 0.0) && break

                    delta_j, j, broke = begin
                            _extend_krylov_qdirect!(
                                s, nlp, x_n, Q, j, diag_list, offdiag_list,
                                delta_j, lanczos_Hq, lanczos_w
                            )
                        end

                    s.krylov_extensions += 1

                    # Lanczos breakdown means that there is no additional
                    # Krylov direction to generate in this outer iteration.
                    broke && break
                end

                # Only after the whole batch has been generated do we rebuild
                # the active view and update the spectral lower bound.  The
                # next inner-loop pass then solves the target-ratio equation
                # once at this enlarged Krylov dimension.
                Q_mat = @view Q[:, 1:j]
                a_j = begin
                    max(0.0, -_min_eig_tridiag(diag_list, offdiag_list))
                end
                continue
            end

            sigma *= s.rho
            s.ratio_increases += 1
            continue
        end

        if accepted
            begin
                kappa_n = last_lambda / max(norm(last_v), 1e-15)
                push!(s.accepted_kappa, kappa_n)
                push!(s.attained_dims, j)

                @inbounds @simd for i in eachindex(x_n, last_v)
                    x_n[i] += last_v[i]
                end
                f_n = last_f_trial
                g_n = last_g_trial
                g_norm = last_g_trial_norm
                g_norm_sq = g_norm * g_norm

                sigma_ref = max(s.sigma_min, kappa_n / s.rho)
            end
        else

            begin
                fallback_scale = -1e-6 / max(g_norm, 1e-15)
                @inbounds @simd for i in eachindex(x_n, g_n)
                    x_n[i] += fallback_scale * g_n[i]
                end
                f_n = obj(nlp, x_n); s.func_count += 1
                g_n = grad(nlp, x_n); s.grad_count += 1
                g_norm = norm(g_n)
                g_norm_sq = g_norm * g_norm
                sigma_ref = max(s.sigma_min, sigma_ref / s.rho)
            end
        end

        push!(s.loss_history, f_n)
        push!(s.grad_norm_history, g_norm)
        push!(s.time_history, time() - start_time)
    end

    s.total_time = time() - start_time

    return x_n
end

# end # module SRNCG
