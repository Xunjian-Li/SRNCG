using LinearAlgebra
using Random
using Printf
using NLPModels

export make_rosenbrock, make_logsumexp, make_polytope, make_tregression

import NLPModels: reset!, obj, grad

function make_rosenbrock(dim::Int = 200; x0 = ones(dim) * 2)
    function my_obj(x)
        s = 0.0
        @inbounds for i in 1:length(x)-1
            s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
        end
        return s
    end
    function my_grad!(g, x)
        n = length(x)
        fill!(g, 0.0)
        @inbounds for i in 1:n-1
            g[i]   += -400.0 * x[i] * (x[i+1] - x[i]^2) - 2.0 * (1.0 - x[i])
            g[i+1] +=  200.0 * (x[i+1] - x[i]^2)
        end
    end
    function my_hprod!(Hv, x, v)
        n = length(x)
        fill!(Hv, 0.0)
        @inbounds for i in 1:n-1
            t = x[i+1] - x[i]^2
            Hv[i]   += (-400.0 * t + 800.0 * x[i]^2 + 2.0) * v[i]
            Hv[i]   += (-400.0 * x[i]) * v[i+1]
            Hv[i+1] += (-400.0 * x[i]) * v[i]
            Hv[i+1] += ( 200.0) * v[i+1]
        end
    end
    return x0, my_obj, my_grad!, my_hprod!
end

function make_logsumexp(dim::Int = 200, n::Int = 1000, rho::Float64 = 0.05;
                        seed::Int = 42, x0 = zeros(dim))
    rng = MersenneTwister(seed)
    A = randn(rng, n, dim)
    b = randn(rng, n)

    # Objective -> gradient cache.
    # Only z = A*x - b is shared. The validated gradient arithmetic
    # (maximum, exp, sum, normalization, and A' * w) remains unchanged.
    obj_linear = Vector{Float64}(undef, n)
    grad_linear = Vector{Float64}(undef, n)
    grad_w = Vector{Float64}(undef, n)
    obj_x = similar(x0)
    obj_x_valid = Ref(false)

    function my_obj(x)
        mul!(obj_linear, A, x)
        @inbounds @simd for i in eachindex(obj_linear, b)
            obj_linear[i] -= b[i]
        end

        # Cache the point associated with obj_linear and grad_w.
        copyto!(obj_x, x)
        obj_x_valid[] = true

        mx = maximum(obj_linear)
        s = 0.0
        @inbounds @simd for i in eachindex(obj_linear, grad_w)
            wi = exp((obj_linear[i] - mx) / rho)
            grad_w[i] = wi
            s += wi
        end
        return rho * (log(s) + mx / rho)
    end

    function my_grad!(g, x)
        if obj_x_valid[] && isequal(x, obj_x)
            # Fused objective-gradient path:
            # obj_linear and the unnormalized exponentials in grad_w
            # were already computed by my_obj at exactly this x.
            copyto!(grad_linear, obj_linear)
        else
            # Preserve the original standalone-gradient path.
            mul!(grad_linear, A, x)
            @inbounds @simd for i in eachindex(grad_linear, b)
                grad_linear[i] -= b[i]
            end
            mx = maximum(grad_linear)
            @inbounds @simd for i in eachindex(grad_w, grad_linear)
                grad_w[i] = exp((grad_linear[i] - mx) / rho)
            end
        end

        # Preserve the validated normalization and A' * w arithmetic.
        sw = sum(grad_w)
        @inbounds @simd for i in eachindex(grad_w)
            grad_w[i] /= sw
        end

        mul!(g, transpose(A), grad_w)
        return g
    end

    # HVP cache: w(x) and A'w(x) are constant for all Lanczos products
    # performed at the same outer iterate x.
    hess_x = similar(x0)
    hess_x_valid = Ref(false)
    hess_w = Vector{Float64}(undef, n)
    hess_Atw = Vector{Float64}(undef, dim)
    hess_linear = Vector{Float64}(undef, n)
    hess_Av = Vector{Float64}(undef, n)
    hess_tmp = Vector{Float64}(undef, n)

    function refresh_hess_cache!(x)
        if !hess_x_valid[] || !isequal(x, hess_x)
            mul!(hess_linear, A, x)
            @inbounds @simd for i in eachindex(hess_linear, b)
                hess_linear[i] -= b[i]
            end
            mx = maximum(hess_linear)
            sw = 0.0
            @inbounds for i in eachindex(hess_w, hess_linear)
                wi = exp((hess_linear[i] - mx) / rho)
                hess_w[i] = wi
                sw += wi
            end
            invsw = inv(sw)
            @inbounds @simd for i in eachindex(hess_w)
                hess_w[i] *= invsw
            end
            mul!(hess_Atw, transpose(A), hess_w)
            copyto!(hess_x, x)
            hess_x_valid[] = true
        end
        return nothing
    end

    function my_hprod!(Hv, x, v)
        refresh_hess_cache!(x)
        mul!(hess_Av, A, v)
        wAv = dot(hess_w, hess_Av)
        @inbounds @simd for i in eachindex(hess_tmp, hess_w, hess_Av)
            hess_tmp[i] = hess_w[i] * hess_Av[i]
        end
        mul!(Hv, transpose(A), hess_tmp)
        @inbounds @simd for i in eachindex(Hv, hess_Atw)
            Hv[i] = (Hv[i] - hess_Atw[i] * wAv) / rho
        end
        return Hv
    end
    return x0, my_obj, my_grad!, my_hprod!
end

function make_polytope(dim::Int = 400, m::Int = 1000, p::Int = 3;
                       seed::Int = 42, x0 = ones(dim))
    rng = MersenneTwister(seed)
    A = randn(rng, m, dim)
    b = randn(rng, m)

    res   = Vector{Float64}(undef, m)
    coeff = Vector{Float64}(undef, m)
    Av    = Vector{Float64}(undef, m)
    tmp_m = Vector{Float64}(undef, m)
    obj_x = similar(x0)
    obj_x_valid = Ref(false)

    function fill_res!(x)
        mul!(res, A, x)
        @inbounds @simd for i in eachindex(res, b)
            res[i] -= b[i]
        end
        return res
    end

    function my_obj(x)
        fill_res!(x)
        copyto!(obj_x, x)
        obj_x_valid[] = true
        ss = 0.0
        @inbounds @simd for i in eachindex(res)
            r = res[i] > 0 ? res[i] : 0.0
            ss += r^p
        end
        return ss
    end

    function my_grad!(g, x)
        if !(obj_x_valid[] && isequal(x, obj_x))
            fill_res!(x)
        end
        @inbounds @simd for i in eachindex(res, tmp_m)
            r = res[i] > 0 ? res[i] : 0.0
            tmp_m[i] = p * r^(p - 1)
        end
        mul!(g, transpose(A), tmp_m)
        return g
    end

    hess_x = similar(x0)
    hess_x_valid = Ref(false)
    hess_coeff = similar(coeff)
    hess_res = similar(res)

    function refresh_hess_cache!(x)
        if !hess_x_valid[] || !isequal(x, hess_x)
            mul!(hess_res, A, x)
            @inbounds @simd for i in eachindex(hess_res, b)
                hess_res[i] -= b[i]
            end
            @inbounds @simd for i in eachindex(hess_res, hess_coeff)
                r = hess_res[i] > 0 ? hess_res[i] : 0.0
                hess_coeff[i] = r > 0 ? p * (p - 1) * r^(p - 2) : 0.0
            end
            copyto!(hess_x, x)
            hess_x_valid[] = true
        end
        return nothing
    end

    function my_hprod!(Hv, x, v)
        refresh_hess_cache!(x)
        mul!(Av, A, v)
        @inbounds @simd for i in eachindex(tmp_m, hess_coeff, Av)
            tmp_m[i] = hess_coeff[i] * Av[i]
        end
        mul!(Hv, transpose(A), tmp_m)
        return Hv
    end

    return x0, my_obj, my_grad!, my_hprod!
end

function make_tregression(dim::Int = 200, m::Int = 500, nu::Float64 = 0.001;
                          seed::Int = 42)
    rng = MersenneTwister(seed)
    x_true = randn(rng, dim)
    A_raw = randn(rng, m, dim)
    for i in 2:dim
        A_raw[:, i] .+= 0.3 .* A_raw[:, i-1]
    end
    A = A_raw ./ sqrt(1 + 0.3^2)
    clean = A * x_true
    z = randn(rng, m) .* 0.5
    chi = rand(rng, m) .* 2.0
    noise = z ./ sqrt.(chi)
    b = clean .+ noise
    x0 = A \ b

    res   = Vector{Float64}(undef, m)
    coeff = Vector{Float64}(undef, m)
    Av    = Vector{Float64}(undef, m)
    tmp_m = Vector{Float64}(undef, m)
    obj_x = similar(x0)
    obj_x_valid = Ref(false)

    function fill_res!(x)
        mul!(res, A, x)
        @inbounds @simd for i in eachindex(res, b)
            res[i] -= b[i]
        end
        return res
    end

    function my_obj(x)
        fill_res!(x)
        copyto!(obj_x, x)
        obj_x_valid[] = true
        ss = 0.0
        @inbounds @simd for i in eachindex(res)
            ss += log1p(res[i]^2 / nu)
        end
        return ss
    end

    function my_grad!(g, x)
        if !(obj_x_valid[] && isequal(x, obj_x))
            fill_res!(x)
        end
        @inbounds @simd for i in eachindex(res, tmp_m)
            ri = res[i]
            tmp_m[i] = (2 * ri / nu) / (1 + ri^2 / nu)
        end
        mul!(g, transpose(A), tmp_m)
        return g
    end

    hess_x = similar(x0)
    hess_x_valid = Ref(false)
    hess_coeff = similar(coeff)
    hess_res = similar(res)

    function refresh_hess_cache!(x)
        if !hess_x_valid[] || !isequal(x, hess_x)
            mul!(hess_res, A, x)
            @inbounds @simd for i in eachindex(hess_res, b)
                hess_res[i] -= b[i]
            end
            @inbounds @simd for i in eachindex(hess_res, hess_coeff)
                r2nu = hess_res[i]^2 / nu
                denom = 1 + r2nu
                hess_coeff[i] = (2 / nu * (1 - r2nu)) / (denom^2)
            end
            copyto!(hess_x, x)
            hess_x_valid[] = true
        end
        return nothing
    end

    function my_hprod!(Hv, x, v)
        refresh_hess_cache!(x)
        mul!(Av, A, v)
        @inbounds @simd for i in eachindex(tmp_m, hess_coeff, Av)
            tmp_m[i] = hess_coeff[i] * Av[i]
        end
        mul!(Hv, transpose(A), tmp_m)
        return Hv
    end

    return x0, my_obj, my_grad!, my_hprod!
end
