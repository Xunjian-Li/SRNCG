using LinearAlgebra
using Random
using Printf
using NLPModels

const REFERENCES = Dict(
    "Rosenbrock" => (
        iter=2517, hvp=15148, nobj=7566, ngrad=7566,
        f=3.17346662737801853e-24,
        g=3.59321285318693075e-11,
    ),
    "LogSumExp" => (
        iter=54, hvp=419, nobj=206, ngrad=206,
        f=1.462335784021944,
        g=8.358669231604742e-8,
    ),
    "Polytope" => (
        iter=51, hvp=98, nobj=112, ngrad=112,
        f=9.23044694464708764e-12,
        g=4.65727981150681864e-08,
    ),
    "TRegression" => (
        iter=193, hvp=1230, nobj=708, ngrad=708,
        f=1.39040741191090160e+03,
        g=2.50859371341009197e-08,
    ),
)

function build_problems()
    problems = NamedTuple[]

    x0, o, g, hp = make_rosenbrock(1000; x0=zeros(1000))
    push!(problems, (name="Rosenbrock", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_logsumexp(200, 1000, 0.05; x0=zeros(200))
    push!(problems, (name="LogSumExp", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_polytope(400, 500, 4; x0=zeros(400))
    push!(problems, (name="Polytope", x0=x0, obj=o, grad=g, hprod=hp))

    x0, o, g, hp = make_tregression(200, 500, 0.001)
    push!(problems, (name="TRegression", x0=x0, obj=o, grad=g, hprod=hp))

    return problems
end

function run_srncg(problem)
    nlp = SRNCG.ClosureNLPModel(
        copy(problem.x0), problem.obj, problem.grad, problem.hprod;
        name=problem.name,
    )

    solver = SRNCG.SRN_CG(max_outer=10000)
    reset!(nlp)

    elapsed = @elapsed x = SRNCG.optimize!(
        solver, nlp;
        initial_theta=copy(problem.x0),
        max_iter=10000,
        tol=1e-7,
    )

    # Evaluate final values only after recording algorithm counters.
    nobj = nlp.counters.neval_obj
    ngrad = nlp.counters.neval_grad
    f_final = obj(nlp, x)
    g_final = norm(grad(nlp, x))

    return (
        time=elapsed,
        f=f_final,
        g=g_final,
        iter=length(solver.loss_history) - 1,
        hvp=solver.hvp_count,
        nobj=nobj,
        ngrad=ngrad,
    )
end

same_float(a, b; rtol=5e-5, atol=1e-14) =
    isapprox(a, b; rtol=rtol, atol=atol)

function main()
    println("="^110)
    println("SRN-CG clean GitHub build: regression check against frozen production reference")
    println("="^110)
    @printf("%-14s %9s %9s %9s %9s %13s %13s %10s %10s\n",
            "Problem", "Iter", "HVP", "Obj", "Grad", "f(x*)", "|g(x*)|", "Time(s)", "Status")
    println("-"^110)

    all_pass = true

    for p in build_problems()
        r = run_srncg(p)
        ref = REFERENCES[p.name]

        counts_ok =
            r.iter == ref.iter &&
            r.hvp == ref.hvp &&
            r.nobj == ref.nobj &&
            r.ngrad == ref.ngrad

        values_ok =
            same_float(r.f, ref.f) &&
            same_float(r.g, ref.g)

        pass = counts_ok && values_ok
        all_pass &= pass

        @printf("%-14s %9d %9d %9d %9d %13.4e %13.4e %10.4f %10s\n",
                p.name, r.iter, r.hvp, r.nobj, r.ngrad,
                r.f, r.g, r.time, pass ? "PASS" : "FAIL")

        if !pass
            @printf("  reference: Iter=%d HVP=%d Obj=%d Grad=%d f=%.4e |g|=%.4e\n",
                    ref.iter, ref.hvp, ref.nobj, ref.ngrad, ref.f, ref.g)
        end
    end

    println("-"^110)
    if all_pass
        println("PASS.")
    else
        println("FAIL: at least one trajectory/result differs from the frozen reference.")
        error("SRN-CG regression check failed.")
    end
    println("="^110)
end

main()
