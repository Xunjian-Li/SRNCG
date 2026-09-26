# SRNCG.jl

**SRNCG.jl** is a Julia implementation of **SRN-CG**, a matrix-free second-order optimization method based on regularized Newton steps and adaptive Krylov subspace approximations.

The method is designed for large-scale smooth optimization problems where explicitly forming or factorizing the Hessian is expensive or impractical. Instead of constructing the full Hessian matrix, SRN-CG accesses second-order information through **Hessian-vector products (HVPs)** and constructs a low-dimensional Lanczos/Krylov approximation of the regularized Newton system.

The package is built on top of [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) and can be used either with an `NLPModels`-compatible optimization model or with the lightweight `ClosureNLPModel` interface provided in this repository.

---

## Features

SRNCG.jl provides:

- matrix-free second-order optimization;
- Hessian-vector-product-based computation;
- adaptive Lanczos/Krylov subspace construction;
- regularized Newton steps;
- reduced-dimensional subproblem solves;
- adaptive regularization;
- compatibility with `NLPModels.jl`;
- a lightweight closure-based interface for custom objectives;
- detailed solver statistics and optimization histories;
- deterministic regression tests;
- benchmark scripts for comparison with several first- and second-order optimization methods.

The core algorithm does **not** require explicit construction or factorization of the Hessian.

---

## Requirements

The current implementation is developed with Julia 1.12.

The main dependencies are:

- `NLPModels.jl`
- `LinearAlgebra`

The repository also includes dependencies used by the regression tests and benchmark scripts:

- `AdaptiveRegularization.jl`
- `BenchmarkTools.jl`
- `TimerOutputs.jl`
- `Random`
- `Printf`

All dependencies are specified in `Project.toml`.

---

## Installation

### Install directly from GitHub

Once the repository is public, the package can be installed with

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SRNCG.git")
```

Then load the package with

```julia
using SRNCG
```

If the repository name or GitHub username is different, replace the URL accordingly.

### Development installation

If you want to modify the source code, clone the repository and activate it locally:

```bash
git clone https://github.com/Xunjian-Li/SRNCG.git
cd SRNCG
```

Start Julia with the project environment:

```bash
julia --project=.
```

Then instantiate the dependencies:

```julia
using Pkg
Pkg.instantiate()
```

and load SRNCG:

```julia
using SRNCG
```

Alternatively, from another Julia environment you can use

```julia
using Pkg
Pkg.develop(path="/path/to/SRNCG")
```

---

# Quick Start

The basic workflow is

```julia
using SRNCG

solver = SRN_CG()

x = optimize!(
    solver,
    nlp;
    initial_theta=x0,
    tol=1e-7,
    max_iter=10_000,
)
```

where:

- `nlp` is an `NLPModels.jl`-compatible model;
- `x0` is the initial point;
- `tol` is the first-order stopping tolerance;
- `max_iter` is the maximum number of outer iterations.

The returned value `x` is the final iterate.

---

# Defining a Problem

SRN-CG requires three basic operations:

1. objective evaluation
   \[
   f(x),
   \]

2. gradient evaluation
   \[
   \nabla f(x),
   \]

3. Hessian-vector product
   \[
   \nabla^2 f(x)v.
   \]

The full Hessian matrix does not need to be formed.

SRNCG.jl provides `ClosureNLPModel` to conveniently wrap these three operations into an `NLPModels` model.

---

## Example: Quadratic Optimization

Consider

\[
f(x)=\frac12 x^\top A x-b^\top x,
\]

where \(A\) is symmetric positive definite.

Its gradient and Hessian-vector product are

\[
\nabla f(x)=Ax-b,
\]

and

\[
\nabla^2 f(x)v=Av.
\]

A complete SRN-CG example is:

```julia
using SRNCG
using LinearAlgebra
using Random

Random.seed!(1234)

n = 100

A = randn(n, n)
A = A' * A + I

b = randn(n)
x0 = zeros(n)

function obj_fn(x)
    return 0.5 * dot(x, A * x) - dot(b, x)
end

function grad_fn!(g, x)
    mul!(g, A, x)
    g .-= b
    return g
end

function hprod_fn!(Hv, x, v)
    mul!(Hv, A, v)
    return Hv
end

nlp = ClosureNLPModel(
    copy(x0),
    obj_fn,
    grad_fn!,
    hprod_fn!;
    name="Quadratic",
)

solver = SRN_CG(max_outer=1000)

x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    max_iter=1000,
    tol=1e-7,
)

println("Final objective      = ", solver.loss_history[end])
println("Final gradient norm  = ", solver.grad_norm_history[end])
println("HVP evaluations      = ", solver.hvp_count)
println("Total time           = ", solver.total_time)
```

For this example, the exact solution is

```julia
x_exact = A \ b
```

so the numerical solution can be checked using

```julia
norm(x - x_exact) / norm(x_exact)
```

---

# `ClosureNLPModel`

The constructor is

```julia
ClosureNLPModel(
    x0,
    obj_fn,
    grad_fn,
    hprod_fn;
    name="Closure",
)
```

The callbacks must have the following forms.

### Objective

```julia
obj_fn(x)
```

returns the scalar objective value.

### Gradient

```julia
grad_fn(g, x)
```

writes the gradient into `g`.

For example,

```julia
function grad_fn!(g, x)
    # compute gradient
    return g
end
```

### Hessian-vector product

```julia
hprod_fn(Hv, x, v)
```

writes

\[
Hv=\nabla^2f(x)v
\]

into `Hv`.

For example,

```julia
function hprod_fn!(Hv, x, v)
    # compute Hessian-vector product
    return Hv
end
```

For large-scale applications, `hprod_fn` should preferably compute the HVP directly without explicitly constructing the Hessian.

---

# Solver Interface

Create a solver with

```julia
solver = SRN_CG()
```

The current constructor is

```julia
SRN_CG(
    sigma_0_ref = 1.0,
    sigma_min   = 1e-8,
    rho         = 2.0,
    beta        = 0.25,
    gamma       = 1e-3,
    tau         = 0.1,
    max_outer   = 500,
    max_krylov  = 0,
    max_newton  = 10,
    tol_newton  = 1e-3,
    verbose     = false,
)
```

The optimization routine is called as

```julia
x = optimize!(
    solver,
    nlp;
    initial_theta=x0,
    max_iter=1000,
    tol=1e-7,
)
```

For example, verbose output can be enabled with

```julia
solver = SRN_CG(verbose=true)
```

---

# Solver Statistics

After calling `optimize!`, computational statistics are stored in the solver object.

Important fields include

```julia
solver.hvp_count
solver.grad_count
solver.func_count

solver.krylov_extensions
solver.ratio_increases
solver.newton_iters_total

solver.accepted_kappa
solver.attained_dims

solver.loss_history
solver.grad_norm_history
solver.time_history
solver.total_time
```

For example:

```julia
println("Iterations          = ", length(solver.loss_history) - 1)
println("HVP evaluations     = ", solver.hvp_count)
println("Objective evals     = ", solver.func_count)
println("Gradient evals      = ", solver.grad_count)
println("Krylov extensions   = ", solver.krylov_extensions)
println("Total time          = ", solver.total_time)

println("Final objective     = ", solver.loss_history[end])
println("Final gradient norm = ", solver.grad_norm_history[end])
```

The histories can also be used to inspect convergence:

```julia
f_history = solver.loss_history
g_history = solver.grad_norm_history
t_history = solver.time_history
```

---

# Included Test Problems

The repository contains several smooth optimization problems used for testing and benchmarking:

- **Rosenbrock**
- **Log-Sum-Exp**
- **Polytope**
- **T-Regression**

They are available through

```julia
make_rosenbrock
make_logsumexp
make_polytope
make_tregression
```

For example:

```julia
using SRNCG

x0, obj_fn, grad_fn!, hprod_fn! =
    make_rosenbrock(1000; x0=zeros(1000))

nlp = ClosureNLPModel(
    copy(x0),
    obj_fn,
    grad_fn!,
    hprod_fn!;
    name="Rosenbrock",
)

solver = SRN_CG(max_outer=10_000)

x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    max_iter=10_000,
    tol=1e-7,
)
```

---

# Regression Test

The repository contains a deterministic regression test that checks the current SRN-CG implementation against frozen numerical reference results.

The regression suite uses:

| Problem | Dimension / Parameters |
|---|---|
| Rosenbrock | \(d=1000\) |
| Log-Sum-Exp | \(d=200,\ n=1000,\ \rho=0.05\) |
| Polytope | \(d=400,\ m=500,\ p=4\) |
| T-Regression | \(d=200,\ m=500,\ \nu=0.001\) |

The stopping tolerance is

\[
\|\nabla f(x)\| \le 10^{-7}.
\]

## Run from Julia

From the repository root:

```julia
using Pkg
Pkg.activate(".")

using SRNCG

include("src/test.jl")
```

## Run from the command line

```bash
julia --project=. src/test.jl
```

A successful regression run should produce results consistent with:

```text
Problem         Iter      HVP      Obj     Grad
------------------------------------------------
Rosenbrock      2517    15148     7566     7566
LogSumExp         54      419      206      206
Polytope          51       98      112      112
TRegression      193     1230      708      708
```

and end with

```text
PASS.
```

These reference values are intended to detect unintended changes in the numerical trajectory of the implementation.

Small wall-clock timing differences across machines are expected.

---

# Benchmark

A benchmark script is included in

```text
src/benchmark.jl
```

The current benchmark compares

- **ARCqK**
- **SRN-CG**
- **Gradient Descent (GD)**
- **Nesterov Accelerated Gradient (NAG)**
- **ANCG**
- **ARNCG**

on the same four test problems.

The benchmark uses a common first-order tolerance

\[
\|\nabla f(x)\|\le 10^{-7},
\]

with a maximum of 10,000 outer iterations.

Wall-clock timing is performed using `BenchmarkTools.jl`.

Each benchmark evaluation corresponds to one complete optimization run:

```text
evals = 1
```

and the default benchmark duration is

```text
5 seconds per method/problem
```

Problem construction and mutable cache initialization are performed outside the timed expression.

Algorithmic counters and final objective/gradient values are obtained from a separate deterministic run.

This separates:

1. **wall-clock performance**, measured using `BenchmarkTools`, and
2. **algorithmic work**, measured using iterations and oracle evaluations.

---

## Running the Benchmark

From Julia:

```julia
using Pkg
Pkg.activate(".")

using SRNCG

include("src/benchmark.jl")
```

or directly from the command line:

```bash
julia --project=. src/benchmark.jl
```

The output contains columns of the form

```text
Problem
Method
Median(s)
Q25(s)
Q75(s)
Samples
Iter
HVP
Obj
Grad
f(x*)
|g(x*)|
Status
```

For example, timing is summarized using the median and interquartile range rather than a single wall-clock measurement.

---

# Benchmark Methodology

When comparing second-order methods, wall-clock time alone can be implementation-dependent. The benchmark therefore reports several measures of computational work.

### Iter

Number of outer iterations.

### HVP

Number of Hessian-vector products

\[
v\mapsto\nabla^2f(x)v.
\]

This is particularly important for matrix-free second-order algorithms.

### Obj

Number of objective evaluations.

### Grad

Number of gradient evaluations.

### Final objective

```text
f(x*)
```

### First-order stationarity

```text
|g(x*)|
```

denotes

\[
\|\nabla f(x^\ast)\|.
\]

A run is reported as converged when the final gradient norm satisfies the benchmark tolerance.

---

# Notes on Benchmark Baselines

The benchmark contains implementations of several comparison methods.

The ARNCG implementation used in the benchmark is a Julia translation based directly on the authors' MATLAB implementation, including the corresponding CappedCG procedure and adaptive regularized Newton logic.

The benchmark implementations are included for numerical comparison and are not part of the core SRNCG API.

Users interested only in SRN-CG do not need to use these benchmark routines.

---

# Repository Structure

The current repository is organized as

```text
SRNCG/
├── Project.toml
├── Manifest.toml
├── README.md
└── src/
    ├── SRNCG.jl
    ├── SRN_CG.jl
    ├── problems.jl
    ├── test.jl
    └── benchmark.jl
```

The files have the following roles.

### `src/SRNCG.jl`

Main package module.

It loads the SRN-CG implementation and the included test-problem constructors.

### `src/SRN_CG.jl`

Core SRN-CG implementation.

This file contains:

- `SRN_CG`
- `ClosureNLPModel`
- `optimize!`
- Lanczos/Krylov operations
- reduced regularized subproblem calculations
- solver statistics and histories

### `src/problems.jl`

Test-problem constructors:

```julia
make_rosenbrock
make_logsumexp
make_polytope
make_tregression
```

### `src/test.jl`

Deterministic regression tests against frozen SRN-CG reference results.

### `src/benchmark.jl`

Benchmark suite comparing SRN-CG with other optimization methods.

---

# Using SRN-CG in Your Own Code

For most users, only three objects are needed:

```julia
using SRNCG

ClosureNLPModel
SRN_CG
optimize!
```

A typical application therefore has the following structure:

```julia
using SRNCG

# 1. Define objective
function f(x)
    # ...
end

# 2. Define gradient
function grad_f!(g, x)
    # ...
    return g
end

# 3. Define Hessian-vector product
function Hv_f!(Hv, x, v)
    # ...
    return Hv
end

# 4. Initial point
x0 = ...

# 5. Build NLPModels-compatible model
nlp = ClosureNLPModel(
    copy(x0),
    f,
    grad_f!,
    Hv_f!;
    name="MyProblem",
)

# 6. Construct SRN-CG solver
solver = SRN_CG()

# 7. Optimize
x = optimize!(
    solver,
    nlp;
    initial_theta=copy(x0),
    tol=1e-7,
    max_iter=10_000,
)

# 8. Inspect the result
println("f(x)   = ", solver.loss_history[end])
println("|g(x)| = ", solver.grad_norm_history[end])
println("HVPs   = ", solver.hvp_count)
```

The main computational requirement is therefore an efficient implementation of

```julia
Hv_f!(Hv, x, v)
```

rather than an explicit Hessian matrix.

This is the intended use case of SRN-CG for large-scale problems.

---

# Reproducibility

The supplied stochastic test problems use fixed random seeds where applicable.

The regression suite checks deterministic quantities such as:

- iteration counts;
- Hessian-vector-product counts;
- objective evaluation counts;
- gradient evaluation counts;
- final objective values;
- final gradient norms.

Wall-clock timings are not used as regression criteria because they depend on hardware, Julia version, BLAS configuration, and system load.

For reliable benchmark timing, it is recommended to run the benchmark in a fresh Julia session.

---

# Citation

If you use SRNCG.jl in academic work, please cite the accompanying SRN-CG paper.

The full citation will be added when the manuscript is publicly available.

A BibTeX entry will be provided here:

```bibtex
@article{SRNCG,
  title   = {SRN-CG},
  author  = {Xun-Jian Li and coauthors},
  journal = {...},
  year    = {...}
}
```

---

# Author

**Xun-Jian Li**

Department of Biostatistics  
University of California, Los Angeles

---

# License

Please see the `LICENSE` file for licensing terms.

If no license has yet been added to the repository, users should not assume permission to redistribute or modify the source code.