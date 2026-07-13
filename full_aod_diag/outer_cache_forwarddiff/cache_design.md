# Exact-point outer-evaluation cache — design

Implementation: `cached_outer_loop.jl`. Struct actually used (Julia, not pseudocode):

```julia
mutable struct OuterEvalCache
    use_cache::Bool

    theta::Vector{Float64}      # cache-owned copy of the last-solved theta (never a KNITRO buffer alias)
    theta_set::Bool
    x::Vector{Float64}          # inner solution (ζ,λ) at `theta`
    objSol::Float64             # K(theta) at the inner solution
    nStatus::Int                # inner KNITRO status at `theta`

    grad::Vector{Float64}       # divergence-constraint gradient at `theta`, if computed
    constr::Vector{Float64}     # outer constraint values at `theta`
    jac::Vector{Float64}        # flattened outer constraint Jacobian at `theta`
    grad_jac_set::Bool          # is (grad,constr,jac) valid for the CURRENT `theta`?

    n_callback::Int; n_unique_theta::Int; n_inner_solve::Int; n_grad_compute::Int
    n_cache_hit::Int; n_cache_miss::Int
    t_inner::Float64; t_grad::Float64; t_other::Float64

    prev_theta::Union{Nothing,Vector{Float64}}
    trace::Vector{CallbackTraceRow}
end
```

## 6A. Exact keying

`exact_same(a,b) = length(a)==length(b) && all(a .== b)` — plain Float64 `==`, no tolerance.
`ensure_inner!` copies `θ` into `cache.theta` (`cache.theta .= θ`) rather than storing a
reference, so a later in-place mutation of KNITRO's `evalRequest.x` buffer cannot silently
invalidate the cache key.

## 6B. Configuration validity

This cache is **one `OuterEvalCache` per `outer_loop_instrumented` call**, i.e. per (obj,
θ_lb, θ_ub, θ_init, `.opt` file) combination — never shared across separate `outer_loop`
invocations, and therefore never shared between a lower-bound and an upper-bound solve, or
across a change of `find_smallest`, `δ`, `U`, `moments!`, or any other field of `obj`. No global
signature hashing was needed because the cache's lifetime is scoped to a single outer solve by
construction (a fresh `OuterEvalCache(...)` is built inside `outer_loop_instrumented` every
call). If a future use case needs a cache that survives ACROSS separate `outer_loop` calls (e.g.
multi-start), it would need an explicit signature — deliberately not built here since nothing in
the current audit needs it, per the instruction not to add machinery beyond what's justified.

## 6C. Lazy computation

`ensure_inner!` always runs (every callback needs at least the objective/constraint value).
`ensure_grad!` is called ONLY from `cb_G!`/`cb_FG!` (never from `cb_F!`), and internally checks
`cache.grad_jac_set` — a fresh inner solve unconditionally clears this flag
(`cache.grad_jac_set = false` inside `ensure_inner!`'s miss branch), so a stale gradient from a
previous θ can never leak through even if `ensure_grad!` is called at a genuinely new θ before
`ensure_inner!` "sees" it (in practice `ensure_inner!` always runs first in every registered
callback, but the invariant is enforced at the data-structure level, not by call-order
convention alone).

## 6D. Callback compatibility (`eval_fcga`, `hessopt`, `algorithm`)

`outer_loop_instrumented` reads `KN_get_int_param(kc, "eval_fcga")` from the SAME `.opt` file the
caller supplies, exactly as production's `outer_loop` does, and registers `cb_FG!` or
`cb_F!`+`cb_G!` accordingly — the cache is shared across both registration shapes because the
cache lives on `userParams` (`CachedProblem`), not on which callback fired. This was the actual
point of the exercise: production's `eval_fcga=no` path (2 callbacks/point) and the diagnostics'
`eval_fcga=yes` path (1 callback/point) both collapse to ~1 inner solve/point once cached. No
`hessopt`/`algorithm` interaction was found or expected — the cache only intercepts the F/G/Jac
evaluation, never the Hessian callback (`callbackEvalH_inner!`, which belongs to the INNER
KNITRO problem, a separate `KN_new()` instance entirely, not the outer one this cache wraps).

## 6E. Thread safety

Not addressed — `outer_loop_instrumented`, like production's `outer_loop`, assumes a single
serial KNITRO solve (`par_numthreads` in the `.opt` files controls KNITRO's OWN internal
parallelism, e.g. multi-start or algorithm-level threading, not concurrent callback invocation
from multiple Julia tasks). No lock was added because nothing in this codebase calls outer
callbacks concurrently; if that ever changes, `OuterEvalCache` would need either a lock around
`ensure_inner!`/`ensure_grad!` or one cache per thread — deliberately not built speculatively.

## Correctness guarantee

`ensure_inner!`/`ensure_grad!` never call any NEW numerical code — they call the exact same
`inner_loop_internal` and `(Q::PsiObjectiveBundleImplicit)(...)` callable production uses,
just conditionally. On a cache hit, the returned `(objSol, x, nStatus)` are byte-identical to
what a fresh solve at that θ would have produced (they ARE that fresh solve's output, stored).
Empirically: `κ` from the cached and uncached D=4 runs matched to `0.912028119649117` exactly
(15+ significant digits, not just "close").
