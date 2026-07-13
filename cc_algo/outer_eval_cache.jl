# ============================================================================
# Production exact-point outer-loop cache.
#
# Generalizes the validated diagnostic version
# (full_aod_diag/outer_cache_forwarddiff/cached_outer_loop.jl — see
# full_aod_diag/outer_cache_forwarddiff/README.md for the D=4 audit that
# found production's real eval_fcga=no config double-solves the inner CC
# problem at every accepted outer iterate, and validated this design: 44%
# duplicate-inner-solve elimination, kappa/opt_err bit-identical, 1.36x wall
# time at D=4). This version is method-agnostic (works for both the
# sequential-profiled and full-A formulations) and keys on the FREE outer
# vector x_free (see free_param_map.jl), not the full theta vector.
#
# Scoping (§3.8-3.9): a fresh OuterEvalCache is constructed inside every call
# to `outer_loop_cached`, so it is automatically scoped to exactly one
# (method, bound direction, delta, draw set, model configuration) — never
# shared across separate outer solves, and in particular never shared between
# an upper-bound and a lower-bound solve, since those are always separate
# `outer_loop_cached` calls with their own `obj`/cache.
# ============================================================================

using KNITRO

mutable struct CallbackTraceRow
    idx::Int
    kind::String
    x_hash::UInt64
    same_as_prev_call::Bool
    cache_hit::Bool
    inner_solved::Bool
    grad_computed::Bool
    warm_started::Bool
    elapsed::Float64
    nstatus::Int
end

mutable struct OuterEvalCache
    use_cache::Bool

    x_free::Vector{Float64}     # cache-owned copy of the free outer vector at the cached point
    x_set::Bool
    inner_x::Vector{Float64}    # inner (eta,zeta,lambda)/(zeta,lambda) solution at x_free
    objSol::Float64
    nStatus::Int

    grad::Vector{Float64}       # divergence-constraint gradient, w.r.t. x_free, at x_free
    constr::Vector{Float64}
    jac::Vector{Float64}        # flattened, w.r.t. x_free
    grad_jac_set::Bool

    n_callback::Int
    n_unique_x::Int
    n_inner_solve::Int
    n_grad_compute::Int
    n_cache_hit::Int
    n_cache_miss::Int
    n_warm_started::Int
    n_cold::Int
    t_inner::Float64
    t_grad::Float64

    prev_x::Union{Nothing,Vector{Float64}}
    trace::Vector{CallbackTraceRow}
end

function OuterEvalCache(n_free::Int, ncon::Int; use_cache::Bool = true)
    OuterEvalCache(use_cache, fill(NaN, n_free), false, Float64[], NaN, -999,
        fill(NaN, n_free), fill(NaN, max(ncon, 0)), fill(NaN, max(ncon, 0) * n_free), false,
        0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, nothing, CallbackTraceRow[])
end

exact_same(a, b) = length(a) == length(b) && all(@. a == b)
x_hash(x) = hash(x)

"""
    ensure_inner!(cache, obj, x_free, θ_full) -> (objSol, inner_x, nStatus, solved, hit, warm)

`θ_full` must already be `reconstruct_full(x_free, map)`. Solves the inner CC
problem via the EXISTING, unmodified `inner_loop_internal(obj, θ_full)` only
if `x_free` does not exactly match the cached point; otherwise reuses the
cached inner solution with zero solver calls.
"""
function ensure_inner!(cache::OuterEvalCache, obj, x_free::AbstractVector, θ_full::AbstractVector)
    hit = cache.use_cache && cache.x_set && exact_same(cache.x_free, x_free)
    solved = false
    warm = obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6
    t0 = time()
    if hit
        cache.n_cache_hit += 1
    else
        cache.n_cache_miss += 1
        cache.n_unique_x += 1
        objSol, inner_x, nStatus = inner_loop_internal(obj, θ_full)
        cache.x_free .= x_free
        cache.x_set = true
        cache.inner_x = copy(inner_x)
        cache.objSol = objSol
        cache.nStatus = nStatus
        cache.grad_jac_set = false
        cache.n_inner_solve += 1
        if warm
            cache.n_warm_started += 1
        else
            cache.n_cold += 1
        end
        solved = true
    end
    cache.t_inner += time() - t0
    return cache.objSol, cache.inner_x, cache.nStatus, solved, hit, warm
end

"""
    ensure_grad!(cache, grad_fn!, x_free) -> (grad, computed)

`grad_fn!(g, x_free)` must fill `g` (length `n_free`) with the divergence-
gradient at `x_free`, using the CURRENTLY-CACHED inner solution (i.e. it must
be called only immediately after `ensure_inner!` at the same `x_free`).
Lazily computed at most once per distinct cached point.
"""
function ensure_grad!(cache::OuterEvalCache, grad_fn!, x_free::AbstractVector)
    computed = false
    t0 = time()
    if !(cache.use_cache && cache.grad_jac_set)
        grad_fn!(cache.grad, x_free)
        cache.grad_jac_set = true
        cache.n_grad_compute += 1
        computed = true
    end
    cache.t_grad += time() - t0
    return cache.grad, computed
end

function log_row!(cache::OuterEvalCache, kind, x_free, inner_solved, grad_computed, warm, elapsed, nstatus)
    same_prev = cache.prev_x !== nothing && exact_same(cache.prev_x, x_free)
    cache.n_callback += 1
    push!(cache.trace, CallbackTraceRow(cache.n_callback, kind, x_hash(x_free), same_prev,
        cache.use_cache && exact_same(cache.x_free, x_free) && !inner_solved, inner_solved, grad_computed,
        warm, elapsed, nstatus))
    cache.prev_x = copy(x_free)
end

function summarize(cache::OuterEvalCache; label = "")
    n_unique = length(Set(r.x_hash for r in cache.trace))
    println("---- ", label, " ----")
    println("total callback invocations : ", cache.n_callback)
    println("unique free-x points       : ", n_unique)
    println("inner solves                : ", cache.n_inner_solve, "  (warm=", cache.n_warm_started, " cold=", cache.n_cold, ")")
    println("gradient computations      : ", cache.n_grad_compute)
    println("cache hits / misses         : ", cache.n_cache_hit, " / ", cache.n_cache_miss)
    println("inner-solve time (s)       : ", round(cache.t_inner, digits = 3))
    println("gradient time (s)          : ", round(cache.t_grad, digits = 3))
    println("inner solves / unique x     : ", round(cache.n_inner_solve / max(n_unique, 1), digits = 3))
    flush(stdout)
end

function write_trace_csv(path, cache::OuterEvalCache)
    open(path, "w") do io
        println(io, "idx,kind,x_hash,same_as_prev_call,cache_hit,inner_solved,grad_computed,warm_started,elapsed,nstatus")
        for r in cache.trace
            println(io, join((r.idx, r.kind, r.x_hash, r.same_as_prev_call, r.cache_hit,
                               r.inner_solved, r.grad_computed, r.warm_started, r.elapsed, r.nstatus), ","))
        end
    end
end
