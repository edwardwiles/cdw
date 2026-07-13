# ============================================================================
# Additive, isolated outer-loop instrumentation + exact-point cache.
#
# Does NOT modify cc_algo/outer_loop_functions.jl, cc_algo/inner_loop_functions.jl,
# or any other production file. It calls the existing, unmodified
# `inner_loop_internal` and the existing `(Q::PsiObjectiveBundleImplicit)(...)`
# callable exactly as production's callbacks do, but wraps every call in an
# exact-Float64-equality cache and a chronological trace log, and registers its
# own KNITRO eval callbacks (parallel to, not replacing, the production ones in
# outer_loop_functions.jl).
#
# Use `use_cache=false` to get a pure INSTRUMENTED baseline (answers "does
# production duplicate inner solves today?"), and `use_cache=true` to get the
# cached behavior, using the identical KNITRO .opt file and starting point so
# the two runs are apples-to-apples.
# ============================================================================

using KNITRO

mutable struct CallbackTraceRow
    idx::Int
    kind::String            # "F", "G", or "FG"
    theta_hash::UInt64
    same_as_prev_call::Bool # bit-identical theta to the immediately preceding callback call (any kind)
    cache_hit::Bool         # bit-identical theta to the cached point (cache.theta)
    inner_solved::Bool      # did this call trigger a fresh inner_loop_internal solve?
    grad_computed::Bool     # did this call trigger a fresh gradient/Jacobian computation?
    elapsed::Float64
    nstatus::Int
end

mutable struct OuterEvalCache
    use_cache::Bool

    # exact-point cache: valid only for cache.theta (cache-owned copy, never a KNITRO buffer alias)
    theta::Vector{Float64}
    theta_set::Bool
    x::Vector{Float64}
    objSol::Float64
    nStatus::Int

    grad::Vector{Float64}
    constr::Vector{Float64}
    jac::Vector{Float64}
    grad_jac_set::Bool      # gradient+Jacobian valid for the CURRENT cache.theta

    # instrumentation
    n_callback::Int
    n_unique_theta::Int
    n_inner_solve::Int
    n_grad_compute::Int
    n_cache_hit::Int
    n_cache_miss::Int
    t_inner::Float64
    t_grad::Float64
    t_other::Float64

    prev_theta::Union{Nothing,Vector{Float64}}
    trace::Vector{CallbackTraceRow}
end

function OuterEvalCache(l::Int, ncon::Int; use_cache::Bool=true)
    OuterEvalCache(use_cache, fill(NaN, l), false, Float64[], NaN, -999,
        fill(NaN, l), fill(NaN, max(ncon,0)), fill(NaN, max(ncon,0) * l), false,
        0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, nothing, CallbackTraceRow[])
end

# exact element-by-element Float64 equality (NOT a tolerance comparison, per spec)
exact_same(a, b) = length(a) == length(b) && all(@. a == b)

theta_hash(θ) = hash(θ)

# Ensure the inner CC solve is available for θ, reusing the cache on an exact hit.
# Returns (objSol, x, nStatus, solved::Bool, hit::Bool)
function ensure_inner!(cache::OuterEvalCache, obj, θ)
    hit = cache.use_cache && cache.theta_set && exact_same(cache.theta, θ)
    solved = false
    t0 = time()
    if hit
        cache.n_cache_hit += 1
    else
        cache.n_cache_miss += 1
        cache.n_unique_theta += 1
        objSol, x, nStatus = inner_loop_internal(obj, θ)
        cache.theta .= θ
        cache.theta_set = true
        cache.x = copy(x)
        cache.objSol = objSol
        cache.nStatus = nStatus
        cache.grad_jac_set = false   # any new inner solve invalidates the cached gradient
        cache.n_inner_solve += 1
        solved = true
    end
    cache.t_inner += time() - t0
    return cache.objSol, cache.x, cache.nStatus, solved, hit
end

# Ensure gradient/Jacobian is available for θ (assumes ensure_inner! already ran at this θ).
function ensure_grad!(cache::OuterEvalCache, obj, θ; need_constr::Bool=false)
    computed = false
    t0 = time()
    if !(cache.use_cache && cache.grad_jac_set)
        obj(cache.x, cache.grad, θ, constr = cache.constr, jac = cache.jac)
        cache.grad_jac_set = true
        cache.n_grad_compute += 1
        computed = true
    end
    cache.t_grad += time() - t0
    return cache.grad, cache.constr, cache.jac, computed
end

function log_row!(cache::OuterEvalCache, kind, θ, inner_solved, grad_computed, elapsed, nstatus)
    same_prev = cache.prev_theta !== nothing && exact_same(cache.prev_theta, θ)
    cache.n_callback += 1
    push!(cache.trace, CallbackTraceRow(cache.n_callback, kind, theta_hash(θ), same_prev,
        cache.use_cache && exact_same(cache.theta, θ) && !inner_solved, inner_solved, grad_computed,
        elapsed, nstatus))
    cache.prev_theta = copy(θ)
end

mutable struct CachedProblem{O}
    obj::O
    cache::OuterEvalCache
end

# ---- KNITRO eval callbacks, instrumented + optionally cached ----

function cb_F!(kc, cb, evalRequest, evalResult, userParams)
    t0 = time()
    P = userParams
    θ = evalRequest.x
    objSol, x, nStatus, solved, hit = ensure_inner!(P.cache, P.obj, θ)
    evalResult.obj[1] = -objSol
    P.obj(x, constr = evalResult.c)
    if abs(objSol) == 1e10
        evalResult.c .= 1e9
    end
    log_row!(P.cache, "F", θ, solved, false, time() - t0, nStatus)
    return 0
end

function cb_G!(kc, cb, evalRequest, evalResult, userParams)
    t0 = time()
    P = userParams
    θ = evalRequest.x
    objSol, x, nStatus, solved, hit = ensure_inner!(P.cache, P.obj, θ)
    grad, constr, jac, computed = ensure_grad!(P.cache, P.obj, θ)
    evalResult.objGrad .= -1.0 .* grad
    evalResult.jac .= jac
    log_row!(P.cache, "G", θ, solved, computed, time() - t0, nStatus)
    return 0
end

function cb_FG!(kc, cb, evalRequest, evalResult, userParams)
    t0 = time()
    P = userParams
    θ = evalRequest.x
    objSol, x, nStatus, solved, hit = ensure_inner!(P.cache, P.obj, θ)
    evalResult.obj[1] = -objSol
    grad, constr, jac, computed = ensure_grad!(P.cache, P.obj, θ)
    evalResult.objGrad .= -1.0 .* grad
    evalResult.c .= constr
    evalResult.jac .= jac
    if abs(objSol) == 1e10
        evalResult.c .= 1e9
    end
    log_row!(P.cache, "FG", θ, solved, computed, time() - t0, nStatus)
    return 0
end

"""
    outer_loop_instrumented(obj, θ_lb, θ_ub, θ_init; use_cache=true)

Parallel implementation of `cc_algo/outer_loop_functions.jl::outer_loop`, using
the SAME `obj.outer_loop_opt` file (so eval_fcga/hessopt/algorithm/maxit are
whatever the caller configured — this must be run with production's actual
`ek_outer_loop_options.opt` to answer the caching audit's real question) but
routed through the instrumented (and optionally cached) callbacks above.
Returns (κ_min, θ_min, nStatus, cache).
"""
function outer_loop_instrumented(obj, θ_lb, θ_ub, θ_init; use_cache::Bool=true)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, length(θ_init))
    KNITRO.KN_set_var_lobnds_all(kc, θ_lb)
    KNITRO.KN_set_var_upbnds_all(kc, θ_ub)
    KNITRO.KN_set_var_primal_init_values_all(kc, θ_init)

    cIndices = outer_loop_constraints!(kc, obj)
    ncon = length(cIndices)
    cache = OuterEvalCache(obj.l, ncon; use_cache = use_cache)
    P = CachedProblem(obj, cache)

    if KNITRO.KN_get_int_param(kc, "eval_fcga") == 1
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_FG!)
    else
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
        KNITRO.KN_set_cb_grad(kc, cb, cb_G!,
            jacIndexCons = repeat(cIndices, inner = length(xIndices)),
            jacIndexVars = repeat(xIndices, outer = length(cIndices)))
    end
    KNITRO.KN_set_cb_user_params(kc, cb, P)

    INNER_SOLVE_COUNT[] = 0; INNER_INFEAS_COUNT[] = 0; INNER_ITERS_TOTAL[] = 0
    _t = time()
    KNITRO.KN_solve(kc)
    wall = time() - _t
    nStatus, κ_min, θ_min, lambda_ = KNITRO.KN_get_solution(kc)
    if !obj.find_smallest
        κ_min *= -1.0
    end
    opt_err = _kn_opt_err(kc)
    outer_iters = _kn_num_iters(kc)
    outer_fc = _kn_num_fc(kc)
    KNITRO.KN_free(kc)

    return (κ_min = κ_min, θ_min = θ_min, nStatus = nStatus, cache = cache, wall = wall,
            opt_err = opt_err, outer_iters = outer_iters, outer_fc = outer_fc,
            inner_solve_count_global = INNER_SOLVE_COUNT[])
end

function write_trace_csv(path, cache::OuterEvalCache)
    open(path, "w") do io
        println(io, "idx,kind,theta_hash,same_as_prev_call,cache_hit,inner_solved,grad_computed,elapsed,nstatus")
        for r in cache.trace
            println(io, join((r.idx, r.kind, r.theta_hash, r.same_as_prev_call, r.cache_hit,
                               r.inner_solved, r.grad_computed, r.elapsed, r.nstatus), ","))
        end
    end
end

function summarize(cache::OuterEvalCache; label = "")
    n_unique = length(Set(r.theta_hash for r in cache.trace))
    println("---- ", label, " ----")
    println("total callback invocations : ", cache.n_callback)
    println("unique theta points (hash) : ", n_unique)
    println("inner solves               : ", cache.n_inner_solve)
    println("gradient computations      : ", cache.n_grad_compute)
    println("cache hits                 : ", cache.n_cache_hit)
    println("cache misses               : ", cache.n_cache_miss)
    println("inner-solve time (s)       : ", round(cache.t_inner, digits = 3))
    println("gradient time (s)          : ", round(cache.t_grad, digits = 3))
    println("inner solves / unique theta: ", round(cache.n_inner_solve / max(n_unique,1), digits = 3))
    println("callbacks / unique theta   : ", round(cache.n_callback / max(n_unique,1), digits = 3))
    flush(stdout)
end
