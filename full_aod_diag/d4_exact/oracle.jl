# ============================================================================
# Task §7: deterministic exact full-A value oracle.
#
# Wraps the EXISTING, already-validated machinery (inner_loop_internal for
# PsiObjectiveBundleImplicit, FreeParamMap, gravity_tariff.jl, winners.jl from
# this same directory) -- does not reimplement the inner CC solve or the
# moment kernel. Adds: primal-divergence recovery (phi, the Legendre dual of
# Psi -- not implemented elsewhere in this codebase for the full-A path),
# an explicit KKT-residual check for the inner dual's own first-order
# conditions (mean(m)=1, mean(m.*G_j)=0), and an exact-point cache keyed on
# more than just x_free (task's explicit requirement: no tolerance-bucket
# cache; key must include every option that changes the MATHEMATICAL value).
# ============================================================================
using SpecialFunctions: gamma as spgamma

"""
    phi(m)

Primal divergence generator (Legendre dual of Psi!/cc_algo/Psi.jl), per
task brief §3: `phi(m) = m*log(m) - m + 1` for `0 < m <= e`,
`m^2/(2e) - e/2 + 1` for `m > e`. Matches `Psi!`'s two-piece exp/quadratic
split at the a=1 <-> m=e conjugate point exactly (Legendre duality of
`Psi(a)=e^a-1` <-> `phi(m)=m*log(m)-m+1` on `a<=1 <=> 0<m<=e`, and
`Psi(a)=(e/2)(a^2+1)-1` <-> `phi(m)=m^2/(2e)-e/2+1` on `a>1 <=> m>e`).
"""
function phi(m::Real)
    if m <= 0
        return m == 0 ? 1.0 : Inf   # phi(0)=1 by continuity (m*log(m)->0); phi undefined/+inf for m<0
    elseif m <= ℯ
        return m * log(m) - m + 1
    else
        return m^2 / (2ℯ) - ℯ / 2 + 1
    end
end

"primal_divergence(m_weights::Vector) -> (1/W) sum_s phi(W * p(s)), p(s)=m(s)/sum(m). Task §3's div(F)."
function primal_divergence(m_weights::AbstractVector)
    W = length(m_weights)
    total = sum(m_weights)
    return sum(phi(W * mi / total) for mi in m_weights) / W
end

"""
    FullAEvalKey

Explicit cache key -- deliberately includes every option this investigation
varies that changes the MATHEMATICAL value (not just x_free), per task §7.3-5:
x_free (exact), delta, find_smallest, inner_loop_opt path (encodes inner
tolerance/algorithm), and a `mode` tag (:hard is the only mode implemented so
far -- moments!/hFunction! always uses the hard MinInd! branch in this
codebase; a :smooth mode using smoothMinIndNew! is NOT implemented, see
docs/fullA_d4_code_audit.md §6 and task §12E-F -- requesting it errors rather
than silently falling back to :hard). W and the draw seed are NOT part of the
key because they are baked into `ctx.U` at construction (one ctx = one fixed
draw set); a genuinely different W/seed requires a different `ctx`, which
naturally produces a different cache (see `oracle_cache_for`).
"""
struct FullAEvalKey
    x_free::Vector{Float64}
    δ::Float64
    find_smallest::Bool
    inner_loop_opt::String
    mode::Symbol
end
Base.:(==)(a::FullAEvalKey, b::FullAEvalKey) = a.x_free == b.x_free && a.δ == b.δ &&
    a.find_smallest == b.find_smallest && a.inner_loop_opt == b.inner_loop_opt && a.mode == b.mode
Base.hash(k::FullAEvalKey, h::UInt) = hash((k.x_free, k.δ, k.find_smallest, k.inner_loop_opt, k.mode), h)

"""
    SafeExactCache

Lock-guarded exact-point cache: production's answer to a real SIGABRT found
under 20-thread concurrent access on the raw `Dict{FullAEvalKey,NamedTuple}`
this file used to hand out directly (GC corruption, reproduced synthetically
and on the real `evaluate_fullA_screened` call in the
`diag/fullA-d20-warmstart-replay` investigation this was ported from). The
lock wraps ONLY the O(1) dict get/store -- never the multi-second KNITRO
solve -- so concurrent misses on different keys are not serialized.

Every `cache::Union{Nothing,Dict,SafeExactCache}` call site in this file,
`infeasibility_screen.jl`, and `fast_range_screen.jl` goes through
`_cache_lookup`/`_cache_store!` below, which dispatch correctly for either a
raw `Dict` (legacy, unlocked, still supported for any external caller that
constructs its own) or a `SafeExactCache` (production's own, from
`oracle_cache_for`).
"""
struct SafeExactCache
    d::Dict{FullAEvalKey, NamedTuple}
    lock::ReentrantLock
end
SafeExactCache() = SafeExactCache(Dict{FullAEvalKey, NamedTuple}(), ReentrantLock())
Base.length(c::SafeExactCache) = lock(() -> length(c.d), c.lock)

_cache_lookup(cache::Nothing, key) = nothing
_cache_lookup(cache::Dict, key) = get(cache, key, nothing)
_cache_lookup(cache::SafeExactCache, key) = lock(() -> get(cache.d, key, nothing), cache.lock)

_cache_store!(cache::Nothing, key, result) = nothing
_cache_store!(cache::Dict, key, result) = (cache[key] = result; nothing)
_cache_store!(cache::SafeExactCache, key, result) = (lock(() -> (cache.d[key] = result), cache.lock); nothing)

"""
    is_cacheable_result(result) -> Bool

An exact-point cache must only ever store a genuine feasible solve
(`inner_status in (0,-100,-101,-103)`) or an exact screen certificate (every
screen sentinel in `infeasibility_screen.jl`/`fast_range_screen.jl` is
`inner_status <= -9000`, see `infeasible_result`/`infeasible_result_ranged`).
A bare unresolved numerical failure (KNITRO `-300`/unbounded-dual, or any
other non-solved, non-certified status) is NOT a certificate of anything and
must never be cached as if it were one -- caching it would silently turn a
transient/point-dependent solver failure into a permanent (and potentially
wrong, since a nearby retry or different warm start might succeed) answer
for that exact point for the rest of the run.
"""
is_cacheable_result(result)::Bool = let s = get(result, :inner_status, -300)
    s in (0, -100, -101, -103) || s <= -9000
end

"Fresh, empty exact-point cache for one (method, bound-direction, draw-set) scope -- never share across those, per task §7.4. Lock-guarded (SafeExactCache), safe under KNITRO-callback-driven concurrent access."
oracle_cache_for(ctx) = SafeExactCache()

"""
    evaluate_fullA(x_free, ctx; cache=nothing, mode=:hard, tag="") -> NamedTuple

Deterministic exact full-A value oracle (task §7). `ctx` from
`d4_exact_setup()`. Returns a NamedTuple with (at minimum) every field the
task brief §7 lists. Repeated calls at the SAME x_free with `use_cache=true`
return a cache hit with `elapsed.total≈0` and byte-identical other fields;
with `use_cache=false` every call re-solves from scratch and the task's
determinism requirement (§7.2) says the two should still agree to solver
tolerance (checked by `test_oracle.jl`, not assumed here).
"""
function evaluate_fullA(x_free::AbstractVector{Float64}, ctx;
        cache = nothing, use_cache::Bool = true,
        mode::Symbol = :hard, warm::Bool = true, tag::String = "")

    mode == :hard || error("evaluate_fullA: mode=:$mode not implemented -- only :hard (hFunction!'s MinInd! branch, this codebase's only wired path) exists. See docs/fullA_d4_code_audit.md sec 6.")

    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode)
    if cache !== nothing && use_cache
        hit = _cache_lookup(cache, key)
        if hit !== nothing
            return merge(hit, (cache_hit = true, tag = tag))
        end
    end

    t_total0 = time()
    if !warm
        obj.x .= NaN   # force cold start: inner_loop_initial_values falls back to zeros(outer_constr_index)
    end

    θ_full = CS.reconstruct_full(x_free, ctx.m)

    t_inner0 = time()
    K_hard, inner_x, nStatus = CS.inner_loop_internal(obj, θ_full)
    t_inner = time() - t_inner0

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]   # cumulative counter; caller can diff across calls for a per-call count
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, ctx.D),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, moment_resid = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
        return result
    end

    # ---- fill K/G at this theta (byte-identical to what inner_loop_internal just used internally) ----
    W = size(obj.U, 1); d = obj.d
    K = zeros(W); G = zeros(W, d)
    obj.moments!(K, G, θ_full, obj.U, obj)

    # ---- Delta_dual + gravity/divergence constraint values (mirrors outer_loop_cached.jl's own
    #      compute_constr_values pattern, reused not re-derived) ----
    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj(inner_x, constr = @view(cbuf[1:ncon]))
    # SIGN CONVENTION (verified from code, not assumed -- see task brief sec 3/9 "verify the exact
    # sign"): f = mean(Psi(arg0)) + zeta is task-brief's L(x,y*) exactly, but PsiObjectiveBundle.jl
    # sets constr[1] = -f*1e10 and outer_loop_cached.jl enforces constr[1] <= 1e10*delta, i.e.
    # -f <= delta, i.e. the quantity KNITRO (and this whole codebase) calls "Delta(theta)" is -f,
    # not f. Using fval directly here FAILED the primal-dual-gap check below by an exact sign flip
    # (Delta_dual=-0.0010030 vs Delta_primal=+0.0010030) before this fix -- caught, not assumed.
    Delta_dual = cbuf[1] / 1e10              # == -fval; equals task-brief's Delta(x)=min_y L(x,y)
    # obj.arg1 now holds dPsi(arg0) = m(s) (un-normalized LFD weight), by construction of the
    # callable above (dPsi! is invoked whenever constr is requested) -- see PsiObjectiveBundle.jl:186
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = primal_divergence(m_weights)

    # ---- inner KKT residual: the dual problem's OWN first-order conditions, computed directly
    #      (not read off a KNITRO-reported number) -- task §8's exact requirement ----
    mean_m_resid = abs(sum(m_weights) / W - 1.0)   # should be ~0 at a converged solution: mean(m)=1
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    moment_kkt = [abs(sum(m_weights .* G[:, j]) / W) for j in 1:min(length(λstar), size(G,2))]
    max_abs_moment_kkt_resid = isempty(moment_kkt) ? NaN : maximum(moment_kkt)

    # ---- gravity residual (raw obj column + gravity_tariff.jl's target-scaled value + the
    #      task-brief R_sum/R_mean/R_beta family, computed on the SAME q_tilde/N_obs machinery
    #      validated in gravity_tariff.jl / test_free_param_and_gravity.jl) ----
    gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN   # obj's own (possibly rescaled-by-caller) column; here UNSCALED raw value from a fresh obj(...) call
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
    gravity_val = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)   # -(1/N_obs) sum q_tilde*log(A_od); see gravity_tariff.jl
    logA = -log.(AodPow)   # log(A_od) = -log(AodPow), per gravity_tariff.jl's module docstring
    # R_sum = sum(q_tilde .* logA_tilde) where logA_tilde is the two-way-demeaned logA; by the FWL
    # identity gravity_tariff.jl's own docstring proves (q_tilde already one-sided-residualized),
    # this equals sum(q_tilde .* logA) exactly -- reused here rather than re-demeaning logA.
    R_sum = sum(ctx.q_tilde .* logA)
    R_mean = R_sum / ctx.D^2
    R_beta = R_sum / sum(ctx.q_tilde .^ 2)

    moment_resid = vec(sum(G, dims=1)) ./ W
    max_abs_moment_resid = isempty(moment_resid) ? NaN : maximum(abs.(moment_resid))

    winner, price_, gap_ = compute_winners(θ_full, ctx)
    winner_hash = hash(winner)

    t_total = time() - t_total0
    elapsed = (total = t_total, inner = t_inner, post = t_total - t_inner)

    result = (x_free = collect(x_free), θ_full = θ_full,
              gamma_focal_prime = θ_full[3+ctx.D], logA = logA,
              K_hard = K_hard, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              Delta_minus_delta = Delta_dual - obj.δ,
              gravity_raw = gravity_raw, gravity_value = gravity_val,
              gravity_R_sum = R_sum, gravity_R_mean = R_mean, gravity_R_beta = R_beta,
              moment_resid = moment_resid, max_abs_moment_resid = max_abs_moment_resid,
              zeta = ζstar, lambda = collect(λstar),
              m_mean = sum(m_weights)/W, m_min = minimum(m_weights), m_max = maximum(m_weights),
              weight_norm_resid = abs(sum(p_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              winner_hash = winner_hash, inner_status = nStatus, inner_iters = inner_iters,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              cache_hit = false, warm_started = warm, tag = tag,
              elapsed = elapsed, error_reason = nothing)

    cache !== nothing && is_cacheable_result(result) && _cache_store!(cache, key, result)
    return result
end
