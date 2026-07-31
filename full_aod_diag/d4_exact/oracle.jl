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
using SHA

"""
    reject_point(x, msg) -> never returns

Remediation task Part E (finding F6): named helper making explicit a hidden type contract every
KNITRO outer-loop callback (`cb_F!`/`cb_G!` in c10_d20_production_driver.jl, cm_checkpoint.jl,
cm_outer_driver.jl) relies on: KNITRO.jl v1.2.1's `_try_catch_handler`
(`~/.julia/packages/KNITRO/*/src/C_wrapper.jl`) maps a caught `DomainError` specifically to
`KN_RC_EVAL_ERR` (a graceful "reject this point, try another" backtrack) and maps ANY OTHER
exception type to `KN_RC_CALLBACK_ERR`, which ABORTS the outer solve. A future refactor that
replaces `DomainError` with `ErrorException` (or lets a different exception type -- e.g. a leaked
`TiedWinnerError` -- escape a callback) silently changes "reject this point" into "abort the
run," with no compile-time or type-system signal that anything changed. Use this helper at every
callback reject site instead of a bare `throw(DomainError(...))`, so the contract is named once
and grep-able, not re-derived at each of the ~6 call sites.
"""
reject_point(x, msg::AbstractString) = throw(DomainError(x, msg))

"""
    sha256_of_matrix(M::AbstractMatrix{Float64}) -> String

AUD-11 fix: a stable, cross-process/cross-Julia-version content digest. Julia's built-in
`hash()` is explicitly NOT a content digest -- the Julia manual documents that `hash` values are
only guaranteed stable within one Julia process/version, not across processes or versions, which
is exactly what draw/checkpoint reproducibility (draw_design.jl) and context-fingerprinting
(`context_fingerprint` below, AUD-08) both need to detect. Hashes canonical little-endian Float64
bytes PLUS the matrix's own shape (so two same-byte-count but differently-shaped matrices cannot
collide), independent of host endianness or Julia version.

Lives here (not in draw_design.jl, where the AUD-11 fix first introduced it) because
`context_fingerprint`'s dependency on it is UNCONDITIONAL and oracle.jl is the more universally-
included file across this codebase -- a real UndefVarError was caught live in
test_cross_delta_cache.jl (includes oracle.jl but not draw_design.jl) before this move.
draw_design.jl now guards its own use of this function with an include-if-needed fallback.
"""
function sha256_of_matrix(M::AbstractMatrix{Float64})::String
    Md = Matrix{Float64}(M)   # materialize (handles views/reshapes/Adjoint), canonical column-major order
    buf = IOBuffer()
    write(buf, htol(Int64(size(Md, 1))))
    write(buf, htol(Int64(size(Md, 2))))
    @inbounds for x in Md
        write(buf, htol(reinterpret(UInt64, x)))
    end
    return bytes2hex(SHA.sha256(take!(buf)))
end

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

const CONTEXT_FINGERPRINT_SCHEMA = 4   # bumped 2026-07-31 (Brazil-Korea gravity-exclusion task):
                                        # digest now also includes exclude_diagonal_gravity and
                                        # gravity_exclude_cells -- CLOSES A PRE-EXISTING GAP: two
                                        # contexts sharing the same τ/L/wHat but differing ONLY in
                                        # exclude_diagonal_gravity (hence different ctx.q_tilde,
                                        # different pivot, different gravity residual) previously
                                        # hashed IDENTICALLY, since only τ itself (not the mask
                                        # applied to it) was ever hashed. Schema 3 (2026-07-30,
                                        # sigma3 campaign prep) added the sigma segment; schema 2
                                        # (2026-07-23) added destination_sample/row_idx/D_dest.

# AUD-08 fix: memoized per-ctx SHA-256 fingerprint, keyed by objectid(ctx.U) (uniquely identifies
# one ctx's draw set/instance -- cheap identity lookup, avoids re-hashing large W x D draw
# matrices on every single evaluation call, which would otherwise be prohibitively expensive at
# W=80,000).
const _CTX_FINGERPRINT_CACHE = IdDict{Any,String}()
const _CTX_FINGERPRINT_LOCK = ReentrantLock()

"""
    context_fingerprint(ctx) -> String

AUD-08 fix: a versioned SHA-256 digest of everything that changes the MATHEMATICAL answer of an
inner solve at a fixed exact theta -- draws (via ctx.draw_meta's own checksums when present,
falling back to hashing ctx.U/ctx.γ.Uσ directly for older/test ctx shapes that predate
draw_meta), draw design/seed, shapes (D, W), fixed trade/model data (wHat, L, LPrime, τ, τPrime),
CM config (if any), option-file CONTENTS (not just the path string, which the old FullAEvalKey
already included -- two paths with the same name but different contents must not alias), and the
actually-loaded KNITRO release. Explicitly does NOT include δ or find_smallest -- neither
changes the inner CC dual problem at a fixed theta (see FullAEvalKey's own docstring on why δ
does not belong in the key either); direction/delta remain separate outer-run metadata (AUD-08).
"""
function context_fingerprint(ctx)::String
    lock(_CTX_FINGERPRINT_LOCK) do
        get!(_CTX_FINGERPRINT_CACHE, ctx.U) do
            buf = IOBuffer()
            write(buf, htol(Int64(CONTEXT_FINGERPRINT_SCHEMA)))
            write(buf, htol(Int64(ctx.D)))
            write(buf, htol(Int64(size(ctx.U, 1))))   # W
            # Part A (2026-07-23, omit-ROW-destination): destination-sample identity, so a full-D
            # and a ROW-excluded context can never be treated as fingerprint-equivalent even if
            # they happened to share ctx.U by construction. row_idx/destination_sample absent
            # (nothing/:all_legacy) reproduces the pre-Part-A digest exactly on this new segment
            # (D_dest==D, "all_legacy" tag) -- old digests from before this field existed are still
            # distinguished from it by CONTEXT_FINGERPRINT_SCHEMA below if that was bumped, but this
            # segment's own content is a no-op addition for every legacy (row_idx===nothing) ctx.
            write(buf, htol(Int64(hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D)))
            row_idx_here = hasproperty(ctx, :row_idx) ? ctx.row_idx : nothing
            write(buf, row_idx_here === nothing ? "all_legacy" : "exclude_row_$(row_idx_here)")
            write(buf, row_idx_here === nothing ? "square_v1" : "rectangular_D_x_Dminus1_true_shrink_v1")
            # sigma (2026-07-30, sigma3 campaign prep): CES elasticity of substitution enters the
            # inner solve's own price-index/CES formulas directly, not merely as one entry of the
            # outer theta vector that a fixed x_free would otherwise pin down -- two contexts built
            # identically except for sigma can give DIFFERENT inner-solve answers at the same
            # x_free. Previously unhashed here, so a sigma=2.5 and a sigma=3.0 context could alias
            # to the same cache key/directory (exactly the risk the sigma3 campaign brief's "do not
            # reuse caches built under sigma=2.5" requirement warns about). ctx.σ is present on
            # every context builder in this file family (context.jl/context_scaled.jl/
            # context_real_d20.jl/qmc_context_real_d20.jl all return σ=σ).
            write(buf, htol(Float64(ctx.σ)))
            # gravity-mask segment (2026-07-31, Brazil-Korea gravity-exclusion task, schema 4):
            # exclude_diagonal_gravity and gravity_exclude_cells both change ctx.q_tilde/N_obs
            # (hence the pivot cell, the pivot residual, and theta_star itself) without touching
            # τ/L/wHat -- must be hashed explicitly, not left to alias via the data hash below.
            write(buf, hasproperty(ctx, :exclude_diagonal_gravity) && ctx.exclude_diagonal_gravity ? "diag_excl" : "diag_incl")
            excl_cells = hasproperty(ctx, :gravity_exclude_cells) ? ctx.gravity_exclude_cells : Tuple{Int,Int}[]
            write(buf, htol(Int64(length(excl_cells))))
            for (o, d) in sort(collect(excl_cells))
                write(buf, htol(Int64(o))); write(buf, htol(Int64(d)))
            end
            if hasproperty(ctx, :draw_meta)
                write(buf, ctx.draw_meta.checksum_uniform)
                write(buf, ctx.draw_meta.checksum_transformed)
                write(buf, string(ctx.draw_meta.draw_design))
                write(buf, htol(Int64(ctx.draw_meta.draw_seed)))
            else
                write(buf, sha256_of_matrix(ctx.U))
                hasproperty(ctx.γ, :Uσ) && write(buf, sha256_of_matrix(ctx.γ.Uσ))
            end
            for fld in (:wHat, :L, :LPrime, :τ, :τPrime)
                if hasproperty(ctx.γ, fld)
                    v = getproperty(ctx.γ, fld)
                    M = v isa AbstractMatrix ? Matrix{Float64}(v) : reshape(Vector{Float64}(v), :, 1)
                    write(buf, sha256_of_matrix(M))
                end
            end
            write(buf, hasproperty(ctx, :cm) && ctx.cm !== nothing ? string(ctx.cm) : "no_cm")
            write(buf, isdefined(Main, :LOADED_KNITRO_RELEASE) ? Main.LOADED_KNITRO_RELEASE : "unknown_knitro_release")
            opt_paths = (hasproperty(ctx.obj, :inner_loop_opt) ? ctx.obj.inner_loop_opt : nothing,
                         hasproperty(ctx.obj, :outer_loop_opt) ? ctx.obj.outer_loop_opt : nothing)
            for optpath in unique(filter(p -> p isa AbstractString && isfile(p), opt_paths))
                write(buf, read(optpath))
            end
            bytes2hex(SHA.sha256(take!(buf)))
        end
    end
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

AUD-08 fix: also carries `ctx_fingerprint` (see `context_fingerprint` above) so a cache accidentally
shared across two DIFFERENT contexts that happen to agree on x_free/delta/find_smallest/option-path/
mode (e.g. differing only in their draw set, CM config, or fixed trade data) can no longer alias --
the old key omitted any context-identifying information at all, a confirmed latent cross-context
aliasing risk (AUD-08).
"""
struct FullAEvalKey
    x_free::Vector{Float64}
    δ::Float64
    find_smallest::Bool
    inner_loop_opt::String
    mode::Symbol
    ctx_fingerprint::String
end
Base.:(==)(a::FullAEvalKey, b::FullAEvalKey) = a.x_free == b.x_free && a.δ == b.δ &&
    a.find_smallest == b.find_smallest && a.inner_loop_opt == b.inner_loop_opt && a.mode == b.mode &&
    a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::FullAEvalKey, h::UInt) = hash((k.x_free, k.δ, k.find_smallest, k.inner_loop_opt, k.mode, k.ctx_fingerprint), h)

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

Parametric over the key type `K` (was hardcoded to `FullAEvalKey`) so the same
lock-guarded mechanism serves the CM production path's `CMEvalKey`
(`cm_config.jl`) without a second cache implementation -- `SafeExactCache()`
still defaults to `K=FullAEvalKey` for every existing call site.
"""
struct SafeExactCache{K}
    d::Dict{K, NamedTuple}
    lock::ReentrantLock
end
SafeExactCache{K}() where {K} = SafeExactCache{K}(Dict{K, NamedTuple}(), ReentrantLock())
SafeExactCache() = SafeExactCache{FullAEvalKey}()
Base.length(c::SafeExactCache) = lock(() -> length(c.d), c.lock)

_cache_lookup(cache::Nothing, key) = nothing
_cache_lookup(cache::Dict, key) = get(cache, key, nothing)
_cache_lookup(cache::SafeExactCache, key) = lock(() -> get(cache.d, key, nothing), cache.lock)

_cache_store!(cache::Nothing, key, result) = nothing
_cache_store!(cache::Dict, key, result) = (cache[key] = result; nothing)
_cache_store!(cache::SafeExactCache, key, result) = (lock(() -> (cache.d[key] = result), cache.lock); nothing)

"""
    VerifiedSuccessTolerances

AUD-04 fix. Provisional default tolerances for `classify_inner_result`'s independent
residual/gap gate -- KNITRO's own statuses 0/-100/-101/-103 are NOT themselves an optimality
certificate (KNITRO documents -101/-103 as tolerance-based approximate stops), so a result must
also pass these checks before it is treated as scientifically verified.

NOT empirically calibrated against real D=20/W=80,000 production solves in this remediation pass
-- `test_oracle.jl`'s own D=4 assertion (`primal_dual_gap < 1e-6`) is the only existing empirical
anchor in this codebase, and D=4/small-W convergence is characteristically tighter than
D=20/W=80,000 will achieve. `primal_dual_gap_tol`/`max_abs_moment_kkt_resid_tol` are deliberately
looser than the D=4 anchor for that reason; `weight_norm_resid_tol`/`mean_m_resid_tol` are exact
algebraic KKT identities (mean(m)=1) that should hold near machine precision at ANY converged
point regardless of D/W, so they are left tight. Treat all four as a starting point to be
tightened/loosened once the matched cache-disabled cold D=20 runs (task §15) establish what a
genuinely well-converged large-W solve's residuals actually look like -- not yet a validated
scientific standard. See docs/fullA_independent_audit_remediation.md AUD-04.
"""
Base.@kwdef struct VerifiedSuccessTolerances
    primal_dual_gap_tol::Float64 = 1e-3
    mean_m_resid_tol::Float64 = 1e-6
    max_abs_moment_kkt_resid_tol::Float64 = 1e-3
    m_min_floor::Float64 = 0.0   # conjugate-domain guard: phi/Psi! require m>0
end
# Remediation task Part E: removed weight_norm_resid_tol -- `weight_norm_resid =
# abs(sum(m ./ sum(m)) - 1)` is ~1e-16 by floating-point construction (sum(p) for a normalized
# p=m/sum(m) is a tautology, not an independent check) and gates nothing. `mean_m_resid =
# abs(mean(m)-1)` is the real normalization check (an actual KKT identity at a converged
# solution, not true by construction) and remains gated. `weight_norm_resid` itself is still
# computed and returned by archC_verified_state/evaluate_fullA for diagnostic visibility; it is
# just no longer part of the acceptance gate below.
const DEFAULT_VERIFIED_SUCCESS_TOL = VerifiedSuccessTolerances()

"""
    InnerResultClass

AUD-04 typed inner-solve outcome (replaces a bare status-code membership check). Coordinates
with, and does not erase, the separate negative-cache audit's typed distinction
(`negative_cache.jl`'s `SolvedInnerResult`/`CertifiedInfeasibleResult`/`ConfirmedNegativeResult`):
this enum classifies the POSITIVE side (was the status a real, residual-verified solve?) that the
negative-cache module's confirmation policy assumed but never itself checked.
"""
@enum InnerResultClass VerifiedSolved ApproximateSolved ExactInfeasible ConfirmedNumericalNegative TransientFailure

"""
    classify_inner_result(result; tol=DEFAULT_VERIFIED_SUCCESS_TOL) -> InnerResultClass

  - `VerifiedSolved` -- status feasible AND every independent check (finite Delta_dual, primal
    normalization, weighted moment/KKT residual, primal-dual gap, conjugate-domain m>0) passes
    `tol`. Only these may enter the exact-result cache or become the final reported incumbent.
  - `ApproximateSolved` -- status feasible but at least one residual/gap check failed `tol` (or is
    non-finite). May seed the successful-dual bank, trigger a tighter retry, or be logged as
    provisional -- never cached as exact, never the final incumbent.
  - `ExactInfeasible` -- `inner_status <= -9000`, a screen certificate (never touched KNITRO).
  - `ConfirmedNumericalNegative` -- status not feasible and not a screen certificate. This
    classifier cannot itself distinguish confirmed-vs-transient (that needs the second-start
    confirmation `negative_cache.jl`'s Policy B already implements); callers doing that
    confirmation should treat this return value as "not solved," not as a final disposition.
"""
function classify_inner_result(result; tol::VerifiedSuccessTolerances = DEFAULT_VERIFIED_SUCCESS_TOL)::InnerResultClass
    s = get(result, :inner_status, -300)
    s <= -9000 && return ExactInfeasible
    s in (0, -100, -101, -103) || return ConfirmedNumericalNegative

    Δ = get(result, :Delta_dual, NaN)
    gap = get(result, :primal_dual_gap, NaN)
    mmr = get(result, :mean_m_resid, NaN)
    kkt = get(result, :max_abs_moment_kkt_resid, NaN)
    mmin = get(result, :m_min, NaN)

    ok = isfinite(Δ) && isfinite(gap) && isfinite(mmr) && isfinite(kkt) &&
         isfinite(mmin) && mmin > tol.m_min_floor &&
         gap <= tol.primal_dual_gap_tol &&
         mmr <= tol.mean_m_resid_tol && kkt <= tol.max_abs_moment_kkt_resid_tol

    return ok ? VerifiedSolved : ApproximateSolved
end

"Convenience predicate: classify_inner_result(result) == VerifiedSolved."
is_verified_success(result; tol::VerifiedSuccessTolerances = DEFAULT_VERIFIED_SUCCESS_TOL) =
    classify_inner_result(result; tol = tol) == VerifiedSolved

"""
    is_cacheable_result(result) -> Bool

AUD-04 fix: an exact-point cache must only ever store a scientifically VERIFIED solve
(`classify_inner_result(result) == VerifiedSolved` -- status feasible AND residual/gap tolerances
pass, not status alone) or an exact screen certificate (`ExactInfeasible`). An `ApproximateSolved`
result (feasible status but a failed residual/gap check) or an unresolved numerical failure
(`ConfirmedNumericalNegative`) is NOT a certificate of anything and must never be cached as if it
were one -- caching it would silently turn a transient/tolerance-marginal solver outcome into a
permanent (and potentially wrong) answer for that exact point for the rest of the run.
"""
is_cacheable_result(result)::Bool = classify_inner_result(result) in (VerifiedSolved, ExactInfeasible)

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

    # Part A (2026-07-23): D_dest (destination count) defaults to ctx.D for any context builder
    # that predates the omit-ROW-destination release and so has no D_dest field of its own --
    # legacy square contexts (D_dest==D) are unaffected by this fallback.
    D_dest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode, context_fingerprint(ctx))
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
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, D_dest),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, benchmark_unweighted_moment_mean = Float64[], max_abs_moment_resid = NaN,
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
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*D_dest], ctx.D, D_dest)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (D_dest, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
    gravity_val = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs; exclude_diagonal=get(ctx, :exclude_diagonal_gravity, false), exclude_cells=get(ctx, :gravity_exclude_cells, Tuple{Int,Int}[]))   # -(1/N_obs) sum q_tilde*log(A_od); see gravity_tariff.jl
    logA = -log.(AodPow)   # log(A_od) = -log(AodPow), per gravity_tariff.jl's module docstring
    # R_sum = sum(q_tilde .* logA_tilde) where logA_tilde is the two-way-demeaned logA; by the FWL
    # identity gravity_tariff.jl's own docstring proves (q_tilde already one-sided-residualized),
    # this equals sum(q_tilde .* logA) exactly -- reused here rather than re-demeaning logA.
    R_sum = sum(ctx.q_tilde .* logA)
    R_mean = R_sum / (ctx.D * D_dest)
    R_beta = R_sum / sum(ctx.q_tilde .^ 2)

    benchmark_unweighted_moment_mean = vec(sum(G, dims=1)) ./ W
    max_abs_moment_resid = isempty(benchmark_unweighted_moment_mean) ? NaN : maximum(abs.(benchmark_unweighted_moment_mean))

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
              benchmark_unweighted_moment_mean = benchmark_unweighted_moment_mean, max_abs_moment_resid = max_abs_moment_resid,
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
