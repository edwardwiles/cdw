# ================================================================================================
# OUTER-loop production layer for the CM + pairwise-quantile family (family #7, 2026-08-12).
#
# `cm_pairwise_quantile_production.jl` is the INNER solve; `cm_pairwise_quantile_hessian_assembly.jl`
# is its exact Hessian. This file adds what an OUTER loop needs on top, mirroring
# `pairwise_quantile_outer_production.jl` function for function:
#
#   1. build_cm_pairwise_quantile_production_context  <- build_pairwise_quantile_production_context
#   2. archCMPQ_verified_state                        <- archPQ_verified_state
#   3. cm_pairwise_quantile_production_gradient       <- pairwise_quantile_production_gradient
#   4. d_delta_dual_d_cmpq_mass_fd                    <- d_delta_dual_d_mass_fd
#
# THE ONE STRUCTURAL DIFFERENCE FROM THE STANDALONE FAMILY, and it is the trap in this file:
# the inner dual index has THREE blocks, `[E | level+pair | CM-grid]`, not two. So the mandatory `q0`
# fold must add BOTH `-G_R*lambda_R` AND `-G_CM*lambda_CM`. Folding only the first would linearize
# the shared economic gradient about the wrong base point, and -- per memory
# `feedback-q0-restriction-fold-is-a-level-not-a-derivative` -- the tempting argument that the CM
# rows "don't depend on theta" is true about the DERIVATIVE and irrelevant to `q0`, which is a LEVEL.
# `build_lfix_base_cache_cmpq` folds both, and cross-checks the result against the verifier's own
# independently recomputed `r` as a hard error, which is what would catch it if anyone dropped one.
#
# SIGN CONVENTION: the inner solve MINIMIZES `f = mean(Psi(r)) + zeta` and the reported divergence is
# `Delta_dual = -f`. `d_delta_dual_d_mu_shared` (cm_pairwise_quantile_moments.jl) already
# differentiates `Delta_dual` -- its `-mean_m` prefactor IS that sign -- so NOTHING is negated at
# this layer. The reoptimized-FD gate carries a negative control that fails if anyone adds one.
#
# Requires: cm_pairwise_quantile_production.jl and its include chain,
# cm_pairwise_quantile_hessian_assembly.jl, cm_pairwise_quantile_verification.jl, shared_a_gradient.jl,
# operator_verification.jl, lfix_cm_aware.jl, oracle.jl.
# ================================================================================================

using LinearAlgebra: norm

isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))
isdefined(Main, :verify_namedtuple_from_operator) || include(joinpath(@__DIR__, "operator_verification.jl"))
isdefined(Main, :verify_inner_solution_operator_cmpairwisequantile!) ||
    include(joinpath(@__DIR__, "cm_pairwise_quantile_verification.jl"))
# These appear in METHOD SIGNATURES below, so they must exist at definition time, not call time.
isdefined(Main, :BaseDualState) || include(joinpath(@__DIR__, "three_way_derivatives.jl"))
isdefined(Main, :RestrictedDualBank) || include(joinpath(@__DIR__, "cm_dual_bank_production.jl"))
isdefined(Main, :CMScreenCounters) || include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
isdefined(Main, :with_q0) || include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
isdefined(Main, :build_lfix_base_workspace) || include(joinpath(@__DIR__, "lfix_base_workspace.jl"))

"""
    reshape_cmpq_duals_lambda(lambda, op, ncore1) -> (lambda_L, lambda_P, lambda_CM)

`reshape_cmpq_duals` for a vector that has ALREADY had `zeta` stripped (`base.λstar`, i.e.
`inner_x[2:end]`), rather than the full inner variable vector. One line of offset arithmetic, in one
place, rather than a `vcat(0.0, λstar)` at each of the three outer-layer call sites -- this family's
known hazard is layout drift between the solve, the verifier and the outer layer, so the offsets get
a name instead of being re-derived.
"""
function reshape_cmpq_duals_lambda(lambda::AbstractVector{Float64}, op::PairwiseQuantileOperator,
                                   ncore1::Int)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nL = n_cmpq_level_rows(L); nP = nc * nc * npair
    n_restr = n_cmpq_restr_rows(D, L)
    length(lambda) >= ncore1 + n_restr ||
        error("reshape_cmpq_duals_lambda: length(lambda)=$(length(lambda)) < ncore1+n_restr=" *
              "$(ncore1 + n_restr)")
    λ_L = @view lambda[ncore1+1 : ncore1+nL]
    λ_P = reshape(@view(lambda[ncore1+nL+1 : ncore1+nL+nP]), nc, nc, npair)
    λ_CM = @view lambda[ncore1+n_restr+1 : end]
    return (λ_L, λ_P, λ_CM)
end

"""
    build_cm_pairwise_quantile_production_context(ctx, cfg; inner_opt) -> NamedTuple

Analog of `build_pairwise_quantile_production_context`: assembles the restriction-augmented context
ONCE per run and returns a NamedTuple with a `.ctx_cm` field, which is what `prepare_production_run`'s
`_resolve_bundle` (production_bundle_api.jl) looks for -- so a driver wraps this in
`prepare_production_run(:cm_pairwise_quantile, "<runner>", () -> build_...(...))` and gets the
OperatorPsiBundle invariant asserted for free, exactly like every other family.

`ctx` MUST be the plain, unaugmented economic context. It is carried forward on
`ctx_cm.cmpq_econ_ctx` because `archCMPQ_base_state`/`_verified_state` genuinely need BOTH: `cf_build`/
`prime_operator!` read dimensionality off the ECONOMIC-only bundle, and passing the augmented context
there corrupts `cf.oci` (a real bug caught live for the standalone family on 2026-08-09).

EVERY SCIENTIFIC PARAMETER ARRIVES ON `cfg` (a `CMPairwiseQuantileConfig`, all fields required) --
`L`, `cm_grid_size`, `cm_moment_families`, `contrasts`, `min_bin_count`, `mass_start`. Nothing is
defaulted here. `inner_opt` is the KNITRO option file (tolerances, not the economic problem) and is
required at this layer because a production run must state whether it is the exact-Hessian file or
not; there is no "whichever the economic bundle had" fallback at the OUTER layer.
"""
function build_cm_pairwise_quantile_production_context(ctx, cfg::CMPairwiseQuantileConfig;
                                                       inner_opt::String)
    cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = inner_opt)
    ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx = true)
    ctx_cm = merge(ctx_cm, (cmpq_econ_ctx = ctx,))
    # `cm_pairwise_quantile_attach` builds the FG state against a ctx_cm that does not yet carry
    # `cmpq_econ_ctx`; the state closes over `obj`/`op`/`mass_state`/`core_cf_ref` only (never the
    # context object), so re-merging afterwards cannot leave it pointing at a stale context. Asserted
    # rather than argued, since a mismatch here would be silent.
    ctx_cm.cmpq_fg_state.op === cmpq.op && ctx_cm.cmpq_fg_state.mass_state === cmpq.mass_state ||
        error("build_cm_pairwise_quantile_production_context: the FG state is not bound to this " *
              "context's own operator/mass state")

    println(stdout, "cm_restriction_basis [cm_pairwise_quantile] = CM grid (G=", cmpq.G, ", ",
            cmpq.Lcm, " levels, ", cmpq.n_families, " family/families, contrasts=:", cmpq.contrasts,
            ") + FRECHET-z quantile bins (L=", cmpq.L, ") on a SHARED reference marginal; ",
            "n_restr=", cmpq.n_restr, ", ncm=", cmpq.ncm)
    println(stdout, "cm_pairwise_quantile outer mass coordinates: n_raw=", n_cmpq_raw(cmpq.L),
            " (ONE shared simplex, stick-breaking; the standalone family needs ",
            (cmpq.L - 1) * ctx.D, "); bin gates: cells checked=", cmpq.gates.bin_cells_checked,
            ", min marginal=", cmpq.gates.min_marginal_count,
            ", min joint=", cmpq.gates.min_joint_count)
    flush(stdout)

    return (ctx_cm = ctx_cm, cmpq = cmpq, hess_ctx = ctx_cm.cmpq_hess_ctx, L = cmpq.L, G = cmpq.G,
            n_families = cmpq.n_families, contrasts = cmpq.contrasts,
            min_bin_count = cfg.min_bin_count, mass_start = cfg.mass_start,
            raw_start = cmpq.raw_start, raw_bounds = cmpq.raw_bounds, gates = cmpq.gates,
            mu_frechet = ctx.μHat, inner_opt = inner_opt)
end

"""
    archCMPQ_verified_state(x_free0, raw_masses, ctx_cm; dual_bank=nothing, eval_id=0)
        -> (base, verify)

Verified analog of `archCMPQ_base_state`, mirroring `archPQ_verified_state`: solve with the EXACT
Hessian, then INDEPENDENTLY recompute the solution's residual/objective/KKT blocks with this family's
own verifier, then hand that to the shared `verify_namedtuple_from_operator` so `verify` carries
precisely the field set `classify_inner_result`/`is_verified_success` already know how to read.

The Hessian builder is passed explicitly (`cmpq_hess_builder_for`) rather than defaulted, matching
`archCMPQ_base_state`'s own required-kwarg discipline: a verified state is a production artifact and
must not be produced by a silently quasi-Newton solve.

Throws `CMExpectedSolveFailure` on a failed inner solve, exactly as every other family's
`*_verified_state` does, so a driver's `cb_F!` can `reject_point`.
"""
function archCMPQ_verified_state(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, ctx_cm;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    obj = ctx_cm.obj
    cmpq = ctx_cm.cmpq_ctx
    econ_ctx = ctx_cm.cmpq_econ_ctx
    op = cmpq.op
    length(raw_masses) == n_cmpq_raw(cmpq.L) ||
        error("archCMPQ_verified_state: length(raw_masses)=$(length(raw_masses)) != " *
              "n_cmpq_raw(L)=$(n_cmpq_raw(cmpq.L))")

    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj,
            vcat(collect(x_free0), collect(raw_masses)))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end

    nStatus, inner_x, _, n_fg, n_hess = archCMPQ_base_state(x_free0, raw_masses, econ_ctx, ctx_cm;
        hess_cb_builder = cmpq_hess_builder_for(ctx_cm))
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral &&
            (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archCMPQ_verified_state: inner solve failed, nStatus=$nStatus " *
                                     "(x_free0=$x_free0, raw_masses=$raw_masses)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = op.W
    ncore1 = cmpq.ncore1
    cf = cmpq.core_cf_ref[]
    cf isa CompressedFactual ||
        error("archCMPQ_verified_state: core_cf_ref[] is not a CompressedFactual -- prime_operator! " *
              "did not run for this outer point (got $(typeof(cf)))")
    econ_ws = economic_operator_workspace(cf)
    ov = verify_inner_solution_operator_cmpairwisequantile!(ζstar, λstar, cf, cmpq,
        ctx_cm.cmpq_mass_state, W, economic_forward!, economic_transpose!, econ_ws,
        obj.Psi!, obj.dPsi!, ncore1)
    m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    verify = merge(verify, (r_current = ov.r,
                            kkt_resid_E = ov.kkt_resid_E,
                            kkt_resid_level = ov.kkt_resid_level,
                            kkt_resid_pairindep = ov.kkt_resid_pairindep,
                            kkt_resid_cm = ov.kkt_resid_cm,
                            max_cumulative_residual = ov.max_cumulative_residual,
                            max_level_cumulative_residual = ov.max_level_cumulative_residual,
                            max_implied_cumulative_residual = ov.max_implied_cumulative_residual,
                            level_prob = ov.level_prob, origin_prob = ov.origin_prob,
                            mu = ov.mu, mu_last = ov.mu_last, Pcum = ov.Pcum,
                            n_fg = n_fg, n_hess = n_hess))

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id,
        vcat(collect(x_free0), collect(raw_masses)), inner_x)
    return base, verify
end

"`(K, base, verify)`; `K` is `obj.H_save`, the same payoff scalar every family reports."
function cm_pairwise_quantile_production_value_verified(x_free0::AbstractVector,
        raw_masses::AbstractVector{Float64}, pcx;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    base, verify = archCMPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm;
                                           dual_bank = dual_bank, eval_id = eval_id)
    return pcx.ctx_cm.obj.H_save, base, verify
end

"Screened drop-in, mirroring `pairwise_quantile_production_value_verified_screened`: the SAME
family-agnostic `cm_screen_precheck!` first, then this family's verified state."
function cm_pairwise_quantile_production_value_verified_screened(x_free0::AbstractVector,
        raw_masses::AbstractVector{Float64}, pcx;
        counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false,
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    cm_screen_precheck!(x_free0, pcx.ctx_cm; counters = counters, use_witness = use_witness)
    return cm_pairwise_quantile_production_value_verified(x_free0, raw_masses, pcx;
                                                          dual_bank = dual_bank, eval_id = eval_id)
end

"""
    ensure_cmpq_masses!(ctx_cm, raw_masses) -> nothing

Refresh `ctx_cm.cmpq_mass_state` so it corresponds to `raw_masses`, before any OUTER-loop consumer
reads it. Same hazard and same remedy as the standalone family's `ensure_pq_masses!`: the mass state
is mutable per-outer-point state that some intervening solve, verification re-solve or FD probe may
have overwritten, and a gradient computed at another point's masses would look entirely normal. Cost
is `O(L)` here -- the shared simplex is one vector, not `D` of them.
"""
function ensure_cmpq_masses!(ctx_cm, raw_masses::AbstractVector{Float64})
    set_cmpq_masses!(ctx_cm.cmpq_mass_state, raw_masses)
    return nothing
end

"""
    cmpq_mass_gradient_vec(base, verify, ctx_cm, raw_masses) -> Vector{Float64}

`d(Delta_dual)/d(raw_k)` for the `L-1` raw mass coordinates at the REAL converged inner dual carried
by `base`/`verify`: the exact closed-form envelope derivative (`d_delta_dual_d_mu_shared`)
chain-ruled through the stick-breaking transform (`chain_cmpq_mass_gradient_to_raw`).

NO SIGN FLIP HERE, deliberately -- see this file's header. Gated end to end by
`test_cm_pairwise_quantile_outer_gradient_fd.jl` (reoptimized FD, rel L2 1.69e-9 at a non-uniform
mu, with a sign negative control that fails if anyone adds one).
"""
function cmpq_mass_gradient_vec(base::BaseDualState, verify, ctx_cm,
                                raw_masses::AbstractVector{Float64})
    ensure_cmpq_masses!(ctx_cm, raw_masses)
    cmpq = ctx_cm.cmpq_ctx
    op = cmpq.op
    λ_L, λ_P, _ = reshape_cmpq_duals_lambda(base.λstar, op, cmpq.ncore1)
    hasproperty(verify, :m_mean) ||
        error("cmpq_mass_gradient_vec: verify carries no m_mean -- the envelope derivative needs " *
              "the LFD weight mean, exactly as origin-ZC's own nu gradient does.")
    mu = ctx_cm.cmpq_mass_state.mu
    d_mu = d_delta_dual_d_mu_shared(collect(λ_L), Array(λ_P), mu, op; mean_m = verify.m_mean)
    return chain_cmpq_mass_gradient_to_raw(d_mu, collect(raw_masses), mu)
end

"""
    cmpq_restriction_q0_contribution!(contrib, ctx_cm, base) -> contrib

`contrib = -(G_R*lambda_R + G_CM*lambda_CM)` at the converged duals -- the term the plain economic
`q0` is missing, for BOTH of this family's restriction blocks.

THE ONE DEFINITION of that fold, called by the dense path (`build_lfix_base_cache_cmpq`) AND the
Backend C+ path (`build_lfix_base_cache_cmpq_C!`). It is factored out rather than written twice on
purpose: this family has TWO blocks to fold where every other family has one, so "the two paths
each fold both blocks" is exactly the invariant most likely to rot when someone edits one of them.
Now there is nothing to keep in sync.

`contrib` is OVERWRITTEN (not accumulated into), and both operators SUBTRACT into it, which is why
the caller ADDS the result to `cache.q0`. Same operators the inner solve itself uses -- never a
hand-written second copy of the moment algebra.
"""
function cmpq_restriction_q0_contribution!(contrib::Vector{Float64}, ctx_cm, base::BaseDualState)
    cmpq = ctx_cm.cmpq_ctx
    op = cmpq.op
    Lcm = cmpq.Lcm; nO = cmpq.nO
    ncm_cdf = nO * Lcm
    length(contrib) == op.W ||
        error("cmpq_restriction_q0_contribution!: length(contrib)=$(length(contrib)) != W=$(op.W)")
    fill!(contrib, 0.0)
    λ_L, λ_P, λ_CM = reshape_cmpq_duals_lambda(base.λstar, op, cmpq.ncore1)

    # (1) level + pair rows
    cm_pq_forward!(contrib, λ_L, λ_P, op, ctx_cm.cmpq_mass_state, cmpq.refIndex1)

    # (2) the CM grid -- the block the standalone family has no analogue of
    λmat_block = zeros(nO, Lcm); λmat_ext = zeros(nO, Lcm + 1); cm_contrib = zeros(op.W)
    λ_cdf = cmpq.Pow === nothing ? λ_CM : (@view λ_CM[1:ncm_cdf])
    apply_contrast!(λmat_block, reshape(λ_cdf, nO, Lcm), cmpq.R)
    suffix_sums!(λmat_ext, λmat_block)
    cumulative_forward_contribution!(cm_contrib, cmpq.Bidx, cmpq.refIndex1, cmpq.origins, λmat_ext)
    contrib .-= cm_contrib
    if cmpq.Pow !== nothing
        λmat_block2 = zeros(nO, Lcm); λmat_ext2 = zeros(nO, Lcm + 1); cm_contrib2 = zeros(op.W)
        λ_pow = @view λ_CM[ncm_cdf+1 : 2*ncm_cdf]
        apply_contrast!(λmat_block2, reshape(λ_pow, nO, Lcm), cmpq.R)
        suffix_sums!(λmat_ext2, λmat_block2)
        cumulative_forward_contribution_pow!(cm_contrib2, cmpq.Bidx, cmpq.refIndex1, cmpq.origins,
                                             λmat_ext2, cmpq.Pow)
        contrib .-= cm_contrib2
    end
    return contrib
end

"""
    cmpq_assert_q0_matches_r(q0_new, verify, tol, where) -> nothing

The exact cross-check both cache builders run: `q0` corrected by the fold above must equal the
verifier's INDEPENDENTLY recomputed `r`, since both are
`r = -zeta - E*lambda_E - G_R*lambda_R - G_CM*lambda_CM` by two different routes. A HARD ERROR, not
a warning -- a mismatch means one of the two folds is missing or wrong, and every economic gradient
built on that cache would be silently wrong. Shared for the same reason the fold itself is.
"""
function cmpq_assert_q0_matches_r(q0_new::AbstractVector{Float64}, verify, tol::Float64,
                                  where::AbstractString)
    (verify === nothing || !hasproperty(verify, :r_current)) && return nothing
    err = maximum(abs, q0_new .- verify.r_current)
    err <= tol ||
        error("$where: corrected q0 disagrees with the independently recomputed r by " *
              "max|diff|=$err (tol=$tol). These are the same quantity by two routes -- a mismatch " *
              "means one of the TWO restriction folds (level+pair, CM-grid) is missing or wrong, " *
              "and every economic gradient built on it would be silently wrong. Refusing to continue.")
    return nothing
end

"""
    build_lfix_base_cache_cmpq(x_free0, ctx_cm, base; verify=nothing, q0_check_tol=1e-8, econ_ws=nothing)
        -> LFixBaseCache

This family's analog of `build_lfix_base_cache_pairwise_quantile`: build the plain economic
`LFixBaseCache`, then fold this family's OWN restriction contribution into `cache.q0` via `with_q0`.

**BOTH restriction blocks must be folded** -- `-G_R*lambda_R` (level + pair) AND
`-G_CM*lambda_CM` (the CM grid). This is the one place this family genuinely differs from the
standalone one, and the failure mode is silent: `q0` is the per-draw LEVEL that the economic block
linearizes AROUND, so omitting either term linearizes about the wrong base point and produces an
economic gradient wrong by an amount unrelated to the restriction's own gradient. The argument that
the CM rows do not depend on theta is true about the DERIVATIVE and irrelevant here (memory
`feedback-q0-restriction-fold-is-a-level-not-a-derivative`).

Both contributions are obtained from the SAME operators the inner solve uses, starting from zeros
(each SUBTRACTS into its accumulator), never from a hand-written second copy of the moment algebra.

`verify`'s independently recomputed `r_current` is then an exact cross-check: the corrected `q0` must
equal it to floating point, since both are `r = -zeta - E*lambda_E - G_R*lambda_R - G_CM*lambda_CM`
by two different routes. A mismatch is a HARD ERROR, and it is precisely what would fire if one of
the two folds were dropped.
"""
function build_lfix_base_cache_cmpq(x_free0::AbstractVector, ctx_cm, base::BaseDualState;
        verify = nothing, q0_check_tol::Float64 = 1e-8,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing)
    # Persistent-workspace path, for the same reason and with the same caveat as the standalone
    # family's: `build_lfix_base_cache` allocates W x D x Ddest tensors fresh (~700 MB per gradient
    # call at W=100k/D=20), and a restriction-folded caller must pass a pre-built cache, which routes
    # around `economic_A_gradient!`'s own persistent workspace. Building IN PLACE into that same
    # workspace and then applying `with_q0` (a pure field copy) keeps the mandatory fold AND drops
    # the allocation. This removes the ALLOCATION, not the compute.
    cache0 = if econ_ws === nothing
        build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = false)
    else
        D = ctx_cm.D
        Ddest = hasproperty(ctx_cm, :D_dest) ? ctx_cm.D_dest : ctx_cm.D
        Wc = size(ctx_cm.obj.U, 1)
        lws = econ_ws.lfix_ws
        if lws === nothing || lws.D != D || lws.Ddest != Ddest || lws.W != Wc
            lws = build_lfix_base_workspace(D, Ddest, Wc)
            econ_ws.lfix_ws = lws
        end
        build_lfix_base_cache!(lws, x_free0, ctx_cm, base; validate_dense = false)
    end

    contrib = zeros(ctx_cm.cmpq_ctx.op.W)
    cmpq_restriction_q0_contribution!(contrib, ctx_cm, base)   # BOTH blocks; see that function
    q0_new = cache0.q0 .+ contrib
    cmpq_assert_q0_matches_r(q0_new, verify, q0_check_tol, "build_lfix_base_cache_cmpq")
    return with_q0(cache0, q0_new)
end

"Optional wall-clock accumulator for the gradient's four components, so a profiler reports a
decomposition that SUMS TO THE TOTAL rather than timing the pieces again in separate calls."
mutable struct CMPQGradTimers
    t_solve::Float64
    t_cache::Float64
    t_econ::Float64
    t_mass::Float64
    t_total::Float64
end
CMPQGradTimers() = CMPQGradTimers(0.0, 0.0, 0.0, 0.0, 0.0)

"""
    cm_pairwise_quantile_production_gradient(x_free0, raw_masses, pcx, ctx, pe;
        base=nothing, verify=nothing, econ_ws=nothing, timers=nothing, kwargs...) -> (g_ext, meta)

Full outer gradient, `g_ext = vcat(g_econ, g_mass)`, length `D*Ddest + (L-1)`.

The economic block is NOT reimplemented for this family: it is the same `economic_A_gradient!`
(shared_a_gradient.jl) every other family calls, on a restriction-aware `LFixBaseCache` built by
`build_lfix_base_cache_cmpq` (whose docstring records why folding BOTH restriction blocks is
mandatory).

Note the length: `D*Ddest + 4` at D=20/L=5, against the standalone family's `D*Ddest + 80`.
"""
function cm_pairwise_quantile_production_gradient(x_free0::AbstractVector,
        raw_masses::AbstractVector{Float64}, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing,
        timers::Union{Nothing,CMPQGradTimers} = nothing, kwargs...)
    t_enter = time()
    if base === nothing || verify === nothing
        t0 = time()
        base, verify = archCMPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm)
        timers === nothing || (timers.t_solve += time() - t0)
    end
    # Both the q0 fold and the closed-form mass gradient read pcx.ctx_cm.cmpq_mass_state, which is
    # mutable per-outer-point state some intervening solve may have moved -- see ensure_cmpq_masses!.
    ensure_cmpq_masses!(pcx.ctx_cm, raw_masses)
    D = pcx.ctx_cm.D
    Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
    Wc = size(pcx.ctx_cm.obj.U, 1)
    ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(Wc) : econ_ws
    t0 = time()
    cache = build_lfix_base_cache_cmpq(x_free0, pcx.ctx_cm, base; verify = verify, econ_ws = ws)
    timers === nothing || (timers.t_cache += time() - t0)
    g_econ = zeros(D * Ddest)
    t0 = time()
    meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    timers === nothing || (timers.t_econ += time() - t0)
    t0 = time()
    g_mass = cmpq_mass_gradient_vec(base, verify, pcx.ctx_cm, raw_masses)
    timers === nothing || (timers.t_mass += time() - t0)
    timers === nothing || (timers.t_total += time() - t_enter)
    return vcat(g_econ, g_mass), meta
end

"""
    d_delta_dual_d_cmpq_mass_fd(x_free0, raw_masses, ctx_cm; h, coords=nothing, verbose=false)
        -> (g_fd, n_probed)

Reoptimized (NOT fixed-dual) central finite difference of `Delta_dual` w.r.t. the raw mass
coordinates -- the trusted ground truth for `cmpq_mass_gradient_vec`. Each probe RE-SOLVES the inner
dual from scratch through the VERIFIED path, so it is genuinely independent of the machinery it
validates.

A plain small `h` is correct here: the masses shift moment TARGETS smoothly and never reassign a
draw between bins, so `Delta_dual` is smooth in them. Measured agreement at D=4/L=5/G=50/two
families is 1.69e-9 relative at `h=1e-5`, falling as `h^2` down the ladder. If it does not agree, do
NOT reintroduce a bandwidth to make it look better.

`h` is REQUIRED, not defaulted: it is the single knob that decides whether this ground truth is
truncation- or noise-dominated, and a caller that has not thought about it should not get a number.

Only `L-1` coordinates exist, so probing all of them costs `2(L-1)` inner solves -- affordable even
at real D=20, unlike the standalone family's `2*D*(L-1)`.
"""
function d_delta_dual_d_cmpq_mass_fd(x_free0::AbstractVector, raw_masses::AbstractVector{Float64},
        ctx_cm; h::Float64, coords::Union{Nothing,AbstractVector{Int}} = nothing,
        verbose::Bool = false)
    n = n_cmpq_raw(ctx_cm.cmpq_ctx.L)
    length(raw_masses) == n ||
        error("d_delta_dual_d_cmpq_mass_fd: length(raw_masses)=$(length(raw_masses)) != $n")
    h > 0 || error("d_delta_dual_d_cmpq_mass_fd: h must be > 0, got $h")
    idxs = coords === nothing ? collect(1:n) : collect(coords)
    g = fill(NaN, n)
    n_probed = 0
    for j in idxs
        1 <= j <= n || error("d_delta_dual_d_cmpq_mass_fd: coordinate $j out of range 1:$n")
        rp = collect(raw_masses); rp[j] += h
        rm = collect(raw_masses); rm[j] -= h
        _, vp = archCMPQ_verified_state(x_free0, rp, ctx_cm)
        _, vm = archCMPQ_verified_state(x_free0, rm, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
        n_probed += 1
        verbose && (println("  [fd] coord $j: h=", h, "  Delta(+)=", vp.Delta_dual,
                            "  Delta(-)=", vm.Delta_dual, "  fd=", g[j]); flush(stdout))
    end
    # Restore the caller's own point: the mass state is left wherever the last probe wrote it, and
    # this function must have no side effect on a live outer loop that calls it mid-run.
    ensure_cmpq_masses!(ctx_cm, raw_masses)
    return (g, n_probed)
end
