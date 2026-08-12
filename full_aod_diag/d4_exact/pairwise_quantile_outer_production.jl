# ================================================================================================
# OUTER-loop production layer for the pairwise-quantile-independence restriction (draft eq. 32),
# version B: FIXED cutoffs + FREE bin masses (2026-08-10).
#
# Everything in pairwise_quantile_production.jl is the INNER solve: given a FIXED outer point
# (theta, masses), run the KNITRO inner dual solve. This file adds the three things an outer loop
# needs on top of that, mirroring cm_originzc_production.jl's own structure function-for-function:
#
#   1. archPQ_verified_state          <- archOZ_verified_state
#      A verified inner solve returning (base::BaseDualState, verify::NamedTuple). `verify` carries
#      `Delta_dual` -- the scalar the outer KNITRO callback actually consumes. Built by handing this
#      restriction's own verifier's output to the SAME family-agnostic
#      `verify_namedtuple_from_operator` (operator_verification.jl) every other family uses, so
#      `classify_inner_result`/`is_verified_success` (oracle.jl) accept it unchanged.
#
#   2. pairwise_quantile_production_gradient  <- cm_originzc_production_gradient
#      The combined outer gradient, `vcat(g_econ, g_mass)`:
#        - `g_econ`: the shared, family-agnostic (g, A_od) economic block, via `economic_A_gradient!`
#          (shared_a_gradient.jl). NOT reimplemented here.
#        - `g_mass`: the EXACT closed-form envelope derivative `d(Delta_dual)/d(raw mass coord)`,
#          via `pairwise_quantile_mass_gradient.jl` -- a mirror of origin-ZC's
#          `d_delta_dual_d_eta_origin_vec`, not a new gradient engine.
#
#   3. d_delta_dual_d_mass_fd         <- d_delta_dual_d_eta_origin_fd
#      Reoptimized (NOT fixed-dual) finite-difference ground truth for the mass block: every probe
#      re-solves the inner dual from scratch, at a plain small `h`.
#
# WHAT CHANGED FROM VERSION A, AND WHY THE VALIDATION BAR WENT UP RATHER THAN DOWN.
# Version A's outer coordinates were the quantile CUTOFFS, which live inside indicator functions:
# `Delta_dual` was a genuine STEP function of every one of them, its exact derivative was zero
# between draw crossings and undefined at them, and both the analytic gradient and its FD ground
# truth had to be matched-bandwidth secants of the same staircase -- which could only ever agree to
# ~20%. Under version B the outer coordinates shift moment TARGETS smoothly and never reassign a
# draw between bins, so `Delta_dual` is smooth in them, a plain small-`h` reoptimized central FD is
# a VALID ground truth, and the gate is held at ~1e-5 relative -- the same standard
# `test_cm_originzc_pure_moments.jl` holds origin-ZC's own nu-gradient to. A few percent is NOT
# acceptable here: the loose version-A tolerance was a property of the step-function objective, not
# a standard to inherit.
#
# SIGN CONVENTION. The inner KNITRO solve MINIMIZES `f = mean(Psi(r)) + zeta`, and the reported
# divergence is `Delta_dual = -f` (verify_namedtuple_from_operator, operator_verification.jl).
# `d_delta_dual_d_mu` (pairwise_quantile_mass_gradient.jl) already differentiates `Delta_dual`, not
# `f` -- the `-mean_m` prefactor in its own derivation IS that sign -- so, unlike version A's
# `pairwise_quantile_cutoff_gradient_vec`, NOTHING is negated at this layer. There is exactly one
# sign, it lives in the derivation, and the FD gate asserts it with a negative control.
#
# Requires pairwise_quantile_production.jl (and its own include chain),
# pairwise_quantile_mass_gradient.jl, shared_a_gradient.jl, operator_verification.jl, and oracle.jl
# to already be included.
# ================================================================================================

using LinearAlgebra: BLAS, dot, norm

isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))
isdefined(Main, :verify_namedtuple_from_operator) || include(joinpath(@__DIR__, "operator_verification.jl"))
isdefined(Main, :d_delta_dual_d_mu) || include(joinpath(@__DIR__, "pairwise_quantile_mass_gradient.jl"))

"""
    build_pairwise_quantile_production_context(ctx, layout; cutoff_source, min_bin_count)
        -> (ctx_cm, aug, hess_ctx, layout, Q, cutoff_source, min_bin_count, bin_report)

Analog of `build_originzc_production_context`: assembles the restriction-augmented context ONCE per
run. Returns a NamedTuple with a `.ctx_cm` field, which is what `prepare_production_run`'s
`_resolve_bundle` (production_bundle_api.jl) looks for -- so a driver wraps this call in
`prepare_production_run(:pairwise_quantile, "<runner name>", () -> build_..._context(...))` and gets
the OperatorPsiBundle invariant asserted for free, exactly like every other family.

`ctx` MUST be the plain, unaugmented economic context (`d20_real_setup_design`/`d4_exact_setup`).
It is carried forward on `ctx_cm.pq_econ_ctx` because `archPQ_base_state`/`archPQ_verified_state`
genuinely need BOTH: `cf_build`/`prime_operator!` read dimensionality off the ECONOMIC-only bundle,
and passing the augmented context there corrupts `cf.oci` (a real bug caught live on 2026-08-09 --
origin-ZC keeps the same thing on `octx.econ_ctx` for exactly this reason).

THREE MODELLING CHOICES, ALL REQUIRED WITH NO DEFAULT (CLAUDE.md's no-silent-defaults rule):
  - `L` (via `layout`) -- the number of quantile bins per origin.
  - `cutoff_source` -- WHERE the now-fixed cutoffs sit (`:frechet_theoretical` |
    `:empirical_quantile`). The entire meaning of the restriction depends on this, and a run is not
    reproducible without it, so it is recorded in the checkpoint alongside the resulting cutoff
    matrix itself. See `pairwise_quantile_fixed_cutoffs`.
  - `min_bin_count` -- the non-degeneracy floor asserted on every marginal bin AND every joint
    cell. It must be read against `W` and `L` and cannot have a default; see
    `assert_pairwise_quantile_bins_nondegenerate`.

(`min_crossed`, version A's secant bandwidth, is GONE: version B's outer gradient is closed-form and
has no bandwidth.)
"""
function build_pairwise_quantile_production_context(ctx, layout::PairwiseQuantileMassLayout;
        cutoff_source::Symbol, min_bin_count::Int)
    ctx.D == layout.D ||
        error("build_pairwise_quantile_production_context: ctx.D=$(ctx.D) != layout.D=$(layout.D)")
    W = size(ctx.U, 1)

    # The restriction is stated on the FRECHET PRODUCTIVITY z_o = U_o^(-muHat), not on the Exp(1)
    # draws `ctx.U` themselves (user directive, 2026-08-10). Z is built once here, via the shared
    # `frechet_power_feature`, and used for BOTH the cutoffs and the bin assignment; nothing
    # downstream keeps it, since only `op.bin` is ever read again.
    hasproperty(ctx, :μHat) ||
        error("build_pairwise_quantile_production_context: ctx carries no μHat -- this restriction " *
              "is defined on the Frechet productivity z = U^(-μHat) and cannot be built without it.")
    Z = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)

    Q = pairwise_quantile_fixed_cutoffs(Z, layout.L; cutoff_source = cutoff_source,
                                        mu_frechet = ctx.μHat)
    aug = build_pairwise_quantile_augmented_obj(ctx, layout, Z, Q)
    bin_report = assert_pairwise_quantile_bins_nondegenerate(aug.op; min_bin_count = min_bin_count)

    println(stdout, "cm_restriction_basis [pairwise_quantile] = FRECHET-z quantile bins (L=", layout.L,
            "), FIXED cutoffs (source=:", cutoff_source, ", muHat=", ctx.μHat,
            "), FREE bin masses; n_total_rows=", n_total_rows(ctx.D, layout.L))
    println(stdout, "pairwise_quantile outer mass coordinates: n_raw=", n_raw(layout),
            " (origin-major, stick-breaking simplex transform); occupancy: min marginal bin=",
            bin_report.min_marginal_count, ", min joint cell=", bin_report.min_joint_count,
            " (floor min_bin_count=", min_bin_count, ", W=", W, ")")
    flush(stdout)

    mass_state = PairwiseQuantileMassState(ctx.D, layout.L)
    hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, mass_state, aug.core_cf_ref)
    ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_mass_state = mass_state,
                          pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx,
                          pq_econ_ctx = ctx, pq_layout = layout, pq_cutoffs = Q,
                          pq_cutoff_source = cutoff_source, pq_min_bin_count = min_bin_count,
                          pq_mu_frechet = ctx.μHat))
    return (ctx_cm = ctx_cm, aug = aug, hess_ctx = hess_ctx, layout = layout, Q = Q,
            cutoff_source = cutoff_source, min_bin_count = min_bin_count, bin_report = bin_report,
            mu_frechet = ctx.μHat)
end

"""
    archPQ_verified_state(x_free0, raw_masses, ctx_cm; dual_bank=nothing, eval_id=0) -> (base, verify)

Verified analog of `archPQ_base_state`, mirroring `archOZ_verified_state` (cm_originzc_production.jl)
exactly: solve, then INDEPENDENTLY recompute the solution's residual/objective/KKT blocks with this
restriction's own verifier (`verify_inner_solution_operator_pairwisequantile!`, never reading any
FG-callback-cached state), then hand that to the shared `verify_namedtuple_from_operator` so the
returned `verify` carries precisely the field set `classify_inner_result`/`is_verified_success`
(oracle.jl) already know how to read -- no new acceptance predicate is invented for this family.

`verify` additionally carries, beyond the shared field set:
  - `r_current`: the converged per-draw `R_w` from the INDEPENDENT verifier recompute. The `q0`
    restriction fold cross-checks against this.
  - `mu`/`mu_last`/`Pcum`: the masses this solve was actually run at, so a gradient or a diagnostic
    downstream can assert it is looking at the same outer point.
  - the restriction's own block KKT residuals and probability/cumulative-residual tables, for
    campaign logging.

Throws `CMExpectedSolveFailure` (reused, not redefined) on an infeasible/failed inner solve, exactly
as every other family's `*_verified_state` does, so the driver's own `cb_F!` can `reject_point`.
"""
function archPQ_verified_state(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, ctx_cm;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    obj = ctx_cm.obj
    layout = ctx_cm.pq_layout
    econ_ctx = ctx_cm.pq_econ_ctx
    op = ctx_cm.pq_op
    length(raw_masses) == n_raw(layout) ||
        error("archPQ_verified_state: length(raw_masses)=$(length(raw_masses)) != n_raw(layout)=$(n_raw(layout))")

    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, vcat(collect(x_free0), collect(raw_masses)))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end

    nStatus, inner_x, _, n_fg, n_hess = archPQ_base_state(x_free0, raw_masses, econ_ctx, ctx_cm, layout)
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archPQ_verified_state: inner solve failed, nStatus=$nStatus " *
                                     "(x_free0=$x_free0, raw_masses=$raw_masses)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = op.W
    ncore1 = obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)

    cf = ctx_cm.pq_core_cf_ref[]
    cf isa CompressedFactual ||
        error("archPQ_verified_state: pq_core_cf_ref[] is not a CompressedFactual -- prime_operator! " *
              "did not run for this outer point (got $(typeof(cf)))")
    econ_ws = economic_operator_workspace(cf)
    ov = verify_inner_solution_operator_pairwisequantile!(ζstar, λstar, cf, op, ctx_cm.pq_mass_state, W,
        economic_forward!, economic_transpose!, econ_ws, obj.Psi!, obj.dPsi!, ncore1)
    m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    verify = merge(verify, (r_current = ov.r,
                            kkt_resid_E = ov.kkt_resid_E,
                            kkt_resid_marginalbin = ov.kkt_resid_marginalbin,
                            kkt_resid_pairindep = ov.kkt_resid_pairindep,
                            max_cumulative_residual = ov.max_cumulative_residual,
                            max_marginal_cumulative_residual = ov.max_marginal_cumulative_residual,
                            marginal_prob = ov.marginal_prob,
                            mu = ov.mu, mu_last = ov.mu_last, Pcum = ov.Pcum,
                            n_fg = n_fg, n_hess = n_hess))

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, vcat(collect(x_free0), collect(raw_masses)), inner_x)
    return base, verify
end

"pairwise_quantile_production_value_verified(x_free0, raw_masses, pcx) -> (K, base, verify). Analog of `cm_originzc_production_value_verified`; `K` is `obj.H_save`, the same payoff scalar every family reports."
function pairwise_quantile_production_value_verified(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, pcx;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    base, verify = archPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm; dual_bank = dual_bank, eval_id = eval_id)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    pairwise_quantile_production_value_verified_screened(x_free0, raw_masses, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Screened drop-in, mirroring `cm_originzc_production_value_verified_screened` (cm_screen_bridge.jl)
verbatim: the SAME family-agnostic `cm_screen_precheck!` first, then this family's verified state.
"""
function pairwise_quantile_production_value_verified_screened(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, pcx;
        counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false,
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    cm_screen_precheck!(x_free0, pcx.ctx_cm; counters = counters, use_witness = use_witness)
    return pairwise_quantile_production_value_verified(x_free0, raw_masses, pcx; dual_bank = dual_bank, eval_id = eval_id)
end

"""
    ensure_pq_masses!(ctx_cm, raw_masses) -> nothing

Refresh `ctx_cm.pq_mass_state` so it corresponds to `raw_masses`, before any OUTER-loop consumer
reads it.

`pq_mass_state` is deliberately MUTABLE per-outer-point state, rebuilt once per inner solve by
`reset_for_solve!` (pairwise_quantile_production.jl) -- the restriction's own "decode once per outer
point, never inside a callback" requirement. That is correct for the inner solve, but it makes every
OUTER-loop consumer (the closed-form mass gradient, the `q0` restriction fold) implicitly dependent
on ambient state that some LATER solve may since have overwritten. In a checkpointed outer driver
that is a real hazard, not a hypothetical: `cb_G!` may run against a cached `base` from an earlier
`cb_F!`, a verification re-solve or a diagnostic FD probe can land in between, and nothing about the
resulting gradient would look wrong -- it would just silently be computed at a different point's
masses.

This is the version-B successor to `ensure_pq_bins!` (version A had the identical hazard on the BIN
state, and it bit live on 2026-08-10: an FD gate ran probes at other cutoff points and then computed
a gradient at the original one; the `q0` cross-check in
`build_lfix_base_cache_pairwise_quantile` caught the resulting max|diff| = 0.62 against an
independently recomputed `r`). Under version B the bins themselves are campaign constants and cannot
go stale at all; only the masses can, so only the masses are refreshed here -- and the `q0`
cross-check is retained behind this call as the backstop that would catch any remaining
inconsistency (e.g. a `base` that came from a genuinely different outer point, which no amount of
refreshing can repair).

Cost is `O(D*L)` -- negligible beside the inner KNITRO solve this sits next to, and not worth
trading for a staleness guess.
"""
function ensure_pq_masses!(ctx_cm, raw_masses::AbstractVector{Float64})
    set_pairwise_quantile_masses!(ctx_cm.pq_mass_state, raw_masses, ctx_cm.pq_layout)
    return nothing
end

"""
    reshape_pq_duals(lambda, op, ncore1) -> (lambda_M, lambda_P)

The ONE place the flat inner-dual vector is split into this restriction's marginal/pair dual blocks
for OUTER-loop use. Uses the identical `reshape(v, nc, D)'` / `reshape(v, nc, nc, npair)` convention
as `dual_index!` (pairwise_quantile_production.jl) and the verifier
(pairwise_quantile_verification.jl) -- see `dual_index!`'s own comment for why the marginal block
needs the transpose (`marginal_row(o,a,L)` is O-MAJOR, Julia's `reshape` is A-MAJOR) and the pair
block does not. Getting this wrong swaps dual columns silently; it was already a real bug once
(found live 2026-08-09), which is why all three sites now name the same convention explicitly.
"""
function reshape_pq_duals(lambda::AbstractVector{Float64}, op::PairwiseQuantileOperator, ncore1::Int)
    D = op.D; npair = op.npair; L = op.L; nc = L - 1
    nM = n_mean_flat(D, L); nP = n_pair_flat(npair, L)
    length(lambda) == ncore1 + nM + nP ||
        error("reshape_pq_duals: length(lambda)=$(length(lambda)) != ncore1+nM+nP=$(ncore1 + nM + nP)")
    λ_M = reshape(@view(lambda[ncore1+1:ncore1+nM]), nc, D)'
    λ_P = reshape(@view(lambda[ncore1+nM+1:ncore1+nM+nP]), nc, nc, npair)
    return (λ_M, λ_P)
end

"""
    pairwise_quantile_mass_gradient_vec(base, verify, ctx_cm, raw_masses) -> Vector{Float64}

`d(Delta_dual)/d(raw_k)` for every one of the `n_raw(layout)` raw mass coordinates, evaluated at the
REAL converged inner dual carried by `base`/`verify`: the exact closed-form envelope derivative
(`d_delta_dual_d_mu`) chain-ruled through the stick-breaking transform
(`chain_mass_gradient_to_raw`).

NO SIGN FLIP HERE, deliberately, and this is a difference from version A worth stating: version A's
`pairwise_quantile_cutoff_gradient_vec` had to negate, because `cutoff_secant_gradient!`
differentiated the inner objective `f` while the outer loop consumes `Delta_dual = -f`.
`d_delta_dual_d_mu` differentiates `Delta_dual` directly -- the `-mean_m` prefactor in its
derivation IS that sign -- so negating here would flip it back. The FD gate carries a negative
control that fails if anyone adds one.
"""
function pairwise_quantile_mass_gradient_vec(base::BaseDualState, verify, ctx_cm,
        raw_masses::AbstractVector{Float64})
    ensure_pq_masses!(ctx_cm, raw_masses)   # see ensure_pq_masses!: pq_mass_state is mutable ambient state
    op = ctx_cm.pq_op
    layout = ctx_cm.pq_layout
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    λ_M, λ_P = reshape_pq_duals(base.λstar, op, ncore1)
    mu = ctx_cm.pq_mass_state.mu
    hasproperty(verify, :m_mean) ||
        error("pairwise_quantile_mass_gradient_vec: verify carries no m_mean -- the envelope " *
              "derivative needs the LFD weight mean, exactly as origin-ZC's own nu gradient does.")
    d_mu = d_delta_dual_d_mu(λ_M, λ_P, mu, op; mean_m = verify.m_mean)
    return chain_mass_gradient_to_raw(d_mu, raw_masses, mu, layout)
end

"""
    build_lfix_base_cache_pairwise_quantile(x_free0, ctx_cm, base; verify=nothing) -> LFixBaseCache

This family's analog of `build_lfix_base_cache_originzc` (cm_originzc_production.jl) and
`build_lfix_base_cache_cm_meanzc`: build the plain economic `LFixBaseCache`, then fold THIS
restriction's own `G_R*lambda_R` contribution into `cache.q0` via the shared `with_q0` mechanism
(lfix_cm_aware.jl).

**This fold is required, and reasoning that it is not is a live trap this task fell into once, and
the reparameterization does not change that.** The tempting argument -- "the restriction's bin
memberships depend only on `U` and the (now fixed) cutoffs, never on theta, so the restriction
contributes nothing to the theta-gradient" -- is true about the DERIVATIVE and irrelevant to `q0`.
`q0` is not a derivative: it is the per-draw LEVEL `q0[s] = -zeta* - sum_j lambda*_j G[s,j]` that
the economic block linearizes AROUND (`build_lfix_base_cache`, lfix_incremental.jl, computes it
from the ECONOMIC moment columns only). Omitting the restriction's own `G_R*lambda_R` term
linearizes the economic gradient about the wrong base point, producing an economic gradient block
that is wrong by an amount that has nothing to do with the restriction's own gradient. Caught live
by `test_pairwise_quantile_outer_gradient_fd.jl`'s economic section (two of four probed economic
coordinates came out ~70x off, with correct-looking values on the other two). See memory
`feedback-q0-restriction-fold-is-a-level-not-a-derivative`.

The contribution is obtained from `pairwise_quantile_forward!` itself -- the SAME operator the inner
solve uses -- rather than from a hand-written second copy of the moment algebra: starting from zeros
it accumulates `-G_R*lambda_R` (it SUBTRACTS into its accumulator), which is exactly the term `q0`
is missing, so it is simply added.

`verify`: when supplied, its independently recomputed `r_current` is used as an exact cross-check --
the corrected `q0` must equal it to floating-point tolerance, since both are the same quantity
`r = -zeta - E*lambda_E - G_R*lambda_R` computed by two different routes (closed-form economic cache
+ operator fold here; full independent operator recompute there). A mismatch is a hard error, not a
warning: it means the fold or the cache is wrong, and every economic gradient built on it would be
silently wrong.
"""
function build_lfix_base_cache_pairwise_quantile(x_free0::AbstractVector, ctx_cm, base::BaseDualState;
        verify = nothing, q0_check_tol::Float64 = 1e-8,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing)
    # PERSISTENT-WORKSPACE PATH (2026-08-11). `build_lfix_base_cache` allocates the price and
    # p^(T-sigma) tensors fresh as `W x D x Ddest` EACH -- 304 MB apiece at W=100,000/D=20/Ddest=19,
    # ~700 MB per gradient call measured. The tensors' CONTENTS genuinely must be recomputed (they
    # depend on theta, which moves every outer iteration), but the BUFFERS need not be reallocated:
    # `build_lfix_base_cache!` (lfix_base_workspace.jl) fills a persistent `LFixBaseWorkspace` in
    # place, and `economic_A_gradient!` already routes through exactly that when it is handed
    # `cache === nothing` (shared_a_gradient.jl:454-466).
    #
    # Restriction-folded callers could not use it, because they must fold their own `G_R*lambda_R`
    # into `q0` and therefore have to pass a pre-built `cache=` -- which routes around the
    # persistent workspace. `EconomicAGradientWorkspace`'s own docstring says as much
    # ("restriction-folded callers that pass their own pre-built `cache=` are unaffected"), and
    # origin-ZC and CM+ZC still pay the allocation for this reason.
    #
    # There is no actual conflict: build the cache IN PLACE into the same workspace
    # `economic_A_gradient!` would have used, then apply the fold with `with_q0`, which is a pure
    # field copy (lfix_cm_aware.jl:47) and does not touch the workspace's own arrays. Passing
    # `econ_ws` therefore keeps the mandatory q0 fold AND drops the per-gradient allocation.
    #
    # NOTE this removes the ALLOCATION, not the compute: neither `build_lfix_base_cache` nor
    # `build_lfix_base_cache!` threads the `Ddest x D` price/p^(T-sigma) fill (grep: zero
    # `Threads.@threads` in either), which measured 3.05 s of the 3.32 s at W=100k/L=5. That loop is
    # embarrassingly parallel over its 380 independent (o,d) cells and is shared by every family --
    # flagged, deliberately not changed here mid-campaign.
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
    op = ctx_cm.pq_op
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    λ_M, λ_P = reshape_pq_duals(base.λstar, op, ncore1)
    contrib = zeros(op.W)
    pairwise_quantile_forward!(contrib, λ_M, λ_P, op, ctx_cm.pq_mass_state)   # contrib = -G_R*lambda_R
    q0_new = cache0.q0 .+ contrib
    if verify !== nothing && hasproperty(verify, :r_current)
        err = maximum(abs, q0_new .- verify.r_current)
        err <= q0_check_tol ||
            error("build_lfix_base_cache_pairwise_quantile: corrected q0 disagrees with the " *
                  "independently recomputed r by max|diff|=$err (tol=$q0_check_tol). These are the " *
                  "same quantity by two routes -- a mismatch means the restriction fold or the " *
                  "economic cache is wrong, and any economic gradient built on it would be silently " *
                  "wrong. Refusing to continue.")
    end
    return with_q0(cache0, q0_new)
end

"""
    PQGradTimers()

Optional wall-clock accumulator for `pairwise_quantile_production_gradient`'s four components, so a
profiler can report a decomposition that SUMS TO THE TOTAL instead of timing the pieces again in
separate calls (which double-counts warm-up and drifts from what production actually executes --
the exact sloppiness that produced two wrong profiling conclusions on 2026-08-10/11).

Default is `nothing`, i.e. not passed, i.e. four untaken branches per gradient call -- unmeasurable
against a multi-second gradient, so there is no "profiling build" to keep in sync with production.
"""
mutable struct PQGradTimers
    t_solve::Float64      # inner re-solve, only when base/verify were not supplied
    t_cache::Float64      # shared LFix cache build (+ this family's q0 fold and its exact check)
    t_econ::Float64       # shared economic_A_gradient!
    t_mass::Float64       # this restriction's closed-form mass gradient
    t_total::Float64
end
PQGradTimers() = PQGradTimers(0.0, 0.0, 0.0, 0.0, 0.0)

"""
    pairwise_quantile_production_gradient(x_free0, raw_masses, pcx, ctx, pe;
                                          base=nothing, verify=nothing, econ_ws=nothing,
                                          kwargs...) -> (g_ext, meta)

Full outer gradient for this family, structurally identical to `cm_originzc_production_gradient`:
the shared (g, A_od) economic block, then this restriction's own block appended --
`g_ext = vcat(g_econ, g_mass)`, length `D*Ddest + n_raw(layout)`.

The economic block is NOT reimplemented and NOT modified for this family: it is the same
`economic_A_gradient!` (shared_a_gradient.jl) call origin-ZC makes, with the same
`get_or_build_econ_a_grad_ws` process-wide per-W workspace cache -- on a restriction-aware
`LFixBaseCache` built exactly as origin-ZC builds its own (see
`build_lfix_base_cache_pairwise_quantile`, whose docstring records why that fold is mandatory).
"""

function pairwise_quantile_production_gradient(x_free0::AbstractVector, raw_masses::AbstractVector{Float64},
        pcx, ctx, pe; base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing,
        timers::Union{Nothing,PQGradTimers} = nothing, kwargs...)
    t_enter = time()
    if base === nothing || verify === nothing
        t0 = time()
        base, verify = archPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm)
        timers === nothing || (timers.t_solve += time() - t0)
    end
    # Both the q0 restriction fold and the closed-form mass gradient read pcx.ctx_cm.pq_mass_state,
    # which is mutable per-outer-point state some intervening solve may have moved -- see
    # ensure_pq_masses!.
    ensure_pq_masses!(pcx.ctx_cm, raw_masses)
    D = pcx.ctx_cm.D
    Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
    Wc = size(pcx.ctx_cm.obj.U, 1)
    ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(Wc) : econ_ws
    # Same workspace for the cache build and the gradient, so the W x D x Ddest tensors are
    # allocated once per process rather than once per gradient call.
    t0 = time()
    cache = build_lfix_base_cache_pairwise_quantile(x_free0, pcx.ctx_cm, base; verify = verify,
                                                    econ_ws = ws)
    timers === nothing || (timers.t_cache += time() - t0)
    g_econ = zeros(D * Ddest)
    t0 = time()
    meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    timers === nothing || (timers.t_econ += time() - t0)
    t0 = time()
    g_mass = pairwise_quantile_mass_gradient_vec(base, verify, pcx.ctx_cm, raw_masses)
    timers === nothing || (timers.t_mass += time() - t0)
    timers === nothing || (timers.t_total += time() - t_enter)
    return vcat(g_econ, g_mass), meta
end

# ------------------------------------------------------------------------------------------------
# Reoptimized-FD validation ground truth for the mass block.
# ------------------------------------------------------------------------------------------------

"""
    d_delta_dual_d_mass_fd(x_free0, raw_masses, ctx_cm; h=1e-4, coords=nothing, verbose=false)
        -> (g_fd, n_probed)

Reoptimized (NOT fixed-dual) central finite difference of `Delta_dual` w.r.t. the raw mass
coordinates -- the trusted ground truth for `pairwise_quantile_mass_gradient_vec`, and the exact
analog of `d_delta_dual_d_eta_origin_fd` (cm_originzc_production.jl) for this family. Each probe
RE-SOLVES the inner dual from scratch at the perturbed masses; nothing is held fixed and no
fixed-dual shortcut is taken, so this is genuinely independent of the machinery it validates.

A PLAIN SMALL `h` IS CORRECT HERE, and that is the whole point of version B. Version A needed
`matched_raw_steps`/`cutoff_probe_points` because `Delta_dual` was a step function of a cutoff: a
probe smaller than the gap to the next draw crossed nothing and returned exactly 0.0, so analytic
and FD had to be secants of the same staircase, and could only ever agree to ~20%. The masses do
not move any draw between bins -- they shift moment targets smoothly -- so `Delta_dual` is smooth in
them, `h=1e-4` is a genuine derivative estimate, and agreement should be ~1e-5 relative or better.
If it is not, something is wrong; do NOT reintroduce a bandwidth to make it look better.

`coords`: which raw coordinates to probe. Each probe is TWO full inner KNITRO solves, so probing all
`n_raw(layout)` of them is only affordable at D=4 scale; at real D=20 pass a subset. `nothing`
means all of them. The bin state cannot go stale under version B (fixed cutoffs), but the MASS state
can, so this function restores it to the caller's own point before returning.
"""
function d_delta_dual_d_mass_fd(x_free0::AbstractVector, raw_masses::AbstractVector{Float64}, ctx_cm;
        h::Float64 = 1e-4, coords::Union{Nothing,AbstractVector{Int}} = nothing, verbose::Bool = false)
    layout = ctx_cm.pq_layout
    n = n_raw(layout)
    length(raw_masses) == n ||
        error("d_delta_dual_d_mass_fd: length(raw_masses)=$(length(raw_masses)) != n_raw(layout)=$n")
    h > 0 || error("d_delta_dual_d_mass_fd: h must be > 0, got $h")
    idxs = coords === nothing ? collect(1:n) : collect(coords)
    g = fill(NaN, n)
    n_probed = 0
    for j in idxs
        1 <= j <= n || error("d_delta_dual_d_mass_fd: coordinate $j out of range 1:$n")
        rp = copy(collect(raw_masses)); rp[j] += h
        rm = copy(collect(raw_masses)); rm[j] -= h
        _, vp = archPQ_verified_state(x_free0, rp, ctx_cm)
        _, vm = archPQ_verified_state(x_free0, rm, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
        n_probed += 1
        verbose && println("  [fd] coord $j: h=", h, "  Delta(+)=", vp.Delta_dual, "  Delta(-)=", vm.Delta_dual,
                           "  fd=", g[j])
        flush(stdout)
    end
    # The mass state is left at whatever the LAST probe wrote; restore it to the caller's own point
    # so this function has no side effect on a live outer loop that calls it mid-run.
    ensure_pq_masses!(ctx_cm, raw_masses)
    return (g, n_probed)
end

"""
    d_delta_dual_d_econ_fd(w_econ0, raw_masses, ctx_cm, pe; coords, h=1e-5) -> Vector{Float64}

Reoptimized central FD of `Delta_dual` w.r.t. the ECONOMIC OUTER coordinates
`w_econ = (gp, zfree)`, at fixed masses -- the ground-truth counterpart for the `g_econ` half of the
combined gradient.

`w_econ`, NOT `x_free`: `economic_A_gradient!` returns its gradient already in this coordinate
system (index 1 = d/dgp, indices 2:end = d/dzfree in pivot-reduced z-space, gravity-pivot chain rule
applied internally -- see its own docstring). Probing `x_free` instead and comparing to `g_econ`
would compare two different coordinate systems and manufacture a disagreement out of nothing, so
this helper takes the same `w_econ` the driver's callbacks receive and maps it forward with the same
`vcat(gp, vec(exp.(pivot_expand(zfree, pe))))` the driver uses.

The one genuine non-smoothness on this side -- winner reassignment in the A_od block -- is exactly
what `economic_A_gradient!`'s own `select_bandwidth`/`count_winner_flips` machinery already handles
internally, and is NOT what this helper gates; this helper exists to confirm the shared economic
block is UNAFFECTED by this restriction being active, not to re-derive that block's own
long-validated bandwidth logic. Do not FD that block at small `h` and treat disagreement as a bug
(`select_bandwidth`'s own docstring: a sub-`h_floor=1e-4` probe reproduces a "known-wrong" gradient).
"""
function d_delta_dual_d_econ_fd(w_econ0::AbstractVector{Float64}, raw_masses::AbstractVector{Float64},
        ctx_cm, pe; coords::AbstractVector{Int}, h::Float64 = 1e-5, verbose::Bool = false)
    w0 = collect(w_econ0)
    xfw(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    g = fill(NaN, length(w0))
    for j in coords
        1 <= j <= length(w0) || error("d_delta_dual_d_econ_fd: coordinate $j out of range 1:$(length(w0))")
        wp = copy(w0); wp[j] += h
        wm = copy(w0); wm[j] -= h
        _, vp = archPQ_verified_state(xfw(wp), raw_masses, ctx_cm)
        _, vm = archPQ_verified_state(xfw(wm), raw_masses, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
        verbose && println("  [fd-econ] coord $j: h=", h, "  fd=", g[j])
        flush(stdout)
    end
    return g
end
