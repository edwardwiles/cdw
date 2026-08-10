# ================================================================================================
# OUTER-loop production layer for the pairwise-quantile-independence restriction (draft eq. 32).
#
# Everything in pairwise_quantile_production.jl is the INNER solve: given a FIXED outer point
# (theta, cutoffs), run the KNITRO inner dual solve. This file adds the three things an outer loop
# needs on top of that, mirroring cm_originzc_production.jl's own structure function-for-function:
#
#   1. archPQ_verified_state          <- archOZ_verified_state
#      A verified inner solve returning (base::BaseDualState, verify::NamedTuple). `verify` carries
#      `Delta_dual` -- the scalar the outer KNITRO callback actually consumes, which the inner-only
#      layer never produced. Built by handing this restriction's own verifier's output to the SAME
#      family-agnostic `verify_namedtuple_from_operator` (operator_verification.jl) every other
#      family uses, so `classify_inner_result`/`is_verified_success` (oracle.jl) accept it unchanged.
#
#   2. pairwise_quantile_production_gradient  <- cm_originzc_production_gradient
#      The combined outer gradient, `vcat(g_econ, g_cut)`:
#        - `g_econ`: the shared, family-agnostic (g, A_od) economic block, via `economic_A_gradient!`
#          (shared_a_gradient.jl). NOT reimplemented here -- origin-ZC's own header states this block
#          is "computed EXACTLY as in the CM-only path" and nothing in it knows a restriction exists.
#        - `g_cut`:  d(Delta_dual)/d(raw cutoff coordinate), via this restriction's own
#          `cutoff_secant_gradient!` (pairwise_quantile_cutoff_gradient.jl) evaluated at the REAL
#          converged (lambda_M, lambda_P, r) of the inner solve.
#
#   3. d_delta_dual_d_cutoff_fd       <- d_delta_dual_d_eta_origin_fd
#      Reoptimized (NOT fixed-dual) finite-difference ground truth for the cutoff block: every probe
#      re-solves the inner dual from scratch. This is the standard this codebase already holds
#      origin-ZC's own nu-gradient to; see its own docstring for the one way this gate must differ.
#
# SIGN CONVENTION -- the single most error-prone point in this file, stated once here and asserted
# by the FD gate. The inner KNITRO solve MINIMIZES `f = mean(Psi(r)) + zeta`, and the reported
# divergence is `Delta_dual = -f` (verify_namedtuple_from_operator, operator_verification.jl:512).
# `fixed_dual_delta_f` (pairwise_quantile_cutoff_gradient.jl) returns a change in `f`. Therefore
#     d(Delta_dual)/d(raw)  =  -1 * cutoff_secant_gradient!'s output,
# and the negation lives in exactly ONE place (`pairwise_quantile_cutoff_gradient_vec` below), never
# duplicated at a call site.
#
# WHY NOT A ZC-STYLE CLOSED FORM (do not "simplify" this back): origin-ZC's `nu` gradient is an
# exact closed-form envelope derivative because `nu` shifts a moment TARGET smoothly and never
# reassigns a draw between bins. This restriction's cutoffs are quantile-bin BOUNDARIES: moving one
# does nothing at all until it crosses an actual draw, at which point that draw's bin membership --
# and every moment row it participates in -- jumps. Delta_dual is a genuine step function of any
# single cutoff, so a closed-form envelope derivative is undefined at the jumps and identically zero
# between them. The correct precedent in this codebase is the economic A_od block's own
# boundary-crossing machinery (select_bandwidth/count_winner_flips, composite_gradient.jl:73-328);
# `cutoff_secant_gradient!` is this restriction's own (cheaper, exact-per-crossing) implementation of
# that same pattern. See docs/PAIRWISE_QUANTILE_OUTER_LOOP_INTEGRATION_HANDOVER_2026-08-10.md.
#
# Requires pairwise_quantile_production.jl (and its own include chain), shared_a_gradient.jl,
# operator_verification.jl, and oracle.jl to already be included.
# ================================================================================================

using LinearAlgebra: BLAS, dot, norm

isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))
isdefined(Main, :verify_namedtuple_from_operator) || include(joinpath(@__DIR__, "operator_verification.jl"))

"""
    build_pairwise_quantile_production_context(ctx, layout; min_crossed) -> (ctx_cm, aug, hess_ctx, layout, min_crossed)

Analog of `build_originzc_production_context`: assembles the restriction-augmented context ONCE per
run. Returns a NamedTuple with a `.ctx_cm` field, which is what `prepare_production_run`'s
`_resolve_bundle` (production_bundle_api.jl:127-133) looks for -- so a driver wraps this call in
`prepare_production_run(:pairwise_quantile, "<runner name>", () -> build_..._context(...))` and gets
the OperatorPsiBundle invariant asserted for free, exactly like every other family.

`ctx` MUST be the plain, unaugmented economic context (`d20_real_setup_design`/`d4_exact_setup`).
It is carried forward on `ctx_cm.pq_econ_ctx` because `archPQ_base_state`/`archPQ_verified_state`
genuinely need BOTH: `cf_build`/`prime_operator!` read dimensionality off the ECONOMIC-only bundle,
and passing the augmented context there corrupts `cf.oci` (a real bug caught live on 2026-08-09,
documented in `archPQ_base_state`'s own docstring -- origin-ZC keeps the same thing on
`octx.econ_ctx` for exactly this reason).

`min_crossed` and `L` (via `layout`) have NO defaults -- both are genuine modelling choices under
this repo's no-silent-defaults rule (CLAUDE.md), and `min_crossed` in particular must be chosen
against the campaign's own `W` (it sets the secant bandwidth of the cutoff gradient).
"""
function build_pairwise_quantile_production_context(ctx, layout::PairwiseQuantileCutoffLayout;
        min_crossed::Int)
    ctx.D == layout.D ||
        error("build_pairwise_quantile_production_context: ctx.D=$(ctx.D) != layout.D=$(layout.D)")
    min_crossed >= 1 ||
        error("build_pairwise_quantile_production_context: min_crossed must be >= 1, got $min_crossed")
    W = size(ctx.U, 1)
    min_crossed < W ||
        error("build_pairwise_quantile_production_context: min_crossed=$min_crossed must be < W=$W " *
              "(a bandwidth that crosses every draw is not a local secant)")
    println(stdout, "cm_restriction_basis [pairwise_quantile] = quantile bins (L=", layout.L,
            "), marginal + pairwise-independence rows; n_total_rows=", n_total_rows(ctx.D, layout.L))
    println(stdout, "pairwise_quantile outer cutoff coordinates: n_raw=", n_raw(layout),
            " (origin-major, softplus-ordered), min_crossed=", min_crossed, " of W=", W)
    flush(stdout)

    aug = build_pairwise_quantile_augmented_obj(ctx, layout)
    bin_state = PairwiseQuantileBinState(W, ctx.D, layout.L)
    hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, bin_state, aug.core_cf_ref)
    ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_bin_state = bin_state,
                          pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx,
                          pq_econ_ctx = ctx, pq_layout = layout, pq_min_crossed = min_crossed))
    return (ctx_cm = ctx_cm, aug = aug, hess_ctx = hess_ctx, layout = layout, min_crossed = min_crossed)
end

"""
    archPQ_verified_state(x_free0, raw_cutoffs, ctx_cm; dual_bank=nothing, eval_id=0) -> (base, verify)

Verified analog of `archPQ_base_state`, mirroring `archOZ_verified_state` (cm_originzc_production.jl)
exactly: solve, then INDEPENDENTLY recompute the solution's residual/objective/KKT blocks with this
restriction's own verifier (`verify_inner_solution_operator_pairwisequantile!`, never reading any
FG-callback-cached state), then hand that to the shared `verify_namedtuple_from_operator` so the
returned `verify` carries precisely the field set `classify_inner_result`/`is_verified_success`
(oracle.jl) already know how to read -- no new acceptance predicate is invented for this family.

`verify` additionally carries, beyond the shared field set:
  - `r_current`: the converged per-draw `R_w` from the INDEPENDENT verifier recompute. This is what
    the cutoff outer gradient consumes; taking it from the verifier (not from the FG callback's
    cached `obj.arg0`) keeps the gradient's input on the same independently-recomputed footing as
    the value it differentiates.
  - the restriction's own block KKT residuals and probability/cumulative-residual tables, for
    campaign logging.

Throws `CMExpectedSolveFailure` (reused, not redefined) on an infeasible/failed inner solve, exactly
as every other family's `*_verified_state` does, so the driver's own `cb_F!` can `reject_point`.
"""
function archPQ_verified_state(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64}, ctx_cm;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    obj = ctx_cm.obj
    layout = ctx_cm.pq_layout
    econ_ctx = ctx_cm.pq_econ_ctx
    op = ctx_cm.pq_op
    length(raw_cutoffs) == n_raw(layout) ||
        error("archPQ_verified_state: length(raw_cutoffs)=$(length(raw_cutoffs)) != n_raw(layout)=$(n_raw(layout))")

    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, vcat(collect(x_free0), collect(raw_cutoffs)))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end

    nStatus, inner_x, _, n_fg, n_hess = archPQ_base_state(x_free0, raw_cutoffs, econ_ctx, ctx_cm, layout)
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archPQ_verified_state: inner solve failed, nStatus=$nStatus " *
                                     "(x_free0=$x_free0, raw_cutoffs=$raw_cutoffs)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = op.W
    ncore1 = obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)

    cf = ctx_cm.pq_core_cf_ref[]
    cf isa CompressedFactual ||
        error("archPQ_verified_state: pq_core_cf_ref[] is not a CompressedFactual -- prime_operator! " *
              "did not run for this outer point (got $(typeof(cf)))")
    econ_ws = economic_operator_workspace(cf)
    ov = verify_inner_solution_operator_pairwisequantile!(ζstar, λstar, cf, op, ctx_cm.pq_bin_state, W,
        economic_forward!, economic_transpose!, econ_ws, obj.Psi!, obj.dPsi!, ncore1)
    m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    verify = merge(verify, (r_current = ov.r,
                            kkt_resid_E = ov.kkt_resid_E,
                            kkt_resid_marginalbin = ov.kkt_resid_marginalbin,
                            kkt_resid_pairindep = ov.kkt_resid_pairindep,
                            max_cumulative_residual = ov.max_cumulative_residual,
                            max_marginal_cumulative_residual = ov.max_marginal_cumulative_residual,
                            marginal_prob = ov.marginal_prob,
                            n_fg = n_fg, n_hess = n_hess))

    base = BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, vcat(collect(x_free0), collect(raw_cutoffs)), inner_x)
    return base, verify
end

"pairwise_quantile_production_value_verified(x_free0, raw_cutoffs, pcx) -> (K, base, verify). Analog of `cm_originzc_production_value_verified`; `K` is `obj.H_save`, the same payoff scalar every family reports."
function pairwise_quantile_production_value_verified(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64}, pcx;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    base, verify = archPQ_verified_state(x_free0, raw_cutoffs, pcx.ctx_cm; dual_bank = dual_bank, eval_id = eval_id)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    pairwise_quantile_production_value_verified_screened(x_free0, raw_cutoffs, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Screened drop-in, mirroring `cm_originzc_production_value_verified_screened` (cm_screen_bridge.jl)
verbatim: the SAME family-agnostic `cm_screen_precheck!` first, then this family's verified state.
"""
function pairwise_quantile_production_value_verified_screened(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64}, pcx;
        counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false,
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    cm_screen_precheck!(x_free0, pcx.ctx_cm; counters = counters, use_witness = use_witness)
    return pairwise_quantile_production_value_verified(x_free0, raw_cutoffs, pcx; dual_bank = dual_bank, eval_id = eval_id)
end

"""
    ensure_pq_bins!(ctx_cm, raw_cutoffs) -> nothing

Refresh `ctx_cm.pq_bin_state` so it corresponds to `raw_cutoffs`, before any OUTER-loop consumer
reads it.

`pq_bin_state` is deliberately MUTABLE per-outer-point state, rebuilt once per inner solve by
`reset_for_solve!` (pairwise_quantile_production.jl:81) -- the restriction's own "decode once per
outer point, never inside a callback" requirement. That is correct for the inner solve, but it makes
every OUTER-loop consumer (`cutoff_secant_gradient!`, the `q0` restriction fold) implicitly dependent
on ambient state that some LATER solve may since have overwritten. In a checkpointed outer driver
that is a real hazard, not a hypothetical: `cb_G!` may run against a cached `base` from an earlier
`cb_F!`, a verification re-solve or a diagnostic FD probe can land in between, and nothing about the
resulting gradient would look wrong -- it would just silently be computed against a different point's
bin assignment.

Found live 2026-08-10 while building this layer: the outer-gradient FD gate ran a sequence of probes
at other cutoff points and then computed a gradient at the original point, and the `q0` cross-check
in `build_lfix_base_cache_pairwise_quantile` caught the resulting mismatch (max|diff| = 0.62 against
an independently recomputed `r`). So this refresh is called unconditionally on the outer-gradient
entry path, and the `q0` cross-check is retained behind it as the backstop that would catch any
remaining inconsistency (e.g. a `base` that came from a genuinely different outer point, which no
amount of bin refreshing can repair).

Cost is `O(W*D)` `searchsortedfirst` calls -- negligible beside the inner KNITRO solve this sits
next to, and not worth trading for a staleness guess.
"""
function ensure_pq_bins!(ctx_cm, raw_cutoffs::AbstractVector{Float64})
    refresh_pairwise_quantile_bins!(ctx_cm.pq_bin_state, ctx_cm.pq_op, ctx_cm.U, raw_cutoffs, ctx_cm.pq_layout)
    return nothing
end

"""
    reshape_pq_duals(lambda, op, ncore1) -> (lambda_M, lambda_P)

The ONE place the flat inner-dual vector is split into this restriction's marginal/pair dual blocks
for OUTER-loop use. Uses the identical `reshape(v, nc, D)'` / `reshape(v, nc, nc, npair)` convention
as `dual_index!` (pairwise_quantile_production.jl:106-107) and the verifier
(pairwise_quantile_verification.jl:58-59) -- see `dual_index!`'s own comment for why the marginal
block needs the transpose (`marginal_row(o,a,L)` is O-MAJOR, Julia's `reshape` is A-MAJOR) and the
pair block does not. Getting this wrong swaps dual columns silently; it was already a real bug once
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
    pairwise_quantile_cutoff_gradient_vec(base, verify, ctx_cm, raw_cutoffs; min_crossed) -> Vector{Float64}

`d(Delta_dual)/d(raw_k)` for every one of the `n_raw(layout)` raw cutoff coordinates, evaluated at
the REAL converged inner dual carried by `base`/`verify`.

This is the ONLY place the sign flip described in this file's header is applied:
`cutoff_secant_gradient!` differentiates the inner objective `f`, and `Delta_dual = -f`, so the
returned vector is its negation. Every caller (production gradient, FD gate, driver) consumes
d(Delta_dual)/d(raw) and must never negate again.

`min_crossed` is passed through, not defaulted -- it sets the secant bandwidth and is a genuine
per-campaign choice (see `build_pairwise_quantile_production_context`).
"""
function pairwise_quantile_cutoff_gradient_vec(base::BaseDualState, verify, ctx_cm,
        raw_cutoffs::AbstractVector{Float64}; min_crossed::Int)
    ensure_pq_bins!(ctx_cm, raw_cutoffs)   # see ensure_pq_bins!: pq_bin_state is mutable ambient state
    op = ctx_cm.pq_op
    layout = ctx_cm.pq_layout
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    λ_M, λ_P = reshape_pq_duals(base.λstar, op, ncore1)
    r_current = verify.r_current
    length(r_current) == op.W ||
        error("pairwise_quantile_cutoff_gradient_vec: length(verify.r_current)=$(length(r_current)) != W=$(op.W)")

    grad_raw = zeros(n_raw(layout))
    cutoff_secant_gradient!(grad_raw, op, ctx_cm.pq_bin_state, λ_M, λ_P, r_current, layout, raw_cutoffs;
        min_crossed = min_crossed)
    grad_raw .*= -1.0   # d(Delta_dual) = -d(f); see this file's header. Applied exactly once, here.
    return grad_raw
end

"""
    build_lfix_base_cache_pairwise_quantile(x_free0, ctx_cm, base; verify=nothing) -> LFixBaseCache

This family's analog of `build_lfix_base_cache_originzc` (cm_originzc_production.jl:214-219) and
`build_lfix_base_cache_cm_meanzc`: build the plain economic `LFixBaseCache`, then fold THIS
restriction's own `G_R*lambda_R` contribution into `cache.q0` via the shared `with_q0` mechanism
(lfix_cm_aware.jl:47).

**This fold is required, and reasoning that it is not is a live trap this task fell into once.**
The tempting argument -- "the restriction's bin memberships depend only on `U` and the cutoffs,
never on theta, so the restriction contributes nothing to the theta-gradient" -- is true about the
DERIVATIVE and irrelevant to `q0`. `q0` is not a derivative: it is the per-draw LEVEL
`q0[s] = -zeta* - sum_j lambda*_j G[s,j]` that the economic block linearizes AROUND
(`build_lfix_base_cache`, lfix_incremental.jl:56, computes it from the ECONOMIC moment columns
only). Omitting the restriction's own `G_R*lambda_R` term linearizes the economic gradient about the
wrong base point, producing an economic gradient block that is wrong by an amount that has nothing
to do with the restriction's own gradient. Caught live by
`test_pairwise_quantile_outer_gradient_fd.jl`'s section 3 (two of four probed economic coordinates
came out ~70x off, with correct-looking values on the other two).

The contribution is obtained from `pairwise_quantile_forward!` itself -- the SAME operator the inner
solve uses -- rather than from a hand-written second copy of the moment algebra: starting from
zeros it accumulates `-G_R*lambda_R` (it SUBTRACTS into its accumulator, see its own `arg0[w] -= Rw`),
which is exactly the term `q0` is missing, so it is simply added.

`verify`: when supplied, its independently recomputed `r_current` is used as an exact cross-check --
the corrected `q0` must equal it to floating-point tolerance, since both are the same quantity
`r = -zeta - E*lambda_E - G_R*lambda_R` computed by two different routes (closed-form economic cache
+ operator fold here; full independent operator recompute there). A mismatch is a hard error, not a
warning: it means the fold or the cache is wrong, and every economic gradient built on it would be
silently wrong.
"""
function build_lfix_base_cache_pairwise_quantile(x_free0::AbstractVector, ctx_cm, base::BaseDualState;
        verify = nothing, q0_check_tol::Float64 = 1e-8)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = false)
    op = ctx_cm.pq_op
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    λ_M, λ_P = reshape_pq_duals(base.λstar, op, ncore1)
    contrib = zeros(op.W)
    pairwise_quantile_forward!(contrib, λ_M, λ_P, op, ctx_cm.pq_bin_state)   # contrib = -G_R*lambda_R
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
    pairwise_quantile_production_gradient(x_free0, raw_cutoffs, pcx, ctx, pe;
                                          base=nothing, verify=nothing, econ_ws=nothing,
                                          min_crossed, kwargs...) -> (g_ext, meta)

Full outer gradient for this family, structurally identical to `cm_originzc_production_gradient`:
the shared (g, A_od) economic block, then this restriction's own block appended --
`g_ext = vcat(g_econ, g_cut)`, length `D*Ddest + n_raw(layout)`.

The economic block is NOT reimplemented and NOT modified for this family: it is the same
`economic_A_gradient!` (shared_a_gradient.jl) call origin-ZC makes, with the same
`get_or_build_econ_a_grad_ws` process-wide per-W workspace cache -- on a restriction-aware
`LFixBaseCache` built exactly as origin-ZC builds its own (see
`build_lfix_base_cache_pairwise_quantile`, whose docstring records why that fold is mandatory).
"""
function pairwise_quantile_production_gradient(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64},
        pcx, ctx, pe; base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, min_crossed::Int, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archPQ_verified_state(x_free0, raw_cutoffs, pcx.ctx_cm)
    end
    # Both the q0 restriction fold and the cutoff secant read pcx.ctx_cm.pq_bin_state, which is
    # mutable per-outer-point state some intervening solve may have moved -- see ensure_pq_bins!.
    ensure_pq_bins!(pcx.ctx_cm, raw_cutoffs)
    cache = build_lfix_base_cache_pairwise_quantile(x_free0, pcx.ctx_cm, base; verify = verify)
    D = pcx.ctx_cm.D
    Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
    ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
    g_econ = zeros(D * Ddest)
    meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
    g_cut = pairwise_quantile_cutoff_gradient_vec(base, verify, pcx.ctx_cm, raw_cutoffs; min_crossed = min_crossed)
    return vcat(g_econ, g_cut), meta
end

# ------------------------------------------------------------------------------------------------
# Reoptimized-FD validation ground truth for the cutoff block.
# ------------------------------------------------------------------------------------------------

"""
    matched_raw_steps(ctx_cm, raw_cutoffs; min_crossed) -> Vector{Float64}

The per-raw-coordinate FD step `h_k` that `d_delta_dual_d_cutoff_fd` should use, derived from the
SAME `cutoff_probe_points` the analytic secant uses (pairwise_quantile_cutoff_gradient.jl).

Why this is not simply `h = 1e-4`, and why using a fixed small `h` here would be a methodological
error rather than a conservative choice: `Delta_dual` is a genuine STEP function of a cutoff. A
probe that moves `q` by less than the gap to the next draw crosses zero draws, changes nothing, and
yields an FD of exactly 0.0 for every coordinate -- which would "disagree" with any correct analytic
gradient and look exactly like a gradient bug. The meaningful comparison is between two secants of
the same staircase taken over the same window. So for raw coordinate `k` of origin `o` we pick

    h_k = (q_up_{o,k} - q_{o,k}) / J[k,k]

i.e. the raw step whose induced movement of that origin's `k`-th physical cutoff equals the up-probe
the analytic secant itself selected (`J[k,k] = dq_k/draw_k`, `cutoff_jacobian_block!`). Because the
Jacobian is lower-triangular, this same step also moves cutoffs `r > k` -- which is correct and
intended: `h_k` is a step in the coordinate KNITRO actually moves, so the reoptimized FD it produces
is the ground truth for the full chain-ruled `grad_raw[k]`, not for one isolated `dq_r`.

Returns `0.0` for any coordinate with no room to probe (a degenerate cutoff with no draws above it);
callers skip those rather than dividing by zero.
"""
function matched_raw_steps(ctx_cm, raw_cutoffs::AbstractVector{Float64}; min_crossed::Int)
    op = ctx_cm.pq_op
    layout = ctx_cm.pq_layout
    state = ctx_cm.pq_bin_state
    D = layout.D; nc = n_cutoffs(layout)
    h = zeros(n_raw(layout))
    J = zeros(nc, nc)
    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        rawo = @view raw_cutoffs[base:base+nc-1]
        Qcol = @view state.Q[:, o]
        cutoff_jacobian_block!(J, rawo, Qcol)
        for k in 1:nc
            (q_up, _) = cutoff_probe_points(op, Qcol, o, k, nc; min_crossed = min_crossed)
            dq = q_up - Qcol[k]
            h[raw_index(layout, o, k)] = (dq > 0 && J[k, k] > 0) ? dq / J[k, k] : 0.0
        end
    end
    return h
end

"""
    d_delta_dual_d_cutoff_fd(x_free0, raw_cutoffs, ctx_cm; min_crossed, coords=nothing, verbose=false)
        -> (g_fd, h_used, n_probed)

Reoptimized (NOT fixed-dual) central finite difference of `Delta_dual` w.r.t. the raw cutoff
coordinates -- the trusted ground truth for `pairwise_quantile_cutoff_gradient_vec`, and the exact
analog of `d_delta_dual_d_eta_origin_fd` (cm_originzc_production.jl:318-330) for this family. Each
probe RE-SOLVES the inner dual from scratch at the perturbed cutoffs; nothing is held fixed and no
fixed-dual shortcut is taken, so this is genuinely independent of the machinery it validates (which
shares neither the solve, the bin refresh, nor the Psi evaluation path with it).

Step sizes come from `matched_raw_steps` (see its docstring for why a fixed `h` would be wrong here
rather than merely conservative).

`coords`: which raw coordinates to probe. Each probe is TWO full inner KNITRO solves, so probing all
`n_raw(layout)` of them is only affordable at D=4 scale; at real D=20 pass a subset. `nothing`
means all of them. Coordinates whose matched step is 0.0 (no room to probe) are skipped and reported
in `n_probed`.
"""
function d_delta_dual_d_cutoff_fd(x_free0::AbstractVector, raw_cutoffs::AbstractVector{Float64}, ctx_cm;
        min_crossed::Int, coords::Union{Nothing,AbstractVector{Int}} = nothing, verbose::Bool = false)
    layout = ctx_cm.pq_layout
    n = n_raw(layout)
    h_all = matched_raw_steps(ctx_cm, raw_cutoffs; min_crossed = min_crossed)
    idxs = coords === nothing ? collect(1:n) : collect(coords)
    g = fill(NaN, n)
    n_probed = 0
    for j in idxs
        1 <= j <= n || error("d_delta_dual_d_cutoff_fd: coordinate $j out of range 1:$n")
        h = h_all[j]
        if !(h > 0)
            verbose && println("  [fd] coord $j: no room to probe (h=0), skipped")
            continue
        end
        rp = copy(collect(raw_cutoffs)); rp[j] += h
        rm = copy(collect(raw_cutoffs)); rm[j] -= h
        _, vp = archPQ_verified_state(x_free0, rp, ctx_cm)
        _, vm = archPQ_verified_state(x_free0, rm, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
        n_probed += 1
        verbose && println("  [fd] coord $j: h=", h, "  Delta(+)=", vp.Delta_dual, "  Delta(-)=", vm.Delta_dual,
                           "  fd=", g[j])
        flush(stdout)
    end
    # The bin state is left at whatever the LAST probe wrote; restore it to the caller's own point so
    # this function has no side effect on a live outer loop that calls it mid-run for diagnostics.
    refresh_pairwise_quantile_bins!(ctx_cm.pq_bin_state, ctx_cm.pq_op, ctx_cm.U, raw_cutoffs, layout)
    return (g, h_all, n_probed)
end

"""
    d_delta_dual_d_econ_fd(w_econ0, raw_cutoffs, ctx_cm, pe; coords, h=1e-5) -> Vector{Float64}

Reoptimized central FD of `Delta_dual` w.r.t. the ECONOMIC OUTER coordinates
`w_econ = (gp, zfree)`, at fixed cutoffs -- the ground-truth counterpart for the `g_econ` half of
the combined gradient.

`w_econ`, NOT `x_free`: `economic_A_gradient!` returns its gradient already in this coordinate
system (index 1 = d/dgp, indices 2:end = d/dzfree in pivot-reduced z-space, gravity-pivot chain rule
applied internally -- see its own docstring). Probing `x_free` instead and comparing to `g_econ`
would compare two different coordinate systems and manufacture a disagreement out of nothing, so
this helper takes the same `w_econ` the driver's callbacks receive and maps it forward with the same
`vcat(gp, vec(exp.(pivot_expand(zfree, pe))))` the driver uses.

Unlike the cutoff block, a plain small fixed `h` IS the right choice here: theta enters through the
economic moment rows smoothly, so `Delta_dual` is (piecewise) smooth in `w_econ` and an ordinary
central FD converges. The one genuine non-smoothness on this side -- winner reassignment in the
A_od block -- is exactly what `economic_A_gradient!`'s own `select_bandwidth`/`count_winner_flips`
machinery already handles internally, and is not what this helper gates; this helper exists to
confirm the shared economic block is UNAFFECTED by this restriction being active, not to re-derive
that block's own long-validated bandwidth logic.
"""
function d_delta_dual_d_econ_fd(w_econ0::AbstractVector{Float64}, raw_cutoffs::AbstractVector{Float64},
        ctx_cm, pe; coords::AbstractVector{Int}, h::Float64 = 1e-5, verbose::Bool = false)
    w0 = collect(w_econ0)
    xfw(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    g = fill(NaN, length(w0))
    for j in coords
        1 <= j <= length(w0) || error("d_delta_dual_d_econ_fd: coordinate $j out of range 1:$(length(w0))")
        wp = copy(w0); wp[j] += h
        wm = copy(w0); wm[j] -= h
        _, vp = archPQ_verified_state(xfw(wp), raw_cutoffs, ctx_cm)
        _, vm = archPQ_verified_state(xfw(wm), raw_cutoffs, ctx_cm)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
        verbose && println("  [fd-econ] coord $j: h=", h, "  fd=", g[j])
        flush(stdout)
    end
    return g
end
