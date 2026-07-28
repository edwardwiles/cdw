isdefined(Main, :CMLookupState) || include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
isdefined(Main, :_callbackEvalFG_inner_cmlookup!) || include(joinpath(@__DIR__, "cm_lookup_live_knitro.jl"))
isdefined(Main, :inner_loop_internal_cmlookup_production) || include(joinpath(@__DIR__, "cm_lookup_production.jl"))
isdefined(Main, :DualBank) || include(joinpath(@__DIR__, "dual_bank.jl"))
isdefined(Main, :RestrictedDualBank) || include(joinpath(@__DIR__, "cm_dual_bank_production.jl"))   # Phase D remediation (2026-07-26)
isdefined(Main, :cf_build) || include(joinpath(@__DIR__, "compressed_factual_buffer_reuse.jl"))   # Phase E remediation (2026-07-26)
isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))   # shared-FG-verification-and-A-gradient release (2026-07-27): flexible-CM's DEFAULT (g,A_od)-block gradient backend, see cm_production_gradient below
isdefined(Main, :verify_inner_solution_operator_cm!) || include(joinpath(@__DIR__, "operator_verification.jl"))   # verification-defaults task (2026-07-27): archC_verified_state's :operator backend below
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))   # default-flips task (2026-07-27), Task C: MOMENT_REPRESENTATION selector used by archC_base_state below

# ============================================================================
# Continuation 13, Sections 3A + 5: production combined bundle.
#
# Ties together, additively, every already-validated Continuation 12/13 piece
# into ONE inner-solve + gradient path suitable for an outer KNITRO loop:
#   - cumulative basis (build_cm_augmented_obj), nested grids (probs=)
#   - Architecture B moment construction (cached G_tmp, no persistent dense
#     CM copy per call -- cm_hessian_architectures.jl, already built)
#   - Architecture C structured Hessian for the INNER dual solve
#     (cm_hessian_architectures.jl, already built) -- this is what Section
#     9's "wire Architecture C into an actual KNITRO OUTER loop" needed:
#     every inner solve triggered during outer optimization (not just a
#     one-off script) now goes through Architecture C.
#   - the CM-aware Lfix outer gradient (lfix_cm_aware.jl)
# No existing function is modified; this file only composes.
# ============================================================================

"""
    CMExpectedSolveFailure <: Exception

Closure task Phase 3B: `archC_base_state`/`archC_verified_state`'s inner-solve failure
(`nStatus` outside `(0,-100,-101,-103)`, i.e. a genuinely infeasible/unbounded/failed KNITRO
inner dual solve -- the ONE documented, expected failure mode every caller in this file already
guards against) previously raised via a bare `error(...)`, i.e. a plain `ErrorException`. Every
`cb_F!`/`run_cm_upper` catch site then did `e isa ErrorException || rethrow()` -- correct in
intent, but `ErrorException` is also what an ordinary programming bug (`error("oops, forgot a
case")`, a typo'd `@assert`, etc.) raises, so that catch could silently swallow a genuine
invariant violation as "reject this point" instead of aborting visibly.

Use this dedicated type for the ONE expected failure class instead: every catch site now does
`e isa CMExpectedSolveFailure || rethrow()`, so any *other* exception (bare `ErrorException`,
`MethodError`, `BoundsError`, a task failure, ...) propagates instead of being silently
reinterpreted as an infeasible point. See `test_cm_expected_solve_failure_typed.jl`.
"""
struct CMExpectedSolveFailure <: Exception
    msg::String
end
Base.showerror(io::IO, e::CMExpectedSolveFailure) = print(io, "CMExpectedSolveFailure: ", e.msg)

"""
    build_cm_production_context(ctx, CS; L, contrasts=:anchored, probs=nothing, use_archB_moments=true)
        -> (ctx_cm, aug, bins, cctx)

`ctx_cm.obj` uses Architecture B's moment construction when
`use_archB_moments=true` (default -- avoids `wrap_moments_with_cm`'s
fresh-`similar` `G_tmp` per call and never stores/copies from a persistent
dense `W x ncm` CM matrix beyond the one built once here for the dense
REFERENCE `aug.CM`/verification; per-call construction goes straight from
bin indices via `fill_cm_columns_from_bins!`). `cctx` is the Architecture-C
bin/scratch context (`build_cm_bin_ctx`), reused for every subsequent inner
solve at this context (bin indices are fixed once the draws U are fixed --
independent of theta).
"""
function build_cm_production_context(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                      probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                      use_archB_moments::Bool = true,
                                      use_compressed_core::Bool = true,   # allocation/Hessian port task §5:
                                      # compressed winner-form core moments (default) vs the original
                                      # dense EK_moments_gammanorm_directgp! path (false, kept for
                                      # correctness comparison/emergency revert) -- see
                                      # wrap_moments_with_cm_archB's own docstring.
                                      threaded_bins::Bool = true,   # allocation/Hessian port task §6:
                                      # pass-through to build_cm_bin_ctx -- true (production default,
                                      # matches build_cm_bin_ctx's own default) selects the threaded
                                      # Architecture-C Hessian; false is an explicit opt-out/benchmark-
                                      # only comparison against the original serial implementation.
                                      inner_fg_backend::Symbol = CM_INNER_FG_BACKEND_DEFAULT[],   # Phase B1
                                      # remediation (2026-07-26): pass-through to build_cm_bin_ctx --
                                      # :dense_reference (default, unchanged) | :cm_lookup (plain
                                      # flexible CM only, see cm_lookup_production.jl).
                                      cm_cross_hessian_backend::Symbol = CM_CROSS_HESSIAN_BACKEND_DEFAULT[])   # winner-aware
                                      # H_ER phase (2026-07-27): pass-through to build_cm_bin_ctx -- :dense_reference
                                      # (default, unchanged) | :winner_bin (winner_pair_cross_hessian.jl).
    isdefined(Main, :record_cm_feature_context_build!) && record_cm_feature_context_build!()   # Phase 3 (2026-07-26): CM feature immutability counters
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts, probs = probs)
    obj_cm = aug.obj_cm
    # 2026-07-25 continuation (task §2 runtime counters investigation): `build_cm_production_context`
    # -- the function `run_cm_upper_checkpointed` (the REAL production driver) actually calls --
    # wraps `wrap_moments_with_cm_archB` INLINE here rather than via `build_cm_augmented_obj_archB`
    # (a separate, parallel construction path). The 2026-07-25 port session threaded `core_cf_ref`
    # through `build_cm_augmented_obj_archB` but MISSED this inline call site entirely: without an
    # explicit `core_cf_ref` here, `wrap_moments_with_cm_archB` silently builds its OWN throwaway
    # default `Ref{Any}(nothing)`, DIFFERENT from the one `build_cm_bin_ctx` below would also
    # default to absent an explicit value -- two disconnected Refs, so `cctx.core_cf_ref[]` stayed
    # `nothing` forever and CM's Hessian callback silently fell back to dense BLAS on every call,
    # UNDETECTED until this session's new runtime backend-use counters (task §2) caught it (0
    # winner-pair calls AND 0 recorded dense-fallback calls on a real solve is impossible -- that
    # contradiction is what surfaced this). Confirmed via a direct D=4 repro:
    # `pcx.cctx.core_cf_ref[] === nothing` after a real feasible archC_base_state solve, before this
    # fix. Fixed by building ONE `core_cf_ref` here and threading it to BOTH call sites.
    core_cf_ref = Ref{Any}(nothing)
    # skip_cm_fill_ref removal (2026-07-27): archC_base_state/archC_verified_state used to toggle a
    # shared `Ref{Bool}` around each inner solve so wrap_moments_with_cm_archB's closure could skip
    # materializing the dense CM columns when they are about to go completely unread (:cm_lookup FG
    # backend AND no post-solve verified-state recompute). Replaced with two SEPARATE, immutable
    # closures built once here -- `moments_archB!` (skip_fill=false, always fills; installed as
    # ctx_cm.obj.moments!, used by every non-skip path AND by archC_verified_state's own inner
    # solve) and `moments_archB_skip!` (skip_fill=true, never fills; installed on `cctx.
    # moments_skip!`, used ONLY by inner_loop_internal_cmlookup_production's priming call when
    # archC_base_state explicitly threads skip_fill=true through). Both share the SAME `core_cf_ref`
    # box so the Hessian callback sees an identical publish regardless of which one ran. See
    # wrap_moments_with_cm_archB's own kwarg docstring for the full rationale.
    moments_archB_skip! = nothing
    if use_archB_moments
        # NOTE: common_marginals_interval.jl and cm_hessian_architectures.jl both define
        # `compute_bin_indices(U,z)` with overlapping-but-distinct signatures (z::Vector{Float64}
        # vs z::AbstractVector{Float64}) -- Julia's most-specific-method dispatch silently prefers
        # the FORMER (Unsigned-typed, for interval_forward_contribution!) regardless of include
        # order, which is NOT what fill_cm_columns_from_bins! below expects (Matrix{Int}). Force
        # the Int-typed variant explicitly rather than depend on ambient method resolution.
        Bidx = Int.(compute_bin_indices(ctx.U, aug.z))
        R = contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
        moments_archB! = wrap_moments_with_cm_archB(ctx.obj.moments!, aug.ncore, Bidx, aug.origins, aug.refIndex1, aug.L, R, ctx;
                                                     use_compressed_core = use_compressed_core, core_cf_ref = core_cf_ref,
                                                     skip_fill = false)
        moments_archB_skip! = wrap_moments_with_cm_archB(ctx.obj.moments!, aug.ncore, Bidx, aug.origins, aug.refIndex1, aug.L, R, ctx;
                                                     use_compressed_core = use_compressed_core, core_cf_ref = core_cf_ref,
                                                     skip_fill = true)
        obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj_cm.δ, find_smallest = obj_cm.find_smallest,
            γ = obj_cm.γ, (moments!) = moments_archB!, moments_jacobian! = error,
            d = obj_cm.d, outer_constr_index = obj_cm.outer_constr_index,
            inequality_index = obj_cm.inequality_index, complement_index = obj_cm.complement_index,
            l = obj_cm.l, U = obj_cm.U, N = obj_cm.N, lower_limit = obj_cm.lower_limit,
            use_cached_x = obj_cm.use_cached_x,
            threshold_state = obj_cm.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
            outer_loop_opt = obj_cm.outer_loop_opt, inner_loop_opt = obj_cm.inner_loop_opt,
            needs_outer_moment_jacobian = obj_cm.needs_outer_moment_jacobian)
    end
    ctx_cm = merge(ctx, (obj = obj_cm,))
    bins = cm_bin_indices_for(ctx, aug)
    aug = merge(aug, (core_cf_ref = core_cf_ref, moments_skip! = moments_archB_skip!))   # so build_cm_bin_ctx's
    # hasproperty(aug, :core_cf_ref)/hasproperty(aug, :moments_skip!) pick up the SAME core_cf_ref
    # box and skip-variant closure moments_archB!/moments_archB_skip! above use/build
    # inner_fg_backend=:cm_lookup is only ever reachable through THIS function (build_cm_production_context
    # is the plain flexible-CM builder -- common-Frechet and CM+meanZC each have their OWN separate
    # build_cm_frechet_production_context/build_cm_meanzc_production_context, neither of which
    # accepts this kwarg), so the "plain flexible CM only" scope restriction from
    # cm_lookup_production.jl's header is structural here, not enforced by an extra runtime check.
    cctx = build_cm_bin_ctx(ctx, aug; threaded_bins = threaded_bins, inner_fg_backend = inner_fg_backend,
        cm_cross_hessian_backend = cm_cross_hessian_backend)
    return (ctx_cm = ctx_cm, aug = aug, bins = bins, cctx = cctx)
end

"""
    delta_dual_from_base(obj, base::BaseDualState) -> Float64

Canonical `Delta_dual = -(mean(Psi(q*)) + zeta*)` (== `cbuf[1]/1e10`, oracle.jl's own
convention -- see `evaluate_fullA`/`three_way_derivatives.jl`'s `fixed_dual_L`), recomputed at
the SAME converged `(zeta*, lambda*)` `base` already holds, via one explicit
`obj(inner_x, constr=...)` call (matches `archC_verified_state`'s own recompute pattern below --
does NOT trust `obj.arg1`/KN_solve's last FG callback as-is, an independent check should not rely
on the same assumption it exists to catch a violation of).

Use this, NOT `-base.ζstar`, anywhere a caller needs the divergence value from an
already-solved `BaseDualState` -- see remediation task Part A / finding F1: `-zeta*` silently
omits `mean(Psi(q*))`, which is nonzero whenever any recovered weight `m*` exceeds `e` in the
quadratic branch of the hybrid KL/quadratic divergence (`cc_algo/Psi.jl`). At a converged
solution this identity holds to machine precision; it is exactly zero only when no draw's
weight exceeds `e` (verified live at real D=20/W=80,000/L=50 points, both near-calibration and
under perturbation -- see `remediation_a1_verify_delta_dual_identity.jl`).
"""
function delta_dual_from_base(obj, base::BaseDualState)
    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    inner_x = vcat(base.ζstar, base.λstar)
    obj(inner_x, constr = @view(cbuf[1:ncon]))
    return cbuf[1] / 1e10
end

"""
    archC_base_state(x_free0, ctx_cm, cctx) -> BaseDualState

Architecture-C-accelerated drop-in replacement for `solve_base_state` (which
always uses the dense Architecture-A Hessian callback). Uses the SAME shared
FG callback as every other architecture (`inner_loop_internal_archgeneric`
only swaps the HESSIAN callback) -- so `obj.arg1` is populated identically to
how `solve_base_state`'s own call sites already read it
(`copy(ctx.obj.arg1)`, see e.g. `c10_d20_production_driver.jl`), reused here
rather than re-derived: no extra recompute needed, `obj.arg1` is trustworthy
immediately after `KN_solve` converges (KNITRO's own last FG call is always
made AT the reported solution).
"""
function archC_base_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, collect(x_free0))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end
    # skip_cm_fill_ref removal, REVERTED 2026-07-28 pending a fix (Goal 10), RE-ENABLED 2026-07-28
    # (no-moments/no-composite-G task): Goal 10 correctly found that `archC_hess_cb_builder`'s
    # Hessian callback called `_archC_prep_for_hessian!(o, xloc)` unconditionally, which recomputes
    # `arg0` via a DENSE `BLAS.gemv!` over `H[:, 2:1+outer_constr_index]` -- spanning the CM-grid
    # columns for this family -- so skipping their fill left that gemv reading stale/garbage data,
    # corrupting the whole Hessian (reproduced empirically, real D=20/W=80,000, 2/2 perturbed points
    # -> nStatus=-400). Goal 10's own fix instruction: "do not re-enable without first fixing
    # `_archC_prep_for_hessian!` itself to not require the dense CM columns (a genuine architectural
    # change, not a re-test)". That architectural change is now done:
    # `archC_hess_cb_builder`/`archA_partitioned_hess_cb_builder` now call
    # `_prep_dual_index_for_archC!`/`_prep_dual_index_for_archA!` (cm_hessian_architectures.jl),
    # which route through `operator_hessian_weights.jl::operator_prep_for_hessian!` -- computing
    # `arg0` via the SAME dense-G-free `dual_index!`/`economic_forward!`/bin-lookup forward kernels
    # `CMLookupState`'s own FG callback already uses (never reading `obj.H`'s CM columns), whenever
    # `cctx.inner_fg_backend != :dense_reference`. Re-validated D=4 and real D=20/W=80,000 via
    # test_shared_core_hessian_d4_gates.jl / test_d20_restricted_full_hessian_gates.jl (which
    # exercise this exact skip_fill_safe=true path for flexible-CM).
    #
    # A `cctx.core_hessian_backend !== :dense_reference` guard was added live (2026-07-28) then
    # REMOVED again in the same session: the economic-fill-skip extension it was protecting against
    # (wrap_moments_with_cm_archB) was itself reverted after causing an unrelated, unexplained H_EE
    # regression, and with that extension gone, `skip_fill=true` only ever skips the CM-grid columns
    # (never the economic ones) -- exactly the pre-existing, already-validated behavior this
    # condition had before today. Isolated empirically: re-adding the guard reproduced the SAME H_EE
    # mismatch `test_shared_core_hessian_d4_gates.jl` caught, even with the economic-skip extension
    # fully reverted -- root cause not yet isolated (suspected `core_ws`/`cf`-identity interaction
    # between the two priming-closure instances sharing `core_cf_ref`, not yet confirmed), so the
    # guard is left OUT rather than shipped with an unexplained side effect. Not a live production
    # concern: `core_hessian_backend=:dense_reference` is never set outside this repo's own explicit
    # comparison gates.
    use_lookup = cctx.inner_fg_backend == :cm_lookup
    skip_fill_safe = use_lookup && MOMENT_REPRESENTATION[] == :operator && cctx.cm_cross_hessian_backend == :winner_bin
    K, x, nStatus, n_fg, n_hess = use_lookup ?
        inner_loop_internal_cmlookup_production(obj, θ_full0, cctx; hess_cb_builder = _obj -> archC_hess_cb_builder(cctx),
            skip_fill = skip_fill_safe) :
        inner_loop_internal_archgeneric(obj, θ_full0;
            hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archC_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    end
    ζstar = x[1]; λstar = collect(x[2:end])
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, collect(x_free0), x)
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    archC_verified_state(x_free0, ctx_cm, cctx) -> (base::BaseDualState, verify::NamedTuple)

AUD-04 gap fix (docs/fullA_independent_audit_remediation.md, "known follow-ups" ->
`cm_checkpoint.jl` had no equivalent of `oracle.jl`'s `classify_inner_result`/
`is_verified_success` gate). Same Architecture-C inner solve as `archC_base_state` (kept
unchanged, still the cheap `base`-only path every existing caller uses), PLUS the independent
residual/gap diagnostics `classify_inner_result` needs (`:inner_status`, `:Delta_dual`,
`:primal_dual_gap`, `:weight_norm_resid`, `:mean_m_resid`, `:max_abs_moment_kkt_resid`,
`:m_min`) -- computed with the SAME explicit-recompute pattern `solve_base_state`
(three_way_derivatives.jl) / `evaluate_fullA`/`evaluate_fullA_fast` (oracle.jl/oracle_fast.jl)
already use: an `obj(inner_x, constr=...)` call to freshly repopulate `obj.arg1` (dPsi at the
converged (zeta*,lambda*)) and the constraint buffer (`Delta_dual`), NOT `archC_base_state`'s
own "trust KNITRO's last FG callback" shortcut -- an independent verification check should not
rely on the same assumption it exists to catch a violation of. Reuses `primal_divergence`
(oracle.jl) and `kkt_residual_blas` (oracle_fast.jl) rather than re-deriving either formula;
`G` itself is `obj.H`'s own moments!-output columns, already built once inside
`inner_loop_internal_archgeneric` above (same Phase-1A "no second moments! call" pattern
oracle_fast.jl uses), read via `CS.select_G_from_H`.

Returns `(base, verify)`: `base` is byte-identical in construction to what `archC_base_state`
returns (same inner solve, same fields); `verify` is a plain NamedTuple directly consumable by
`classify_inner_result`/`is_verified_success` (oracle.jl) via `get(verify, :field, default)`.
Callers that need the AUD-04 gate (e.g. `cm_checkpoint.jl`'s `cb_F!`/final `:stage_complete`
decision, via `cm_production_value_verified` below) should call this, not `archC_base_state`.
"""
function archC_verified_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx;
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0,
        verification_backend::Symbol = CM_VERIFICATION_BACKEND_DEFAULT[])
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, collect(x_free0))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end
    # skip_cm_fill_ref removal, REVISED, then REVERTED AGAIN 2026-07-28 (final-architecture-closure
    # task, Goal 10): a prior revision of this comment argued the skip was safe here because neither
    # `verify_inner_solution_operator_cm!` nor anything in the real production call chain reads
    # `obj.H` under `verification_backend===:operator` -- that reasoning is correct as far as it goes,
    # but incomplete: `skip_fill` here ALSO controls the SAME `_archC_prep_for_hessian!` dense-column
    # dependency documented in `archC_base_state`'s own HISTORY comment above (identical
    # `hess_cb_builder = archC_hess_cb_builder(cctx)`, identical inner-solve mechanism) -- the
    # verification step itself may be safe, but the LIVE INNER SOLVE that happens before it is not.
    # A real D=20/W=80,000 re-test at non-calibration points (see `archC_base_state`'s own comment)
    # confirmed this empirically for the base_state call site; the identical mechanism applies here.
    # `skip_fill_verify` is now kept at `false` unconditionally. Do not re-enable without first fixing
    # `_archC_prep_for_hessian!` itself (a genuine architectural change, not a re-test) -- see
    # docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md for the full corrected record.
    use_lookup_verify = cctx.inner_fg_backend == :cm_lookup
    skip_fill_verify = false   # ALWAYS false -- see HISTORY comment above
    K, inner_x, nStatus, n_fg, n_hess = cctx.inner_fg_backend == :cm_lookup ?
        inner_loop_internal_cmlookup_production(obj, θ_full0, cctx; hess_cb_builder = _obj -> archC_hess_cb_builder(cctx),
            skip_fill = skip_fill_verify) :
        inner_loop_internal_archgeneric(obj, θ_full0;
            hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archC_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = size(obj.U, 1)

    local m_weights, verify
    if verification_backend === :operator
        # Verification-defaults task (2026-07-27): operator-based post-solve verification -- NO
        # dense obj.H read. `cctx.core_cf_ref[]` is refreshed by moments! on every FG call of the
        # inner solve just completed above (see CMBinHessCtx's own docstring), so it holds the
        # CompressedFactual for THIS theta point already; a hard error (no silent fallback) if that
        # invariant doesn't hold.
        cf = cctx.core_cf_ref[]
        cf isa CompressedFactual || error("archC_verified_state: verification_backend=:operator requires cctx.core_cf_ref[] to be a CompressedFactual (got $(typeof(cf))) -- prerequisite not met, refusing silent dense fallback")
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        ov = verify_inner_solution_operator_cm!(ζstar, λstar, cf, cctx.L, cctx.nO, cctx.origins,
            cctx.refIndex1, bins_u, cctx.R, obj, W)
        m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    elseif verification_backend === :dense_reference
        G = CS.select_G_from_H(obj, obj.H)   # already built once by inner_loop_internal_archgeneric above

        # Explicit recompute at the converged point (solve_base_state/oracle.jl's own pattern, not
        # archC_base_state's KN_solve-last-call trust) -- populates obj.arg1 = m(s) fresh and yields
        # the constraint buffer needed for Delta_dual.
        ncon = obj.d - obj.outer_constr_index + 2
        cbuf = zeros(ncon)
        obj(inner_x, constr = @view(cbuf[1:ncon]))
        Delta_dual = cbuf[1] / 1e10
        m_weights = copy(obj.arg1)
        # Allocation fix (shared outer-A-gradient task, 2026-07-27, task §10): p_weights used to be
        # materialized as a fresh O(W) array purely to compute weight_norm_resid=abs(sum(p_weights)-1.0)
        # -- a floating-point-rounding-noise diagnostic (sum(p_weights)==1 identically up to rounding
        # by construction). `sum(x -> x / s_m_weights, m_weights)` is verified BIT-IDENTICAL to
        # `sum(m_weights ./ s_m_weights)` (same pairwise-summation algorithm, same per-element values,
        # same order -- checked directly at n=100/381/400/8000/80000, see
        # docs/VERIFIED_STATE_ALLOCATION_FIXES_2026-07-27.md) -- zero behavior change, zero array.
        s_m_weights = sum(m_weights)
        Delta_primal = primal_divergence(m_weights)   # oracle.jl, reused not re-derived

        mean_m_resid = abs(sum(m_weights) / W - 1.0)
        nkkt = min(length(λstar), size(G, 2))
        max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)   # oracle_fast.jl, reused not re-derived

        verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
                  primal_dual_gap = abs(Delta_dual - Delta_primal),
                  weight_norm_resid = abs(sum(x -> x / s_m_weights, m_weights) - 1.0),
                  mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
                  m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
        record_dense_reference_verification!()
    else
        error("archC_verified_state: unknown verification_backend=:$verification_backend (expected :operator or :dense_reference)")
    end

    base = BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, collect(x_free0), inner_x)
    return base, verify
end

"""
    cm_production_gradient(x_free0, pcx, ctx, pe; gradient_backend=:shared_inplace_pooled,
                            econ_ws=nothing, kwargs...) -> (g, meta)

`pcx = build_cm_production_context(...)`'s return value. One-call entry
point: Architecture-C inner solve (`archC_base_state`) + the (g,A_od)-block
gradient, fully wired for a KNITRO OUTER callback's `cb_G!`.

`gradient_backend` (shared-FG-verification-and-A-gradient release, 2026-07-27): mirrors
`cm_originzc_production_gradient`/`cm_meanzc_production_gradient`'s own kwarg exactly.
  - `:shared_inplace_pooled` (DEFAULT): the shared `economic_A_gradient!` entry point
    (shared_a_gradient.jl). `build_lfix_base_cache_cm`'s own CM-folded `q0` is passed through
    unchanged via `economic_A_gradient!`'s `cache=` kwarg -- same contract as before.
  - `:legacy_unbuffered`: the ORIGINAL, fully-allocating `composite_gradient_at_fast` -- kept ONLY
    as an explicit reference/debug backend.
"""
function cm_production_gradient(x_free0::AbstractVector, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing,
        gradient_backend::Symbol = :shared_inplace_pooled,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, kwargs...)
    base = base === nothing ? archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx) : base
    cache = build_lfix_base_cache_cm(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    if gradient_backend === :shared_inplace_pooled
        D = pcx.ctx_cm.D; Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
        ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
        g_econ = zeros(D * Ddest)
        meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
        return g_econ, meta
    elseif gradient_backend === :legacy_unbuffered
        return composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    else
        error("cm_production_gradient: gradient_backend must be :shared_inplace_pooled|:legacy_unbuffered, got $gradient_backend")
    end
end

"""
    cm_production_value(x_free0, pcx) -> (K, base)

Architecture-C inner solve only (objective value, no gradient) -- for a
KNITRO outer `cb_F!` or a plain feasibility/kappa check.
"""
function cm_production_value(x_free0::AbstractVector, pcx)
    base = archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base
end

"""
    cm_production_value_verified(x_free0, pcx) -> (K, base, verify)

AUD-04-aware analogue of `cm_production_value` -- same Architecture-C inner solve, plus the
`verify` NamedTuple (`archC_verified_state`) needed to gate a result through
`classify_inner_result`/`is_verified_success` (oracle.jl) before it may become an incumbent or a
`:stage_complete` checkpoint. Costs one extra `obj(...)` recompute call over `cm_production_value`
(see `archC_verified_state`'s docstring) -- use this, not `cm_production_value`, at any call site
that needs the AUD-04 gate (currently: `cm_checkpoint.jl`'s `run_cm_upper_checkpointed`).
"""
function cm_production_value_verified(x_free0::AbstractVector, pcx)
    base, verify = archC_verified_state(x_free0, pcx.ctx_cm, pcx.cctx)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end
