# ============================================================================
# Production outer bridge task (2026-08-01), §13: a production outer-runner SCAFFOLD
# supporting the live family combined outer vector, in two EXPLICIT modes.
#
# RECOVERED 2026-08-02 (integration/phase12-13-runner-checkpoints-2026-08-02) from commit
# c8c9c80, which was silently dropped by a later merge and existed in neither this branch's
# history nor its tree. ADAPTED here to the REAL, finished family-adapter surface that did not
# exist when c8c9c80 was written -- concrete diffs from the original, documented inline below and
# in this branch's own commit message (not just asserted):
#
#   1. `run_profiled_family_outer_search`/`PROFILED_ALL_FAMILY_OUTER_AB_HARNESS_2026-08-01.jl`,
#      which the original `:fixed_gp_parameterization_ab` mode was a thin pass-through to, DOES
#      NOT EXIST anywhere in this branch's ancestry (whole-tree grep confirms zero matches) --
#      it must have lived only in whatever sibling session/branch built that file, never merged
#      here. Both modes are now implemented directly by ONE shared internal KNITRO driver,
#      `_run_profiled_outer_knitro_loop`, parameterized by `gp_free::Bool` (false for
#      :fixed_gp_parameterization_ab, true for :production_bound_search) -- the exact same
#      cb_F!/cb_G! SHAPE the original's own `_run_production_bound_search` already used for the
#      free-gp mode, now shared by both modes instead of forked.
#   2. `family_ctx_builder(ctx, ev)` (build fctx AFTER the first solve, from its own `ev`) is
#      backwards versus how the REAL family adapters actually work: `OriginZCFamilyCtx`/
#      `CMZCFamilyCtx`/`FlexCMFamilyCtx`/`FrechetFamilyCtx` are built ONCE, up front, wrapping
#      persistent buffers (`rc0_ws`/`rc0_buf`/`cctx`/`op`/`zc_ws`) that must NOT be reallocated
#      per outer point -- the family ctx is an INPUT to the evaluator, not a derived output of a
#      solve. This file now takes a pre-built `fctx` directly; `evaluate_fn(w_profiled, fctx) ->
#      ev` matches the REAL adapters' own evaluator signature convention
#      (`evaluate_profiled_flexcm_point`/`evaluate_profiled_frechet_point`,
#      profiled_restricted_family_adapters_2026-08-02.jl; `evaluate_profiled_originzc_point`/
#      `evaluate_profiled_cmzc_point`, profiled_zc_lane_point_evaluators_2026-08-02.jl, this
#      branch's own new file).
#   3. The original's "verified" check read `ev.result.primal_dual_gap`/`ev.result.mean_m_resid`/
#      `ev.result.max_abs_moment_kkt_resid` -- fields the REAL restricted-family evaluators' own
#      `result` NamedTuple does NOT have (only `inner_status`/`zeta`/`beta`/`n_fg_calls`/
#      `n_hess_calls`; verification-derived fields were never part of that shape for the four
#      restricted families). Replaced with an optional `verify_fn(ev) -> NamedTuple` callback
#      (defaults to a weaker, always-available `inner_status`-only classification when omitted)
#      so a caller CAN wire the real `operator_verification.jl` verifiers per family without this
#      file hard-coding field names that don't exist on every family's `ev`.
#   4. `combined_outer_coordinate_names` does not exist anywhere in this tree either -- the
#      fallback branch (generic `gp`/`r_free_k` names) is now the ONLY branch; the dead
#      `isdefined(...) ? combined_outer_coordinate_names(fctx) : ...` ternary is removed rather
#      than kept as unreachable dead code.
#   5. Phase 12's new versioned/reduced-basis infrastructure is now wired in (all OPTIONAL,
#      default off, so this stays a strict superset of the original's behavior when a caller
#      passes none of them): `use_screen` (profiled_cm_screen_precheck!,
#      profiled_screen_bridge_2026-08-02.jl), `cache` (profiled_cm_cache_lookup_or_compute! +
#      ProfiledCMProductionEvalKey, profiled_reduced_basis_cache_bank_2026-08-02.jl), `bank`
#      (ProfiledRestrictedDualBank -- RECORDING-ONLY in this pass: every successful solve is
#      recorded, but the real per-family reduced lookup kernels (reserved files, not touched by
#      this task) do not currently expose a warm-start-injection hook this runner could feed a
#      selected candidate into, so this is honest observability infrastructure, not yet a solve-
#      speed optimization -- same "explicit placeholder, not a fabricated concrete value"
#      discipline the original scaffold's own `production_subsystems` manifest already used),
#      and `checkpoint_path`/`checkpoint_interval_s` (CMCheckpointV11, cm_checkpoint.jl).
#
# What this scaffold still does NOT do (honestly, not silently): implement a genuine joint
# (gp, A, eta/nu) outer search -- `nu_full` is fixed per `evaluate_fn` closure (see
# profiled_zc_lane_point_evaluators_2026-08-02.jl's own docstring); the original scaffold never
# had an eta axis either, so this is not a regression, just an unclaimed extension.
# ============================================================================

isdefined(Main, :validate_family_layout_contract) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :stable_layout_digest) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires profiled_stable_layout_digest_2026-08-01.jl to be included first.")
isdefined(Main, :shared_family_outer_gradient) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires profiled_shared_economic_gradient_engine_2026-08-01.jl to be included first.")
isdefined(Main, :set_outer_algorithm_direct!) ||
    error("profiled_production_outer_runner_2026-08-01.jl requires knitro_outer_algorithm.jl to be included first (Direct+SR1 wiring, task §13).")

using KNITRO, Dates
import CSV, DataFrames

const VALID_OUTER_RUNNER_MODES = (:fixed_gp_parameterization_ab, :production_bound_search)

"""
    OuterRunManifest

Immutable record of what a given outer run actually did -- task §13 ("each mode must record its
free-coordinate names in the manifest") + §14/§15's comparability/plumbing needs. Every field is
filled with either a real value or an explicit placeholder Symbol (`:not_yet_wired`,
`:unknown_pending_inner`) -- never silently absent. UNCHANGED from the recovered c8c9c80 shape.
"""
struct OuterRunManifest
    mode::Symbol
    family::Symbol
    label::String
    free_coordinate_names::Vector{String}
    n_free::Int
    gp_free::Bool
    hessopt_tag::String
    knitro_opt_file::String
    stable_layout_digest::String
    economic_parameterization::Symbol   # :profiled_destination_scales (this whole branch) -- see §15
    production_subsystems::NamedTuple
    timestamp::String
end

"""
    default_production_subsystems_manifest(; use_screen=false, use_cache=false, use_bank=false) -> NamedTuple

Task §14's required fields, each honestly tagged (see file header). Adapted 2026-08-02: the three
kwargs let a caller report which Phase 12 subsystems are ACTUALLY wired for this run, rather than
the original's hard-coded `:not_yet_wired` for every one of them -- still defaults to the
original's own honest "nothing wired" baseline when a caller passes nothing.
"""
default_production_subsystems_manifest(; use_screen::Bool = false, use_cache::Bool = false, use_bank::Bool = false) = (
    dual_bank_policy = use_bank ? :recording_only_no_warmstart_injection : :not_yet_wired,
    obj_x_reuse_policy = :inherited_from_ab_harness,   # AB harness's cb_G! reuses last_ev/last_fctx when w unchanged -- see run_profiled_family_outer_search
    exact_cache_policy = use_cache ? :profiled_reduced_basis_exact_cache : :not_yet_wired,
    screen_set = use_screen ? :profiled_cm_screen_precheck : :not_yet_wired,
    restriction_backend = :not_yet_wired,
    solver_options_file = :inherited_from_ab_harness,
)

_combined_names_fallback(pe) = ["gp"; ["r_free_$k" for k in 1:(outer_dim_profiled(pe) - 1)]]

"""
    run_profiled_production_outer(mode, label, w_start; fctx, evaluate_fn, ctx, pe,
        maxtime_real=1800.0, hessopt_tag="sr1", maxit_override=nothing, trace_csv=nothing,
        gp_bounds_halfwidth=0.05, use_screen=false, cache=nothing, bank=nothing,
        checkpoint_path=nothing, checkpoint_interval_s=60.0, verify_fn=nothing) -> (result, manifest, config)

Task §13 entry point, ADAPTED (see file header, item 2) to take a pre-built, persistent `fctx`
(any real family adapter: `OriginZCFamilyCtx`/`CMZCFamilyCtx`/`FlexCMFamilyCtx`/`FrechetFamilyCtx`)
plus `evaluate_fn(w_profiled, fctx) -> ev` matching that family's own real evaluator. `mode in
VALID_OUTER_RUNNER_MODES`, throws on anything else (no silent default mode).
"""
function run_profiled_production_outer(mode::Symbol, label::String, w_start::Vector{Float64}; fctx,
        evaluate_fn::Function, ctx, pe::PivotGravityElimOnRetained,
        maxtime_real::Float64 = 1800.0, hessopt_tag::String = "sr1",
        maxit_override::Union{Nothing,Int} = nothing, trace_csv::Union{Nothing,AbstractString} = nothing,
        gp_bounds_halfwidth::Float64 = 0.05,
        use_screen::Bool = false, cache = nothing, bank = nothing,
        checkpoint_path::Union{Nothing,AbstractString} = nothing, checkpoint_interval_s::Float64 = 60.0,
        verify_fn::Union{Nothing,Function} = nothing,
        inner_layout_digest = :unknown_pending_inner)
    mode in VALID_OUTER_RUNNER_MODES ||
        error("run_profiled_production_outer: mode=:$mode not in $VALID_OUTER_RUNNER_MODES")

    validate_family_layout_contract(fctx)
    names_all = _combined_names_fallback(pe)
    digest = stable_layout_digest(fctx)
    opt_file = joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt")
    subsystems = default_production_subsystems_manifest(; use_screen, use_cache = cache !== nothing, use_bank = bank !== nothing)
    fam = family_kind(fctx)

    config = build_profiled_production_config(fctx; economic_parameterization = :profiled_destination_scales,
        dual_bank_policy = subsystems.dual_bank_policy, inner_dual_layout_digest = inner_layout_digest)

    gp_free = mode == :production_bound_search
    names = gp_free ? names_all : names_all[2:end]
    result = _run_profiled_outer_knitro_loop(label, w_start; fctx, evaluate_fn, ctx, pe,
        maxtime_real, hessopt_tag, maxit_override, trace_csv, gp_bounds_halfwidth, gp_free,
        use_screen, cache, bank, checkpoint_path, checkpoint_interval_s, verify_fn, config)

    manifest = OuterRunManifest(mode, fam, label, names, length(names), gp_free,
        hessopt_tag, opt_file, digest, :profiled_destination_scales, subsystems, string(now()))
    return result, manifest, config
end

"""
    _run_profiled_outer_knitro_loop(label, w_start; fctx, evaluate_fn, ctx, pe, gp_free, ...) -> NamedTuple

ONE shared internal KNITRO driver for BOTH outer-runner modes (adaptation item 1 -- replaces the
original's split between a thin pass-through to a nonexistent AB harness and a separately-forked
`_run_production_bound_search`). `gp_free=false` reproduces `:fixed_gp_parameterization_ab`
(gp fixed at `w_start[1]`, KNITRO variables are the free-A coordinates only, `cb_G!` drops
`g[1]`); `gp_free=true` reproduces `:production_bound_search` (gp is a genuine free KNITRO
variable with its own bounds, `cb_G!` passes the FULL `shared_family_outer_gradient` output).
"""
function _run_profiled_outer_knitro_loop(label::String, w_start::Vector{Float64}; fctx,
        evaluate_fn::Function, ctx, pe::PivotGravityElimOnRetained,
        maxtime_real::Float64, hessopt_tag::String,
        maxit_override::Union{Nothing,Int}, trace_csv::Union{Nothing,AbstractString},
        gp_bounds_halfwidth::Float64, gp_free::Bool,
        use_screen::Bool, cache, bank, checkpoint_path::Union{Nothing,AbstractString},
        checkpoint_interval_s::Float64, verify_fn::Union{Nothing,Function}, config)
    lp(xs...) = (println(xs...); flush(stdout))
    fam = family_kind(fctx)

    t_start = time()
    n_eval = Ref(0); n_grad_calls = Ref(0)
    trace = NamedTuple[]
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    last_w = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_ev = Ref{Any}(nothing)
    t_last_ckpt = Ref(t_start)

    classify(ev) = verify_fn === nothing ?
        (verified = ev.result.inner_status in (0, -100, -101, -103),) : verify_fn(ev)

    function solve_at(w_full::Vector{Float64})
        if use_screen
            try
                profiled_cm_screen_precheck!(w_full, ctx, pe; use_witness = false)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                return nothing   # screen-certified infeasible: no inner solve attempted
            end
        end
        if cache !== nothing
            key = ProfiledCMProductionEvalKey(:profiled_destination_scales, w_full, Float64[],
                ctx.δ, ctx.find_smallest, fam, config.stable_layout_digest, "")
            ev, _ = profiled_cm_cache_lookup_or_compute!(cache, key,
                () -> (ev = evaluate_fn(w_full, fctx); (ev, (inner_status = ev.result.inner_status,))))
            return ev
        end
        return evaluate_fn(w_full, fctx)
    end

    ev0 = solve_at(w_start)
    ev0 !== nothing && ev0.result.inner_status in (0, -100, -101, -103) ||
        error("_run_profiled_outer_knitro_loop($label): start point not inner-feasible/not screen-passing")
    lp("[$label] gp_free=$gp_free family=:$(fam)  gp_start=$(w_start[1])  Delta_dual(zeta)=$(ev0.result.zeta)  status=$(ev0.result.inner_status)")

    n_total = length(w_start)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    # Task §13's explicit ask: "wire ... Direct+SR1". Confirmed via knitro_outer_algorithm.jl
    # (KNITRO_ALGORITHM_DIRECT=1, KNITRO_HESSOPT_SR1=3) -- the ADAPTED original scaffold
    # hardcoded `algorithm=3` (Active-Set/SLQP, NOT Direct) here; fixed to call the existing
    # set_outer_algorithm_direct! helper instead of re-deriving the KNITRO parameter codes.
    set_outer_algorithm_direct!(kc, KNITRO_HESSOPT_SR1)

    z_halfwidth = 30.0
    if gp_free
        n_var = n_total
        lo = copy(w_start) .- z_halfwidth; lo[1] = w_start[1] - gp_bounds_halfwidth
        hi = copy(w_start) .+ z_halfwidth; hi[1] = w_start[1] + gp_bounds_halfwidth
        x0 = w_start
    else
        n_var = n_total - 1
        lo = w_start[2:end] .- z_halfwidth
        hi = w_start[2:end] .+ z_halfwidth
        x0 = w_start[2:end]
    end
    xIndices = KNITRO.KN_add_vars(kc, n_var)
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, x0)

    to_full(x) = gp_free ? collect(Float64, x) : vcat(w_start[1], collect(Float64, x))

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w_full = to_full(evalRequest.x)
        ev = solve_at(w_full)
        if ev === nothing || !(ev.result.inner_status in (0, -100, -101, -103))
            evalResult.obj[1] = 1e10
            n_eval[] += 1
            return 0
        end
        Δ = -ev.result.zeta   # matches the original's own Delta_dual-as-objective convention (fixed-dual functional, negated zeta at q0 -- see shared_family_outer_gradient's own cache for the exact q0/zeta relationship)
        evalResult.obj[1] = Δ
        n_eval[] += 1
        last_w[] = w_full; last_ev[] = ev
        t_el = time() - t_start
        cls = classify(ev)
        verified = cls.verified
        find_smallest = hasproperty(ctx, :find_smallest) ? ctx.find_smallest : true
        is_new_best = verified && (best[] === nothing || (find_smallest ? Δ < best[].Delta : Δ > best[].Delta))
        if is_new_best
            best[] = (w = copy(w_full), Delta = Δ, t_elapsed = t_el, n_eval = n_eval[])
            bank !== nothing && record_success_profiled!(bank, n_eval[], w_full, vcat(ev.result.zeta, ev.result.beta))
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta = Δ, gp = w_full[1], inner_status = ev.result.inner_status, verified = verified))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [$label] eval $(n_eval[]) t=$(round(t_el,digits=1))s Delta=$Δ gp=$(w_full[1]) status=$(ev.result.inner_status)")
        end
        if checkpoint_path !== nothing && (time() - t_last_ckpt[] >= checkpoint_interval_s || is_new_best)
            _write_profiled_checkpoint(checkpoint_path, label, fam, config, w_full, ev, n_eval[], n_grad_calls[], t_el, maxtime_real - t_el)
            t_last_ckpt[] = time()
        end
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w_full = to_full(evalRequest.x)
        ev = (last_w[] !== nothing && last_w[] == w_full) ? last_ev[] : solve_at(w_full)
        g, meta = shared_family_outer_gradient(w_full, ctx, fctx, ev)
        n_grad_calls[] += 1
        evalResult.objGrad .= gp_free ? g : g[2:end]
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    lp("[$label] DONE: status=$nStatus_code wall=$(round(wall_ext,digits=1))s n_eval=$(n_eval[]) n_grad=$(n_grad_calls[])")
    b = best[]
    b !== nothing && lp("  best: Delta=$(b.Delta) found_at_eval=$(b.n_eval) t=$(b.t_elapsed)s")

    if trace_csv !== nothing
        DataFrames.DataFrame(trace) |> df -> CSV.write(trace_csv, df)
    end

    return (label = label, family = fam, knitro_status = nStatus_code, wall_ext = wall_ext,
        n_eval = n_eval[], n_grad_calls = n_grad_calls[], w_terminal = to_full(xsol),
        best = b, trace = trace)
end

"""
    _write_profiled_checkpoint(path, label, family, config, w_full, ev, n_eval, n_grad, wall_elapsed, wall_budget_remaining) -> Nothing

Phase 12 item 3 wiring: writes a real, versioned `CMCheckpointV11` for the profiled/reduced-basis
runner. Only the fields this runner actually tracks are populated with real values; every field
`CMCheckpointV9` carries that has no analogue here (cm_L/cm_probs/... -- CM-grid config the
flexible-CM/CM+ZC families DO have but origin-ZC does not) is filled with a neutral/zero default
rather than a fabricated concrete value, since this checkpoint's OWN `checkpoint_namespace` (from
`config`) is what a resumer must check via `assert_checkpoint_compatible` before trusting anything
else in the file.
"""
function _write_profiled_checkpoint(path::AbstractString, label::String, family::Symbol, config,
        w_full::Vector{Float64}, ev, n_eval::Int, n_grad::Int, wall_elapsed::Float64, wall_budget_remaining::Float64)
    ckpt = CMCheckpointV11(CM_CHECKPOINT_SCHEMA_V11, "profiled_$(label)", label, :cm_upper, true, 1.0,
        0, 0, :sobol_randomized, "", "", 0, Float64[], :anchored, :equal, :orthonormal, :structured, :cplus,
        Symbol(family), 0, 0, :direct, 1,
        w_full[1], w_full[2:end], Float64[], zeros(0, 0), collect(ev.result.beta), Dict{Int,Float64}(),
        (w = w_full, Delta = -ev.result.zeta), n_eval, n_grad, wall_elapsed, wall_budget_remaining, :wall_interval,
        "unknown", :unrecorded, nothing, 0, :common_flexible, :legacy_z,
        config.economic_parameterization, config.stable_layout_digest, string(config.inner_dual_layout_digest),
        config.hessian_backends.H_ZZ, config.hessian_backends.H_CZ, config.hessian_backends.H_EZ,
        config.hessian_backends.source, config.full_a_recovery_convention, config.checkpoint_namespace)
    save_cm_checkpoint(path, ckpt)
    return nothing
end
