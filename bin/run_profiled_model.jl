# fix/profiled-functional-readiness-closeout-2026-08-03, section 6: the missing canonical CLI
# entry point (task's own bin/run_profiled_model.jl). ONE public script dispatching by
# (family, formulation), replacing the prior state where every REDUCED family had to be launched
# via its own bespoke run_coldsolve_*/run_prodscale_*/run_outer_*_reduced_constrained_*.jl script
# with hand-copied context/layout/fctx construction (see FamilyRegistry.jl's own notes on which
# family had which ad-hoc script). Not five family-specific public scripts -- the per-family
# construction below is real (reusing exactly the adapters profiled_restricted_family_adapters_
# 2026-08-02.jl / profiled_originzc_family_adapter_2026-08-02.jl / profiled_cmzc_family_adapter_
# 2026-08-02.jl / profiled_family_adapters_2026-08-01.jl already define), not a new abstraction.
#
# REDUCED (economic_parameterization=:profiled_destination_scales) dispatches to
# run_profiled_upper_constrained for all 5 families -- the genuine gp-free, Delta<=delta-
# constrained REDUCED runner (profiled_production_outer_constrained_2026-08-02.jl), the one this
# whole branch's checkpoint/resume work targets.
#
# FULL (economic_parameterization=:full_gamma_normalized) dispatches to the real production entry
# points per FamilyRegistry's own `outer_evaluator` field. flexible_cm/common_frechet/cm_meanzc
# share run_cm_upper_checkpointed (a single well-understood signature); unrestricted uses
# run_polish_checkpointed_unified (_run_full_unrestricted, its own constructed OuterCoordinateLayout
# traced from 4 corroborating production/campaign sources, see that function's header).
#
# origin_zc (_run_full_originzc, ADDED 2026-08-04 continuation) uses run_originzc_upper_checkpointed
# with distribution_restriction=:origin_specific_moments, K_mean=1, K_pair=0,
# power_target_layout=:origin_by_power -- traced, not guessed, from run_outer_originzc_full_
# 2026-08-02.jl, the REAL dedicated FULL production script built specifically to match the REDUCED
# path's own OriginByPowerLayout(D,1,0) exactly (that script's own header states this explicitly).
# This is the SAME (K_mean=1,K_pair=0) configuration this session's own REDUCED-path free-nu/
# free-eta gates already use throughout (test_zc_free_nu_production_driver_d4/d20_2026-08-04.jl,
# test_zc_free_eta_evaluator_d20_2026-08-04.jl) -- confirmed matched, not coincidental. An EARLIER
# pass through this codebase (same session) had instead traced to campaign_cm_family_runner.jl/
# campaign_cm_family_runner_sigma3.jl, which configure origin_zc for a DIFFERENT restriction type
# entirely (:origin_specific_moments_zero_covariance, a pairwise/zero-covariance restriction, K=1
# vs K=2 in those two files respectively) -- that apparent "conflict" was a red herring: those two
# files' shared restriction type was never the one REDUCED's own gates were built against this
# session, so neither of their K values was ever the right one to trace for this CLI wrapper.
# nu is pinned near 1.0 (nu_bounds ~1e-8-wide box around log(1)=0), matching the dedicated script's
# own fixed-nu smoke convention -- the separately-wired free-nu capability
# (profiled_zc_free_nu_production_driver_2026-08-04.jl) is a REDUCED-only driver, not part of this
# FULL CLI wrapper's own scope.
#
# Usage:
#   julia --project=<repo root> bin/run_profiled_model.jl \
#       --config configs/fullA_production_2026-08-03.toml \
#       --family flexible_cm --formulation reduced --direction upper --delta 1.0 \
#       [--resume <checkpoint path>] [--diagnostic-budget 30]
const REPO_ROOT = dirname(@__DIR__)
const D4X = joinpath(REPO_ROOT, "full_aod_diag", "d4_exact")

include(joinpath(REPO_ROOT, "scientific_manifest", "RunManifest.jl"))
using .RunManifestMod
include(joinpath(REPO_ROOT, "scientific_manifest", "FamilyRegistry.jl"))
using .FamilyRegistryMod

# ----------------------------------------------------------------------------
# CLI parsing -- plain ARGS walk, no external dependency. Every flag is required except --resume
# and --diagnostic-budget (script-execution knobs, not scientific parameters -- CLAUDE.md's
# no-silent-default rule applies to what economic problem is solved, not to how long KNITRO is
# allowed to run for a smoke test).
# ----------------------------------------------------------------------------
function parse_cli(args::Vector{String})
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("run_profiled_model.jl: unexpected positional argument \"$a\" (all arguments are --key value)")
        key = a[3:end]
        i == length(args) && error("run_profiled_model.jl: --$key given with no value")
        d[key] = args[i+1]
        i += 2
    end
    # task §6 fix (profiled-outer-ab-readiness-2026-08-04, live user challenge): --threaded-gradient
    # was originally optional-with-a-default -- exactly the trap this repo's own standing rule
    # exists to prevent (feedback-no-defaults-on-any-input-or-setting: no function/setting may
    # default, including non-scientific execution flags like this one). Required here, same as
    # every other real flag below, so a caller who forgets it gets an immediate, loud error instead
    # of a silently-serial REDUCED run -- exactly the bug this omission caused live this session.
    for req in ("config", "family", "formulation", "direction", "delta", "threaded-gradient")
        haskey(d, req) || error("run_profiled_model.jl: missing required --$req")
    end
    d["formulation"] in ("full", "reduced") ||
        error("run_profiled_model.jl: --formulation must be \"full\" or \"reduced\", got \"$(d["formulation"])\"")
    d["direction"] in ("upper", "lower") ||
        error("run_profiled_model.jl: --direction must be \"upper\" or \"lower\", got \"$(d["direction"])\"")
    d["threaded-gradient"] in ("true", "false") ||
        error("run_profiled_model.jl: --threaded-gradient must be \"true\" or \"false\", got \"$(d["threaded-gradient"])\"")
    # task §6.1 (profiled-outer-ab-completion-2026-08-04): explicit required A-coordinate-mode
    # selection for the REDUCED path, same no-silent-default discipline as --threaded-gradient
    # above. Only meaningful for --formulation reduced (FULL's own A_coordinate_mode is chosen by
    # its existing family-specific dispatch, unrelated to this REDUCED-only mode) -- required only
    # in that case, not forced on FULL callers.
    if d["formulation"] == "reduced"
        haskey(d, "a-coordinate-mode") || error("run_profiled_model.jl: --formulation reduced requires --a-coordinate-mode " *
            "(one of \"profiled_pivot_anchor_relative\"|\"profiled_powered_relative_A\") -- no silent default.")
        d["a-coordinate-mode"] in ("profiled_pivot_anchor_relative", "profiled_powered_relative_A") ||
            error("run_profiled_model.jl: --a-coordinate-mode must be \"profiled_pivot_anchor_relative\" or " *
                  "\"profiled_powered_relative_A\", got \"$(d["a-coordinate-mode"])\"")
    end
    return d
end

"""
    load_scientific_manifest_toml(path) -> ScientificManifest

Loads a `ScientificManifest` from a plain TOML file matching `to_toml_dict`'s own schema (the
same file `scientific_manifest/ScientificManifest.jl`'s own `write_manifest_toml`/
`read_manifest_toml` round-trips, e.g. `configs/fullA_production_2026-08-03.toml`). Deliberately
does NOT re-`include` `ScientificManifest.jl` directly into `Main` -- `RunManifest.jl` (included
above) already `include`s it as the submodule `RunManifestMod.ScientificManifestMod`; a second,
separate top-level `include` would define a SECOND, type-distinct `ScientificManifest` struct
(confirmed live this session: `TypeError: in keyword argument sci, expected
Main.RunManifestMod.ScientificManifestMod.ScientificManifest, got ... Main.ScientificManifestMod.
ScientificManifest` -- Julia's module system treats two separate `include`s of the same file as
two separate types, even though the source is byte-identical). Always go through the ONE copy
`RunManifestMod` already loaded.
"""
function load_scientific_manifest_toml(path::AbstractString)
    return Base.invokelatest(Main.RunManifestMod.ScientificManifestMod.read_manifest_toml, path)
end

function main(args::Vector{String})
    cli = parse_cli(args)

    println("="^90); println("run_profiled_model.jl: canonical (family, formulation) runner"); println("="^90)
    flush(stdout)

    # ---- refuse a dirty worktree for anything that isn't an explicit low-budget dev smoke.
    # task §6: "refuse dirty production-like runs" -- a run is production-like unless
    # --diagnostic-budget is set AND under 300s, i.e. a bounded smoke/gate run, not a real
    # campaign point. This mirrors RunManifest.refuse_if_dirty's own intent without weakening it:
    # a genuinely long/unbounded run always refuses on a dirty tree, no exceptions.
    diagnostic_budget = haskey(cli, "diagnostic-budget") ? parse(Float64, cli["diagnostic-budget"]) : nothing
    is_smoke = diagnostic_budget !== nothing && diagnostic_budget <= 300.0
    if !is_smoke
        RunManifestMod.refuse_if_dirty(repo_dir = REPO_ROOT)
    elseif RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT)
        println("[run_profiled_model] WARNING: worktree is dirty; proceeding ONLY because this is a bounded diagnostic-budget<=300s smoke run, not a production-like run.")
    end
    flush(stdout)

    sci = load_scientific_manifest_toml(cli["config"])
    family_canon = Symbol(cli["family"])
    family_canon in FamilyRegistryMod.CANONICAL_FAMILIES ||
        error("run_profiled_model.jl: --family=$(family_canon) not one of $(FamilyRegistryMod.CANONICAL_FAMILIES)")
    formulation = cli["formulation"] == "reduced" ? :profiled_destination_scales : :full_gamma_normalized
    cap = FamilyRegistryMod.capability(family_canon, formulation)
    println("[run_profiled_model] family=:$family_canon formulation=:$formulation outer_evaluator=$(cap.outer_evaluator)")
    flush(stdout)

    delta = parse(Float64, cli["delta"])
    find_smallest = cli["direction"] == "upper"
    maxtime_real = diagnostic_budget === nothing ? 180.0 : diagnostic_budget
    resume_from = get(cli, "resume", nothing)

    outdir_base = joinpath(REPO_ROOT, "results", "canonical_runner")
    mkpath(outdir_base)

    if formulation == :profiled_destination_scales
        _run_reduced(family_canon, sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)
    else
        _run_full(family_canon, sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)
    end
end

# ----------------------------------------------------------------------------
# REDUCED dispatch -- reuses the same per-family construction pattern proven by
# test_all_family_checkpoint_resume_2026-08-03.jl (this branch, 2026-08-03).
# ----------------------------------------------------------------------------
function _run_reduced(family::Symbol, sci, cli, delta::Float64, find_smallest::Bool,
        maxtime_real::Float64, resume_from, outdir_base::String)
    for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
              "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
              "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
              "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
              "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
              "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
              "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
              "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
              "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
              "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
              "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
              "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
              "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
              "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
              "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
              "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
              "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
              "cm_exact_cache_production.jl", "cm_checkpoint.jl", "draw_design.jl",
              "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
              "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
              "homogeneous_contraction_2026-07-31.jl",
              "reduced_homogeneous_hessian_2026-08-01.jl",
              "reduced_homogeneous_contraction_2026-08-01.jl",
              "profiled_restricted_family_base_2026-08-01.jl",
              "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
              "reduced_operator_verification_2026-08-01.jl",
              "recover_full_a_2026-07-31.jl",
              "profiled_reduced_lookup_kernels_2026-08-02.jl",
              "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
              "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
              "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
              "profiled_outer_gradient_layout_contract_2026-08-01.jl",
              "profiled_stable_layout_digest_2026-08-01.jl",
              "profiled_operator_bundle_2026-08-01.jl",
              "profiled_outer_evaluator_2026-08-01.jl",
              "profiled_lfix_incremental_2026-08-01.jl",
              "profiled_shared_economic_gradient_engine_2026-08-01.jl",
              "profiled_family_adapters_2026-08-01.jl",
              "profiled_restriction_contrib0_operators_2026-08-01.jl",
              "profiled_restricted_family_adapters_2026-08-02.jl",
              "profiled_originzc_family_adapter_2026-08-02.jl",
              "profiled_cmzc_family_adapter_2026-08-02.jl",
              "profiled_zc_lane_point_evaluators_2026-08-02.jl",
              "profiled_reduced_basis_cache_bank_2026-08-02.jl",
              "profiled_screen_bridge_2026-08-02.jl",
              "dual_bank.jl", "cm_dual_bank_production.jl",
              "profiled_production_outer_runner_2026-08-01.jl",
              "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
              "profiled_production_outer_constrained_2026-08-02.jl",
              "profiled_zc_free_eta_2026-08-04.jl",
              "profiled_zc_free_nu_production_driver_2026-08-04.jl",
              "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
              "profiled_coordinate_mode_dispatch_2026-08-04.jl"]
        Base.include(Main, joinpath(D4X, f))
    end

    ctx = Base.invokelatest(Main.d20_real_setup_design; W = sci.W, δ = delta, find_smallest = find_smallest,
        draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        destination_sample = sci.destination_sample, exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells, σHat = sci.sigma)
    D = ctx.D

    # Brazil/Korea anchor-pivot override: the SAME hardcoded (korea_idx=14, brazil_idx=3) every
    # existing D20 REDUCED production driver in this directory uses (e.g.
    # run_coldsolve_flexcm_w100k_2026-08-02.jl:54-56) -- a real, data-specific anchor-tie
    # resolution, not invented here.
    korea_idx, brazil_idx = 14, 3
    spec = Base.invokelatest(Main.build_anchor_spec_from_ctx, ctx; global_overrides = Dict(korea_idx => brazil_idx))
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
    gauge = Base.invokelatest(Main.build_anchor_gauge, z_calib, spec)
    pe = Base.invokelatest(Main.build_pivot_elimination_on_retained, ctx, spec, gauge)
    gp0 = θ0[3+D]
    w0 = Base.invokelatest(Main.reduce_to_w_profiled, gp0, z_calib, pe)

    θ_full_calib = Base.invokelatest(Main.CS.reconstruct_full, ctx.θ0_up[ctx.free_idx], ctx.m)
    cf_probe = Base.invokelatest(Main.build_compressed_factual, collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = Base.invokelatest(Main.build_profiled_economic_moment_layout, ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = Base.invokelatest(Main.build_reduced_base_obj_for_family, ctx, layout, Main.CS)

    # task §3.1/§3.2 fix (profiled-outer-ab-readiness-2026-08-04): origin_zc/cm_meanzc now dispatch
    # to the genuine free-nu production driver (run_profiled_upper_constrained_free_nu,
    # profiled_zc_free_nu_production_driver_2026-08-04.jl) instead of run_profiled_upper_constrained
    # with the fixed-nu 3-arg evaluate_fn -- the earlier version of this dispatch pinned nu at the
    # pes construction (`evaluate_profiled_originzc_point(w, ff, pes)`, pes.nu_full fixed at
    # construction time, never re-searched) even though FamilyRegistry.jl already recorded
    # free_nu_supported=true for these two rows on the strength of the free-nu driver existing --
    # that field was true of the DRIVER, not of what this CLI actually called. Fixed here so a FULL
    # vs REDUCED origin_zc/cm_meanzc A/B through this canonical runner compares two genuinely
    # free-nu arms, matching the fix applied to `_run_full_originzc` above.
    # --threaded-gradient is REQUIRED (parse_cli already enforced this and validated it's
    # "true"/"false") -- per this repo's own no-defaults-on-any-setting rule, not just a
    # scientific-parameter concern; see parse_cli's own comment for the live bug this caused.
    threaded_gradient = cli["threaded-gradient"] == "true"

    if family in (:origin_zc, :cm_meanzc)
        return _run_reduced_zc_free_nu(family, sci, ctx, spec, pe, layout, reduced_obj0, w0, gp0, z_calib,
            delta, maxtime_real, resume_from, outdir_base, D, threaded_gradient)
    end

    fctx, evaluate_fn = if family == :unrestricted
        ev0 = Base.invokelatest(Main.evaluate_profiled_point, w0, ctx, spec, pe)
        f = Base.invokelatest(Main.build_unrestricted_family_ctx, ctx, spec, pe, ev0)
        (f, (w, ff) -> Base.invokelatest(Main.evaluate_profiled_point, w, ff.ctx, ff.spec, ff.pe))
    elseif family == :flexible_cm
        aug = Base.invokelatest(Main.build_cm_augmented_obj_archB, ctx, Main.CS; L = sci.L, contrasts = :anchored,
            base_obj = reduced_obj0, profiled_layout = layout)
        cctx = Base.invokelatest(Main.build_cm_bin_ctx, ctx, aug; profiled_layout = layout,
            inner_fg_backend = :dense_reference, threaded_bins = true)
        f = Base.invokelatest(Main.build_flexcm_family_ctx, ctx, spec, pe, layout, cctx)
        (f, (w, ff) -> Base.invokelatest(Main.evaluate_profiled_flexcm_point, w, ff))
    else # :common_frechet
        aug = Base.invokelatest(Main.build_cm_frechet_augmented_obj_archB, ctx, Main.CS; L = sci.L, contrasts = :anchored,
            base_obj = reduced_obj0, profiled_layout = layout)
        cctx = Base.invokelatest(Main.build_cm_bin_ctx, ctx, aug; profiled_layout = layout,
            inner_fg_backend = :dense_reference, threaded_bins = false,
            core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
        f = Base.invokelatest(Main.build_frechet_family_ctx, ctx, spec, pe, layout, cctx, aug.level_targets)
        (f, (w, ff) -> Base.invokelatest(Main.evaluate_profiled_frechet_point, w, ff))
    end

    # task §6.1 (2026-08-04): --a-coordinate-mode is REQUIRED for --formulation reduced
    # (parse_cli already enforced this and validated the value). Powered mode is fixed-theta only
    # (POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md §3d) and does not apply to
    # :unrestricted (theta jointly searched there) -- run_profiled_upper_constrained's own
    # validate_mode_family_compatibility raises a clear, immediate error on that combination
    # rather than silently falling back to native.
    a_coordinate_mode = Symbol(cli["a-coordinate-mode"])

    manifest_dir = joinpath(outdir_base, "reduced_$(family)_W$(sci.W)_delta$(delta)")
    mkpath(manifest_dir)
    initial_digest = RunManifestMod.digest_economic_state([gp0], z_calib, [1.0])
    rm_ = RunManifestMod.RunManifest(sci = sci, family = family, economic_parameterization = :profiled_destination_scales,
        A_coordinate_mode = a_coordinate_mode, nu_policy = :fixed, nu_bounds = nothing,
        draw_checksum_uniform = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing ? ctx.draw_meta.checksum_uniform : "",
        draw_checksum_transformed = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing ? ctx.draw_meta.checksum_transformed : "",
        outer_algorithm = :knitro_direct_sr1, outer_max_wall_seconds = maxtime_real, outer_max_gradients = 1_000_000,
        cache_policy = :none, dual_bank_policy = :none, warm_start_policy = resume_from === nothing ? :cold : :resumed,
        verification_policy = :reduced_verify_fn_inner_status_only,
        initial_state_digest = initial_digest, source_sha = RunManifestMod.current_source_sha(repo_dir = REPO_ROOT),
        source_dirty = RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT))
    RunManifestMod.write_run_manifest_json(joinpath(manifest_dir, "run_manifest.json"), rm_)
    println("[run_profiled_model] wrote $(joinpath(manifest_dir, "run_manifest.json"))"); flush(stdout)

    ckpt_path = joinpath(manifest_dir, "checkpoint.jls")
    result = Base.invokelatest(Main.run_profiled_upper_constrained, "cli_$(family)", w0; fctx, evaluate_fn,
        ctx = ctx, pe = pe, delta = delta, maxtime_real = maxtime_real, hessopt_tag = "sr1",
        a_coordinate_mode = a_coordinate_mode,
        checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = resume_from, verbose = true,
        threaded_gradient = threaded_gradient)
    println("[run_profiled_model] n_eval=$(result.n_eval) n_grad=$(result.n_grad) wall=$(round(result.wall, digits=1))s " *
        "best=$(result.best === nothing ? "none" : "gp=$(result.best.gp) Delta=$(result.best.Delta)")")
    flush(stdout)
    return result
end

# ----------------------------------------------------------------------------
# REDUCED origin_zc/cm_meanzc -- genuine free-nu dispatch (task §3.1/§3.2). Separate function
# rather than folding into `_run_reduced`'s own big if/elseif: the free-nu driver has a materially
# different call shape (extended [gp;A_free;eta_nu] outer vector, 4-arg evaluate_fn_free_nu/
# gradient_fn_free_nu, required eta_bounds, its own checkpoint writer) -- forcing it into the same
# tuple-returning branch structure as the 2-arg families would obscure exactly the parity property
# this fix exists to make visible.
# ----------------------------------------------------------------------------
function _run_reduced_zc_free_nu(family::Symbol, sci, ctx, spec, pe, layout, reduced_obj0, w0::Vector{Float64},
        gp0::Float64, z_calib::Matrix{Float64}, delta::Float64, maxtime_real::Float64, resume_from,
        outdir_base::String, D::Int, threaded_gradient::Bool)
    if family == :origin_zc
        layout_o = Base.invokelatest(Main.OriginByPowerLayout, D, 1, 0)
        νvec0 = fill(1.0, D)
        aug = Base.invokelatest(Main.build_originzc_augmented_obj, ctx, Main.CS, layout_o;
            base_obj = reduced_obj0, profiled_layout = layout)
        octx = Base.invokelatest(Main.build_originzc_core_hess_ctx, aug, ctx; core_hessian_backend = :exact_winner_pair_parallel,
            zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
        fctx = Base.invokelatest(Main.build_originzc_family_ctx, ctx, spec, pe, layout, aug)
        pes = Base.invokelatest(Main.OriginZCPointEvalState, octx, νvec0)
        evaluate_fn_free_nu = (w, eta, ff, p) -> Base.invokelatest(Main.evaluate_profiled_originzc_point, w, eta, ff, p)
        # threaded_gradient closed over here (gradient_fn_free_nu's own call signature is fixed at
        # 5 positional args by run_profiled_upper_constrained_free_nu's own cb_G!, task §6's
        # threading kwarg is passed via closure capture rather than widening that driver's contract).
        gradient_fn_free_nu = (w, eta, c, ff, ev) -> Base.invokelatest(Main.reduced_originzc_outer_gradient_with_eta, w, eta, c, ff, ev; threaded = threaded_gradient)
    else # :cm_meanzc
        K_MEAN, K_PAIR = sci.K_mean, sci.K_pair
        νvec0 = fill(1.0, max(K_MEAN, 1))
        aug = Base.invokelatest(Main.build_cm_meanzc_augmented_obj, ctx, Main.CS; L = sci.L, K_mean = K_MEAN, K_pair = K_PAIR,
            base_obj = reduced_obj0, profiled_layout = layout)
        cctx = Base.invokelatest(Main.build_cm_meanzc_bin_ctx, ctx, aug; core_hessian_backend = :exact_winner_pair_parallel,
            zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
            profiled_layout = layout)
        bins_u32 = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        fctx = Base.invokelatest(Main.build_cmzc_family_ctx, ctx, spec, pe, layout, aug, cctx, bins_u32)
        pes = Base.invokelatest(Main.CMZCPointEvalState, cctx, νvec0)
        evaluate_fn_free_nu = (w, eta, ff, p) -> Base.invokelatest(Main.evaluate_profiled_cmzc_point, w, eta, ff, p)
        gradient_fn_free_nu = (w, eta, c, ff, ev) -> Base.invokelatest(Main.reduced_cmzc_outer_gradient_with_eta, w, eta, c, ff, ev; threaded = threaded_gradient)
    end

    # `originzc_default_nu_bounds` dispatches correctly on `fctx.zc_layout`'s own type
    # (OriginByPowerLayout for origin_zc, SharedByPowerLayout for cm_meanzc -- the latter via that
    # function's own SharedByPowerLayout method, which forwards to meanzc_default_nu_bounds) --
    # ONE call site correctly serves both families, matching the FULL-side wrapper's own bounds
    # convention exactly (see `_run_full_originzc`'s header comment above).
    eta_bounds = Base.invokelatest(Main.originzc_default_nu_bounds, ctx, fctx.zc_layout)
    n_eta = length(eta_bounds)
    eta_nu0 = zeros(n_eta)   # neutral nu=1 start, matching the FULL-side wrapper's own eta0 = zeros(D)
    eta_bounds_summary = (minimum(b[1] for b in eta_bounds), maximum(b[2] for b in eta_bounds))

    manifest_dir = joinpath(outdir_base, "reduced_$(family)_W$(sci.W)_delta$(delta)")
    mkpath(manifest_dir)
    initial_digest = RunManifestMod.digest_economic_state([gp0], z_calib, [1.0])
    rm_ = RunManifestMod.RunManifest(sci = sci, family = family, economic_parameterization = :profiled_destination_scales,
        A_coordinate_mode = :profiled_pivot_anchor_relative, nu_policy = :free, nu_bounds = eta_bounds_summary,
        draw_checksum_uniform = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing ? ctx.draw_meta.checksum_uniform : "",
        draw_checksum_transformed = hasproperty(ctx, :draw_meta) && ctx.draw_meta !== nothing ? ctx.draw_meta.checksum_transformed : "",
        outer_algorithm = :knitro_direct_sr1, outer_max_wall_seconds = maxtime_real, outer_max_gradients = 1_000_000,
        cache_policy = :none, dual_bank_policy = :none, warm_start_policy = resume_from === nothing ? :cold : :resumed,
        verification_policy = :reduced_verify_fn_inner_status_only,
        initial_state_digest = initial_digest, source_sha = RunManifestMod.current_source_sha(repo_dir = REPO_ROOT),
        source_dirty = RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT))
    RunManifestMod.write_run_manifest_json(joinpath(manifest_dir, "run_manifest.json"), rm_)
    println("[run_profiled_model] wrote $(joinpath(manifest_dir, "run_manifest.json"))"); flush(stdout)

    ckpt_path = joinpath(manifest_dir, "checkpoint.jls")
    result = Base.invokelatest(Main.run_profiled_upper_constrained_free_nu, "cli_$(family)", w0, eta_nu0;
        fctx = fctx, evaluate_fn_free_nu = evaluate_fn_free_nu, gradient_fn_free_nu = gradient_fn_free_nu, pes = pes,
        ctx = ctx, pe = pe, eta_bounds = eta_bounds, delta = delta, maxtime_real = maxtime_real, hessopt_tag = "sr1",
        checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = resume_from, verbose = true)
    println("[run_profiled_model] n_eval=$(result.n_eval) n_grad=$(result.n_grad) wall=$(round(result.wall, digits=1))s " *
        "best=$(result.best === nothing ? "none" : "gp=$(result.best.gp) eta_nu=$(result.best.eta_nu) Delta=$(result.best.Delta)")")
    flush(stdout)
    return result
end

# ----------------------------------------------------------------------------
# FULL dispatch -- flexible_cm/common_frechet/cm_meanzc via run_cm_upper_checkpointed. Kept
# deliberately narrow (see file header) -- unrestricted/origin_zc raise a named error.
# ----------------------------------------------------------------------------
"""
    _run_full_unrestricted(sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)

FULL/:unrestricted, via `run_polish_checkpointed_unified` (`c10_d20_production_driver_unified.jl`).
Task §7 continuation (2026-08-04): the `OuterCoordinateLayout` the file header previously called
"not derivable from ScientificManifest alone" is traced here, NOT guessed --
`make_layout(trade_elasticity_mode=:fixed, A_coordinate_mode=:powered_aspace, gp_coordinate_mode=:raw)`
is corroborated by MULTIPLE independent real production/campaign drivers using this EXACT same
call, byte-for-byte (`campaign_unrestricted_runner.jl`, `campaign_inputs/sigma3_W500k_2026-07-30/
drivers/campaign_unrestricted_runner_sigma3.jl`, `smoke_default_flip_2026-08-01.jl`,
`test_d20_extended_release_gate_2026-07-30.jl`), and independently confirmed as FULL's real
production coordinate mode by `[[full-vs-reduced-forensic-audit-2026-08-03]]` (point 5: "REDUCED's
A-coordinate system is a third, distinct formula -- not FULL's `:powered_aspace`"). The calibration
`w_start` encoding (`reduce_to_w_unified`, `outer_coordinate_layout.jl`) mirrors
`unrestricted_stage_runner.jl`'s own `MODE="calibration"` branch exactly -- not invented.
"""
function _run_full_unrestricted(sci, cli, delta::Float64, find_smallest::Bool,
        maxtime_real::Float64, resume_from, outdir_base::String)
    # Mirrors unrestricted_stage_runner.jl's own include list EXACTLY (5 files) rather than a
    # hand-rolled enumeration -- c10_d20_production_driver.jl already transitively includes every
    # shared dependency (context/oracle/gravity_elimination/cross_delta_cache/etc.) that a
    # hand-copied list kept missing piecemeal (UndefVarError: CrossDeltaExactCache and others,
    # found live 2026-08-04); duplicating that enumeration here risks exactly the double-include
    # struct-redefinition errors this pattern avoids.
    for f in ["c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
              "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl"]
        Base.include(Main, joinpath(D4X, f))
    end

    layout = Base.invokelatest(Main.make_layout,
        trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

    manifest_dir = joinpath(outdir_base, "full_unrestricted_W$(sci.W)_delta$(delta)")
    mkpath(manifest_dir)
    rm_ = RunManifestMod.RunManifest(sci = sci, family = :unrestricted, economic_parameterization = :full_gamma_normalized,
        A_coordinate_mode = :powered_aspace, nu_policy = :fixed, nu_bounds = nothing,
        draw_checksum_uniform = "", draw_checksum_transformed = "",
        outer_algorithm = :knitro_auto, outer_max_wall_seconds = maxtime_real, outer_max_gradients = 1_000_000,
        cache_policy = :exact_cache, dual_bank_policy = :none, warm_start_policy = resume_from === nothing ? :cold : :resumed,
        verification_policy = :cm_production_value_verified,
        initial_state_digest = "not_computed_full_cli_smoke", source_sha = RunManifestMod.current_source_sha(repo_dir = REPO_ROOT),
        source_dirty = RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT))
    RunManifestMod.write_run_manifest_json(joinpath(manifest_dir, "run_manifest.json"), rm_)
    println("[run_profiled_model] wrote $(joinpath(manifest_dir, "run_manifest.json"))"); flush(stdout)

    local w_start::Vector{Float64}
    if resume_from === nothing
        ctx0 = Base.invokelatest(Main.d20_real_setup_design, W = sci.W, δ = delta, find_smallest = find_smallest,
            draw_design = sci.draw_design, draw_seed = sci.draw_seed, destination_sample = sci.destination_sample,
            exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
            σHat = sci.sigma)
        theta_star = 1.0 / ctx0.μHat
        D = ctx0.D; Ddest = ctx0.D_dest
        xy = Base.invokelatest(Main.precompute_aspace_XY, ctx0)
        pgc = Base.invokelatest(Main.build_pivot_elimination_cheap, ctx0;
            mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
        x_free_calib = ctx0.θ0_up[ctx0.free_idx]
        gp0 = x_free_calib[1]
        logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
        w_start = Base.invokelatest(Main.reduce_to_w_unified, theta_star, gp0, logA_full0, pgc, xy, layout)
    else
        # resume: w_start is a required-positional no-op placeholder whenever resume_from is set
        # (run_polish_checkpointed_unified's own convention -- mirrors unrestricted_stage_runner.jl's
        # own "resume" branch, MODE=="resume": w_start = zeros(...)). Length must still match the
        # layout's own outer_dim -- computed the same way, without needing a fresh ctx0/ONLY for sizing.
        ctx0 = Base.invokelatest(Main.d20_real_setup_design, W = sci.W, δ = delta, find_smallest = find_smallest,
            draw_design = sci.draw_design, draw_seed = sci.draw_seed, destination_sample = sci.destination_sample,
            exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
            σHat = sci.sigma)
        D = ctx0.D; Ddest = ctx0.D_dest
        w_start = zeros(Base.invokelatest(Main.outer_dim, layout, D, Ddest))
    end

    result = Base.invokelatest(Main.run_polish_checkpointed_unified, "cli_unrestricted", find_smallest, w_start;
        layout = layout, maxtime_real = maxtime_real, W_in = sci.W, delta_in = delta,
        draw_seed_in = sci.draw_seed, draw_design_in = sci.draw_design,
        ckpt_dir = manifest_dir, checkpoint_interval_s = 60.0, resume_from = resume_from,
        exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
        σHat = sci.sigma, destination_sample = sci.destination_sample)
    println("[run_profiled_model] FULL (unrestricted) run complete."); flush(stdout)
    return result
end

"""
    _run_full_originzc(sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)

FULL/origin_zc via `run_originzc_upper_checkpointed` (cm_originzc_checkpoint.jl). Include list,
`distribution_restriction=:origin_specific_moments`/`K_mean=1`/`K_pair=0`/
`power_target_layout=:origin_by_power`, `A_coordinate_mode=:legacy_z` are copied from
`run_outer_originzc_full_2026-08-02.jl` -- the real, already-existing, dedicated FULL production
script built specifically to match the REDUCED path's own `OriginByPowerLayout(D,1,0)` exactly
(confirmed: this session's own REDUCED-path free-nu/free-eta gates use the identical (1,0) pair
throughout). `run_originzc_upper_checkpointed` rebuilds its own `ctx` internally from the same
deterministic kwargs given here -- the probe `ctx1` built below is only for sizing `w0` (D,
calibration), not reused by the real driver call, matching the dedicated script's own documented
rationale for why this is safe (same fixed draw_seed => same ctx).

task §3.2 fix (profiled-outer-ab-readiness-2026-08-04): `run_outer_originzc_full_2026-08-02.jl`
ALSO pins `nu_bounds` to a ~1e-8-wide box around `log(1.0)=0` for every origin -- that script is a
fixed-nu-near-one DIAGNOSTIC convention, not FULL's actual free-nu production path, even though it
is the file this CLI wrapper's earlier construction otherwise mirrors. `run_originzc_upper_checkpointed`
itself already supports genuine free-nu search: its own `nu_bounds=nothing` default resolves to
`originzc_default_nu_bounds(ctx, layout)` (cm_originzc_checkpoint.jl, confirmed by direct read) --
the real, already-production-tested "4x safety margin on the hard finite-support interval from the
actual draws" policy, i.e. the SAME data-driven convention the REDUCED-side free-nu driver
(`run_profiled_upper_constrained_free_nu`, `profiled_zc_free_nu_production_driver_2026-08-04.jl`)
uses via the identical function. This wrapper now omits `nu_bounds` entirely (leaves it at that
default) instead of pinning it, so a canonical-runner A/B between FULL and REDUCED origin_zc
compares two arms that both genuinely search nu, not one pinned diagnostic vs one free search.
"""
function _run_full_originzc(sci, cli, delta::Float64, find_smallest::Bool,
        maxtime_real::Float64, resume_from, outdir_base::String)
    for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
              "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
              "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
              "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
              "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
              "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
              "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
              "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
              "cm_aspace_coordinate.jl", "cm_checkpoint.jl",
              "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
              "cm_originzc_config.jl", "cm_originzc_checkpoint.jl"]
        Base.include(Main, joinpath(D4X, f))
    end

    ctx1 = Base.invokelatest(Main.d20_real_setup_design, W = sci.W, δ = delta, find_smallest = find_smallest,
        draw_design = sci.draw_design, draw_seed = sci.draw_seed, destination_sample = sci.destination_sample,
        exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
        σHat = sci.sigma)
    D = ctx1.D
    # task §3.2 fix: genuine data-driven per-origin bounds (4x safety margin on the real
    # finite-support interval), NOT the pinned ~1e-8-wide box around log(1)=0 the earlier version
    # copied from the fixed-nu diagnostic script -- see this function's own header comment.
    layout_o1 = Base.invokelatest(Main.OriginByPowerLayout, D, 1, 0)
    nu_bounds = Base.invokelatest(Main.originzc_default_nu_bounds, ctx1, layout_o1)
    # manifest schema only records a single scalar (lo,hi) summary (RunManifest.nu_bounds ::
    # Union{Nothing,Tuple{Float64,Float64}}) -- the real per-origin box KNITRO actually searches
    # is the `nu_bounds` vector passed to run_originzc_upper_checkpointed below, unabbreviated.
    nu_bounds_summary = (minimum(b[1] for b in nu_bounds), maximum(b[2] for b in nu_bounds))

    manifest_dir = joinpath(outdir_base, "full_origin_zc_W$(sci.W)_delta$(delta)")
    mkpath(manifest_dir)
    rm_ = RunManifestMod.RunManifest(sci = sci, family = :origin_zc, economic_parameterization = :full_gamma_normalized,
        A_coordinate_mode = :legacy_z, nu_policy = :free, nu_bounds = nu_bounds_summary,
        draw_checksum_uniform = "", draw_checksum_transformed = "",
        outer_algorithm = :knitro_auto, outer_max_wall_seconds = maxtime_real, outer_max_gradients = 1_000_000,
        cache_policy = :exact_cache, dual_bank_policy = :none, warm_start_policy = resume_from === nothing ? :cold : :resumed,
        verification_policy = :cm_production_value_verified,
        initial_state_digest = "not_computed_full_cli_smoke", source_sha = RunManifestMod.current_source_sha(repo_dir = REPO_ROOT),
        source_dirty = RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT))
    RunManifestMod.write_run_manifest_json(joinpath(manifest_dir, "run_manifest.json"), rm_)
    println("[run_profiled_model] wrote $(joinpath(manifest_dir, "run_manifest.json"))"); flush(stdout)

    local w0::Union{Nothing,Vector{Float64}}
    if resume_from === nothing
        pe1 = Base.invokelatest(Main.build_pivot_elimination, ctx1)
        w0_econ = Base.invokelatest(Main.cm_w0_from_calibration, ctx1, pe1, :legacy_z)
        eta0 = zeros(D)   # neutral nu=1 start; the SEARCH range is now the real data-driven `nu_bounds` above, not a pinned box
        w0 = vcat(w0_econ, eta0)
    else
        w0 = nothing   # run_originzc_upper_checkpointed loads w0 from the checkpoint itself on resume
    end

    result = Base.invokelatest(Main.run_originzc_upper_checkpointed, w0;
        find_smallest = find_smallest, W = sci.W, delta = delta, draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        destination_sample = sci.destination_sample, exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells, σHat = sci.sigma,
        distribution_restriction = :origin_specific_moments, K_mean = 1, K_pair = 0,
        power_target_layout = :origin_by_power, nu_bounds = nu_bounds,
        A_coordinate_mode = :legacy_z, outer_direct_hessopt = :sr1,
        maxtime_real = maxtime_real, ckpt_dir = manifest_dir, run_id = "cli_origin_zc",
        label = "cli_origin_zc", checkpoint_interval_s = 60.0, resume_from = resume_from,
        cm_gradient_backend = :cplus, verbose = true)
    println("[run_profiled_model] FULL (origin_zc) run complete."); flush(stdout)
    return result
end

# ----------------------------------------------------------------------------
# FULL dispatch -- flexible_cm/common_frechet/cm_meanzc via run_cm_upper_checkpointed;
# unrestricted via _run_full_unrestricted; origin_zc via _run_full_originzc. All 5 canonical
# families now wired (origin_zc added 2026-08-04 continuation, see that function's header for the
# traced distribution_restriction/K_mean/K_pair resolution).
# ----------------------------------------------------------------------------
function _run_full(family::Symbol, sci, cli, delta::Float64, find_smallest::Bool,
        maxtime_real::Float64, resume_from, outdir_base::String)
    family == :unrestricted &&
        return _run_full_unrestricted(sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)
    family == :origin_zc &&
        return _run_full_originzc(sci, cli, delta, find_smallest, maxtime_real, resume_from, outdir_base)
    family in (:flexible_cm, :common_frechet, :cm_meanzc) ||
        error("run_profiled_model.jl: FULL formulation for family=:$family is not a recognized canonical family.")
    for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
              "common_marginals_moments.jl", "common_marginals_interval.jl",
              "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
              "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
              "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
              "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
              "no_dense_g_counters.jl", "economic_operator.jl",
              "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
              "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
              "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
              "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
              "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
              "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
              "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
              "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
              "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
              "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
              "country_resolve.jl", "cm_exact_cache_production.jl", "blas_thread_policy.jl", "knitro_outer_algorithm.jl",
              "production_backend_manifest.jl", "incumbent_logic.jl",
              # pre-existing gap found live 2026-08-04 (profiled-outer-ab-readiness task): FULL
              # flexible_cm/common_frechet/cm_meanzc crashed with "A_coordinate_mode=:powered_aspace
              # requires cm_aspace_coordinate.jl to be included" the first time this CLI path was
              # actually exercised for these 3 families -- run_cm_upper_checkpointed defaults to
              # :powered_aspace and needs this file for the a<->z conversion (already correctly
              # included in _run_full_originzc's own list above; this branch's list was simply
              # missing it, not a new requirement this session introduced).
              "cm_aspace_coordinate.jl"]
        Base.include(Main, joinpath(D4X, f))
    end

    marginal_restriction = family == :common_frechet ? :common_frechet : :common_flexible
    cm_extension = family == :cm_meanzc ? :cm_plus_moments : :cm_only
    meanzc_K_mean = family == :cm_meanzc ? max(sci.K_mean, 1) : 0
    meanzc_K_pair = family == :cm_meanzc ? sci.K_pair : 0

    # task §9-10 fix (profiled-outer-ab-readiness-2026-08-04): this branch never actually
    # constructed w0/probs for a fresh (non-resumed) run before -- confirmed live, the first real
    # attempt at this exact CLI path for these 3 families crashed with "w0 required for a fresh
    # (non-resumed) run" and (separately) "probs required (exact cutpoints, not re-derived from
    # L)". Both pre-existing gaps in this branch, not introduced by this task. Fixed by mirroring
    # the real production pattern (cm_production_stage_runner.jl's own MODE=="calibration" branch):
    # probs from nested_grid_sequence at this run's own L; w0 from cm_w0_from_calibration
    # (cm_aspace_coordinate.jl, just added to the include list above) at :powered_aspace --
    # run_cm_upper_checkpointed's own real default (confirmed by direct read, cm_checkpoint.jl:871)
    # -- appending a neutral eta_nu0=zeros(K_mean) start for cm_meanzc (matching this same script's
    # own _run_full_originzc `eta0 = zeros(D)` convention, not the campaign script's own
    # multi-chain-diversity factorial formula, which answers a different question).
    probs = Base.invokelatest(Main.nested_grid_sequence, [sci.L])[sci.L]
    local w0::Union{Nothing,Vector{Float64}}
    if resume_from === nothing
        ctx0 = Base.invokelatest(Main.d20_real_setup_design, W = sci.W, δ = delta, find_smallest = find_smallest,
            draw_design = sci.draw_design, draw_seed = sci.draw_seed, destination_sample = sci.destination_sample,
            exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
            σHat = sci.sigma)
        pe0 = Base.invokelatest(Main.build_pivot_elimination, ctx0)
        w0 = Base.invokelatest(Main.cm_w0_from_calibration, ctx0, pe0, :powered_aspace)
        family == :cm_meanzc && (w0 = vcat(w0, zeros(meanzc_K_mean)))
    else
        w0 = nothing   # run_cm_upper_checkpointed loads w0 from the checkpoint itself on resume
    end

    manifest_dir = joinpath(outdir_base, "full_$(family)_W$(sci.W)_delta$(delta)")
    mkpath(manifest_dir)
    rm_ = RunManifestMod.RunManifest(sci = sci, family = family, economic_parameterization = :full_gamma_normalized,
        A_coordinate_mode = :powered_aspace, nu_policy = :fixed, nu_bounds = nothing,
        draw_checksum_uniform = "", draw_checksum_transformed = "",
        outer_algorithm = :knitro_auto, outer_max_wall_seconds = maxtime_real, outer_max_gradients = 1_000_000,
        cache_policy = :exact_cache, dual_bank_policy = :none, warm_start_policy = resume_from === nothing ? :cold : :resumed,
        verification_policy = :cm_production_value_verified,
        initial_state_digest = "not_computed_full_cli_smoke", source_sha = RunManifestMod.current_source_sha(repo_dir = REPO_ROOT),
        source_dirty = RunManifestMod.source_is_dirty(repo_dir = REPO_ROOT))
    RunManifestMod.write_run_manifest_json(joinpath(manifest_dir, "run_manifest.json"), rm_)
    println("[run_profiled_model] wrote $(joinpath(manifest_dir, "run_manifest.json"))"); flush(stdout)

    result = Base.invokelatest(Main.run_cm_upper_checkpointed, w0;
        find_smallest = find_smallest, W = sci.W, delta = delta, draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        L = sci.L, contrasts = :anchored, probs = probs,
        marginal_restriction = marginal_restriction, cm_extension = cm_extension,
        meanzc_K_mean = meanzc_K_mean, meanzc_K_pair = meanzc_K_pair,
        maxtime_real = maxtime_real, ckpt_dir = manifest_dir, label = "cli_$(family)",
        resume_from = resume_from, σHat = sci.sigma, exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells, destination_sample = sci.destination_sample)
    println("[run_profiled_model] FULL run complete."); flush(stdout)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
