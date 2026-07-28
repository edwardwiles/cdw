# D=20 profiling task (flexible_cm / common_frechet), 2026-07-28: real D=20/Ddest=19/W=100,000
# sub-block Hessian profile + threaded-H_EC correctness gate, driven ENTIRELY through the real
# public checkpointed driver (`run_cm_upper_checkpointed`, cm_checkpoint.jl) -- per this task's own
# spec and the master report's own §5 (KN_RC_CALLBACK_ERR trap for a direct low-level entry point
# at real D=20 for these two families), NO low-level entry point (`archC_verified_state`/
# `archC_base_state`/`archC_frechet_*`) is ever called directly. Sub-block timing and the correctness
# gate both operate on state reached ONLY via this one confirmed-working KNITRO entry point:
#   - Sub-block timing: `cm_hessian_subblock_profiling.jl`'s opt-in `@cmhess_prof` labels, wired
#     inside the SAME shared `hessian_cm_structured_v2!`/`_fill_frechet_level_blocks!`/
#     `_prep_dual_index_for_archC!` functions the real KNITRO Hessian callback calls -- fires
#     naturally, at real solver-visited points, for the whole duration of the driver call.
#   - Correctness gate: reaches into the live `(cctx, obj)` state `run_cm_upper_checkpointed` itself
#     stashes (`CM_LIVE_PCX_STASH[]`) the instant it is built, and re-invokes the SAME packed-
#     Hessian-producing function directly (a plain Julia function call, not a second KNITRO
#     invocation) at REAL captured dual-solve points (`CM_HESSIAN_CAPTURED_X`, one entry per real
#     Hessian callback KNITRO issued during the run) under different `cross_hessian_threaded`/
#     `cross_hessian_workers` settings, comparing the resulting packed Hessian vectors.
#
# Usage: PATH="$HOME/.juliaup/bin:$PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#   julia --project=. -t 20 profile_flexcm_frechet_d20_2026-07-28.jl <FAMILY> <MODE> [MAXTIME]
#   FAMILY: flexible_cm | common_frechet
#   MODE:   threaded (production-candidate config: cross_hessian_threaded=true, workers=20;
#           profiling+capture ON; runs the offline worker-sweep correctness gate afterward) |
#           serial   (cross_hessian_threaded=false, the pre-existing default; profiling+capture ON
#           too, at no cost since default off; used ONLY for the solver-behavior-invariance check,
#           compared against a separate `threaded`-mode run's own recorded KNITRO status/n_eval/n_grad/kappa)
# One family+mode per process (KNITRO cross-run flakiness precedent in this repo, see
# diag_subblock_profile_2026-07-28.jl's own header) -- run this script 4 times total
# (flexible_cm x {threaded,serial}, common_frechet x {threaded,serial}).
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = length(ARGS) >= 1 ? ARGS[1] : error("usage: profile_flexcm_frechet_d20_2026-07-28.jl <flexible_cm|common_frechet> <threaded|serial> [maxtime]")
const MODE = length(ARGS) >= 2 ? ARGS[2] : error("usage: ... <FAMILY> <threaded|serial> [maxtime]")
FAMILY in ("flexible_cm", "common_frechet") || error("FAMILY must be flexible_cm or common_frechet, got $FAMILY")
MODE in ("threaded", "serial") || error("MODE must be threaded or serial, got $MODE")
const MAXTIME = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : (MODE == "threaded" ? 90.0 : 60.0)
const MARGINAL_RESTRICTION = FAMILY == "flexible_cm" ? :common_flexible : :common_frechet

lp("=== profile_flexcm_frechet_d20_2026-07-28: FAMILY=$FAMILY MODE=$MODE MAXTIME=$MAXTIME Threads.nthreads()=$(Threads.nthreads()) ===")
flush(stdout)

const W = 100_000
const DELTA = 1.0
const OUT = joinpath(_D4E, "..", "..", "results", "profile_flexcm_frechet_2026-07-28", "$(FAMILY)_$(MODE)")
rm(OUT; force = true, recursive = true); mkpath(OUT)

# ---- profiling/capture switches (this task's own instrumentation, cm_hessian_subblock_profiling.jl) ----
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
reset_cm_hessian_capture!()
prof_reset!()

# ---- threaded H_EC backend selection (threaded_cross_hessian.jl globals) -- read by
# build_cm_production_context/build_cm_frechet_production_context (via build_cm_bin_ctx) at
# construction time, i.e. must be set BEFORE run_cm_upper_checkpointed is called. ----
const GATE_WORKERS = [1, 4, 8, 10, 20]
if MODE == "threaded"
    CROSS_HESSIAN_THREADED_DEFAULT[] = true
    CROSS_HESSIAN_WORKERS_DEFAULT[] = 20
    Threads.nthreads() >= 20 || lp("WARNING: Threads.nthreads()=$(Threads.nthreads()) < 20 -- the workers=20 arm of the correctness gate and the workers=20 production-candidate profile point will be capped/degenerate. Launch with -t 20.")
else
    CROSS_HESSIAN_THREADED_DEFAULT[] = false
end
lp("CROSS_HESSIAN_THREADED_DEFAULT[]=", CROSS_HESSIAN_THREADED_DEFAULT[], " CROSS_HESSIAN_WORKERS_DEFAULT[]=", CROSS_HESSIAN_WORKERS_DEFAULT[])

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
D = ctx0.D; Ddest = ctx0.D_dest
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
pe0 = build_pivot_elimination(ctx0)
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)

const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]
lp("D=", D, " Ddest=", Ddest)
flush(stdout)

t0 = time()
result = run_cm_upper_checkpointed(copy(w_a_calib);
    W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    cm_hessian_backend = :structured, marginal_restriction = MARGINAL_RESTRICTION, cm_gradient_backend = :cplus,
    ckpt_dir = OUT, run_id = "$(FAMILY)_$(MODE)", label = "$(FAMILY)_$(MODE)",
    checkpoint_interval_s = 60.0, maxtime_real = MAXTIME, verbose = true)
wall = time() - t0

lp("="^90)
@printf("RESULT %s(%s): wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
    FAMILY, MODE, wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
lp("n_hessian_callbacks_captured = ", length(CM_HESSIAN_CAPTURED_X))
lp("best = ", result.best === nothing ? "nothing" : string(result.best))
lp("="^90)

# ---- write the solver-status summary every mode produces (used for the on/off comparison) ----
open(joinpath(OUT, "solver_status.txt"), "w") do io
    println(io, "family=$FAMILY mode=$MODE wall=$wall knitro_status=$(result.knitro_status) n_eval=$(result.n_eval) n_grad=$(result.n_grad) kappa=$(result.kappa)")
    println(io, "best=$(result.best)")
    println(io, "n_hessian_callbacks_captured=$(length(CM_HESSIAN_CAPTURED_X))")
end

# ---- sub-block timing profile (mirrors docs/key_results/cross_hessian_subblock_profile_t*_origin_zc_2026-07-28.csv schema) ----
subblock_labels = ["hessw_operator_prep", "ddpsi", "misc_bookkeeping", "H_EE", "bintables_prep",
                    "H_EC_prep", "H_EC_asm", "H_CZ_prep", "H_CC", "level_table_prep", "H_EF", "H_CF", "H_FF", "packing"]
coarse_labels = ["inner_dual_hessian_callback_archC", "inner_dual_hessian_callback_archC_frechet", "inner_knitro_dual_solve_arch"]
rows = NamedTuple[]
lp("\n--- sub-block profile (family=$FAMILY, mode=$MODE, nthreads=$(Threads.nthreads()), cross_hessian_workers=$(CROSS_HESSIAN_WORKERS_DEFAULT[])) ---")
for label in vcat(subblock_labels, coarse_labels)
    haskey(PROF_TIMES, label) || continue
    times = PROF_TIMES[label]
    isempty(times) && continue
    tmin = minimum(times); tmean = sum(times) / length(times)
    push!(rows, (family = FAMILY, block = label, nthreads = Threads.nthreads(), t_min_s = tmin, t_mean_s = tmean))
    @printf("  [%-15s] %-32s n=%-5d min=%.5fs  mean=%.5fs\n", FAMILY, label, length(times), tmin, tmean)
end
flush(stdout)

outcsv = joinpath(_D4E, "..", "..", "results", "profile_flexcm_frechet_2026-07-28", "subblock_profile_$(FAMILY)_$(MODE)_2026-07-28.csv")
mkpath(dirname(outcsv))
open(outcsv, "w") do io
    println(io, "family,block,nthreads,t_min_s,t_mean_s")
    for r in rows
        println(io, "$(r.family),$(r.block),$(r.nthreads),$(r.t_min_s),$(r.t_mean_s)")
    end
end
lp("Wrote $outcsv")

# ============================================================================================
# Correctness gate (MODE=="threaded" only, needs CM_LIVE_PCX_STASH[] + captured x points): packed
# Hessian bit-exactness of threaded H_EC (workers in GATE_WORKERS) vs the serial reference
# (cross_hessian_threaded=false), at calibration + non-calibration + solver-derived points -- all
# via direct calls to the SAME hessian_cm_structured_v2! the real KNITRO Hessian callback itself
# calls, on the ALREADY-warmed live (cctx,obj) state, at REAL captured dual-solve points. No second
# KNITRO invocation anywhere in this section.
# ============================================================================================
if MODE == "threaded"
    pcx = CM_LIVE_PCX_STASH[]
    if pcx === nothing
        lp("GATE SKIPPED: CM_LIVE_PCX_STASH[] is nothing -- run_cm_upper_checkpointed never reached the stash point (unexpected).")
    else
        cctx = pcx.cctx
        obj = pcx.ctx_cm.obj
        extension = FAMILY == "common_frechet" ? _resolve_frechet_ext!(cctx, pcx.aug.level_targets) : nothing
        npts = length(CM_HESSIAN_CAPTURED_X)
        lp("\n--- correctness gate (family=$FAMILY): $npts captured dual-solve points available ---")
        if npts == 0
            lp("GATE SKIPPED: no Hessian callbacks were captured (unexpected for a real KNITRO solve).")
        else
            # calibration (first callback of the run) + up to 3 more spread across the buffer
            # (>=2 non-calibration + >=1 late/"solver-derived" point, per task spec).
            idxs = unique(clamp.(round.(Int, [1, 0.25npts, 0.6npts, npts]), 1, npts))
            n = cctx.NCORE + cctx.ncm
            npacked = n * (n + 1) ÷ 2
            href = Vector{Float64}(undef, npacked)
            hcmp = Vector{Float64}(undef, npacked)
            gate_rows = NamedTuple[]
            for (pi, idx) in enumerate(idxs)
                xsnap = CM_HESSIAN_CAPTURED_X[idx]
                ptag = idx == 1 ? "calibration(idx=1)" : (idx == npts ? "solver_derived_late(idx=$idx/$npts)" : "non_calibration(idx=$idx/$npts)")
                cctx.cross_hessian_threaded = false
                _prep_dual_index_for_archC!(cctx, obj, xsnap)
                hessian_cm_structured_v2!(href, obj, cctx, extension; threaded_bins = true, tls = cctx.tls, use_syrk = true)
                for wk in GATE_WORKERS
                    wk_eff = min(wk, Threads.nthreads())
                    cctx.cross_hessian_threaded = true
                    cctx.cross_hessian_workers = wk_eff
                    _prep_dual_index_for_archC!(cctx, obj, xsnap)
                    hessian_cm_structured_v2!(hcmp, obj, cctx, extension; threaded_bins = true, tls = cctx.tls, use_syrk = true)
                    maxdiff = maximum(abs.(hcmp .- href))
                    pass = maxdiff < 1e-9
                    push!(gate_rows, (family = FAMILY, point = ptag, workers_requested = wk, workers_effective = wk_eff, maxdiff = maxdiff, pass = pass))
                    @printf("  [GATE] %-30s workers=%-3d(eff=%-3d) maxdiff=%.3e %s\n", ptag, wk, wk_eff, maxdiff, pass ? "PASS" : "FAIL")
                end
            end
            flush(stdout)
            gatecsv = joinpath(_D4E, "..", "..", "results", "profile_flexcm_frechet_2026-07-28", "correctness_gate_$(FAMILY)_2026-07-28.csv")
            open(gatecsv, "w") do io
                println(io, "family,point,workers_requested,workers_effective,maxdiff,pass")
                for r in gate_rows
                    println(io, "$(r.family),$(r.point),$(r.workers_requested),$(r.workers_effective),$(r.maxdiff),$(r.pass)")
                end
            end
            lp("Wrote $gatecsv")
            allpass = all(r.pass for r in gate_rows)
            lp("GATE OVERALL: ", allpass ? "ALL PASS ($(length(gate_rows)) checks)" : "SOME FAIL -- see $gatecsv")
            # restore cctx to the production-candidate setting used for the rest of this run
            cctx.cross_hessian_threaded = true
            cctx.cross_hessian_workers = 20
        end
    end
end

lp("=== DONE FAMILY=$FAMILY MODE=$MODE ===")
