# cm_meanzc (CM+ZC) isolated real-KNITRO A/B gate for threaded H_EC / H_EZ, 2026-07-28.
#
# Purpose: measure the effect of the opt-in threaded H_EC/H_EZ cross-Hessian kernels
# (threaded_cross_hessian.jl, gated by the single coupled Ref pair
# CROSS_HESSIAN_THREADED_DEFAULT[]/CROSS_HESSIAN_WORKERS_DEFAULT[]) on REAL complete-inner-solve
# wall time for cm_meanzc, run in FULL ISOLATION (no other concurrent KNITRO process on this host)
# per this task's own operational rules -- cm_meanzc's real KNITRO solve is confirmed flaky under
# concurrent host load (host-level resource contention, not a code bug -- see
# docs/HANDOVER_NOTE_2026-07-28.md) but reliable when run alone.
#
# Structure/template credit: file-include list, d20_real_setup/cmzc_w0/run_cm_upper_checkpointed
# call pattern, and the CMZC_LIVE_PCX_STASH live-handle pattern are all reused verbatim (not
# re-derived) from full_aod_diag/d4_exact/diag_cmzc_originzc_d20_profile_2026-07-28.jl and
# cross_hessian_live_stash_2026-07-28.jl in this same worktree.
#
# IMPORTANT (checked live, see grep evidence in this session's report): CROSS_HESSIAN_THREADED_DEFAULT[]
# and CROSS_HESSIAN_WORKERS_DEFAULT[] are a SINGLE coupled Ref pair -- both H_EC
# (winner_pair_cross_hessian_fill_threaded!, cm_hessian_threaded.jl:236-237) and H_EZ
# (winner_pair_cross_hessian_zc_block_threaded!, cm_hessian_architectures.jl:831-832) read the SAME
# cctx.cross_hessian_threaded/cctx.cross_hessian_workers fields, populated from the SAME global Ref
# pair at context-construction time (build_cm_meanzc_production_context -> build_cm_bin_ctx-style
# defaults). There is no independent per-block Ref anywhere in the tree (grepped
# cm_hessian_architectures.jl/cm_hessian_threaded.jl/threaded_cross_hessian.jl/
# cm_meanzc_production.jl for a second Ref{Bool}/Ref{Int} near these call sites -- none found).
# Combos 2 ("threaded H_EC + serial H_EZ") and 3 ("serial H_EC + threaded H_EZ") from the task
# brief are therefore NOT INDEPENDENTLY CONTROLLABLE through this codebase's real toggle surface --
# they are recorded as SKIPPED rows below with an explicit note, per the brief's own "skip and note
# why" instruction. Only combo 1 (serial/serial) and combo 4 (threaded/threaded) are real,
# distinct, runnable configurations.
#
# Usage (run from this file's directory, sequentially, nothing else concurrent):
#   source ../../.knitro_env.sh
#   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
#   julia --project=<worktree-root> -t 20 cmzc_isolated_hec_hez_gate_2026-07-28.jl

const D4X = @__DIR__
cd(D4X)
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
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

const NT = Threads.nthreads()
lp("Threads.nthreads() = ", NT, "  BLAS default threads = ", BLAS.get_num_threads())

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "cmzc_isolated_hec_hez_gate_2026-07-28")
mkpath(OUTROOT)

# ================================================================================================
# w0 construction -- verbatim from diag_cmzc_originzc_d20_profile_2026-07-28.jl's cmzc_w0.
# ================================================================================================
function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

lp("\n", "="^100, "\n=== cm_meanzc: building ctx0 for w0 construction ===")
const ctx0c = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
const pe0c = build_pivot_elimination(ctx0c)
const theta0c = cm_fixed_theta(ctx0c)
const xy0c = precompute_cm_aspace_xy(ctx0c)
const W0_CALIB = cmzc_w0(ctx0c, pe0c, theta0c, xy0c)
const RNGC = MersenneTwister(2026_0729)
const W0_NONCALIB = W0_CALIB .* (1.0 .+ 1e-2 .* (rand(RNGC, length(W0_CALIB)) .- 0.5))
lp("w0 dims: ", length(W0_CALIB))

# ================================================================================================
# Combos. Only 1 ("serial_serial") and 4 ("threaded_threaded") are real distinct configurations --
# see header note. 2/3 are recorded as SKIPPED rows.
# ================================================================================================
struct Combo
    key::String
    ec_threaded::Bool   # nominal intent (both fields are actually driven by ONE coupled Ref pair)
    ez_threaded::Bool
    runnable::Bool
    skip_reason::String
end
const COMBOS = [
    Combo("serial_serial",      false, false, true,  ""),
    Combo("threaded_ec_serial_ez", true,  false, false,
        "not independently controllable: CROSS_HESSIAN_THREADED_DEFAULT[]/WORKERS_DEFAULT[] is a single Ref pair driving BOTH H_EC (cm_hessian_threaded.jl:236-237) and H_EZ (cm_hessian_architectures.jl:831-832) -- no separate H_EC-only Ref exists in this codebase."),
    Combo("serial_ec_threaded_ez", false, true, false,
        "not independently controllable: same coupled-Ref reason as combo 2 (H_EZ-only Ref does not exist)."),
    Combo("threaded_threaded",  true,  true,  true,  ""),
]

# ================================================================================================
# Driver call, matching the task brief's exact call signature (reused from the template's
# run_cmzc_driver, extended with pre-call Ref toggling + post-call Hfull checksum extraction).
# ================================================================================================
function set_cross_hessian_refs!(threaded::Bool, workers::Int)
    CROSS_HESSIAN_THREADED_DEFAULT[] = threaded
    CROSS_HESSIAN_WORKERS_DEFAULT[] = workers
end

function run_cmzc_driver(w0::Vector{Float64}, maxtime_real::Float64, run_id::String)
    CMZC_LIVE_PCX_STASH[] = nothing
    out = joinpath(OUTROOT, "cmzc_$(run_id)_t$(NT)")
    rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    result = run_cm_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
        probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = out, run_id = "cmzc_$(run_id)_t$(NT)", label = "cmzc_$(run_id)_t$(NT)",
        checkpoint_interval_s = 3600.0, maxtime_real = maxtime_real, verbose = true)
    wall = time() - t0
    @printf("  [DRIVER cm_meanzc/%s] wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
        run_id, wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
    flush(stdout)
    handle = CMZC_LIVE_PCX_STASH[]
    return (handle = handle, result = result, wall = wall)
end

"Recompute H_EC and H_EZ blocks BOTH ways (serial + threaded) from the SAME live captured state
and return (maxdiff_combined, hfull_norm, note). Trajectory-independent correctness check: valid
regardless of which combo's Ref setting actually produced the captured state, since it always
compares serial-vs-threaded kernel OUTPUT at that one fixed input."
function hessian_cross_check(handle)
    handle === nothing && return (NaN, NaN, "no live handle captured (maxtime_real too short for a real callback)")
    ctx_cm = handle.ctx_cm; cctx = handle.cctx; obj_z = ctx_cm.obj
    w_z = copy(obj_z.arg2); M_z = obj_z.M
    NCORE_z = cctx.NCORE; ncore_core_z = cctx.ncore_core; Lb_z = cctx.L
    normS = norm(w_z)
    if !(isfinite(normS) && normS > 0)
        return (NaN, NaN, "obj.arg2 (w) uninitialized -- no real Hessian callback fired in this budget")
    end
    hfull_norm = norm(cctx.Hfull)

    # --- H_EZ (=HEM) serial vs threaded ---
    wctx_z = serial_ctx(cctx.core_ws)
    op_z = cctx.hzz_zc_op
    refresh_zc_targets!(cctx.hzz_zc_ws, op_z, cctx.hzz_zc_layout, cctx.nu_ref[])
    cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op_z, size(w_z, 1))
    refresh_zc_centered!(cctx.hzz_centered, op_z, cctx.hzz_zc_ws, w_z)
    n_restr_z = NCORE_z - ncore_core_z
    cctx.zc_cross_scratch = _ensure_zc_cross_scratch!(cctx, wctx_z.W, n_restr_z)
    winner_pair_cross_hessian_zc_prep!(cctx.zc_cross_scratch, wctx_z, w_z)
    nx_z = n_restriction(op_z)
    Zview_z = @view cctx.hzz_centered.Zc[:, 1:nx_z]
    HEM_serial = Matrix{Float64}(undef, wctx_z.ncolI + 1, nx_z)
    winner_pair_cross_hessian_zc_block!(HEM_serial, wctx_z, cctx.zc_cross_scratch, w_z, Zview_z, M_z)
    HEM_threaded = copy(HEM_serial)
    winner_pair_cross_hessian_zc_block_threaded!(HEM_threaded, wctx_z, cctx.zc_cross_scratch, w_z, Zview_z, M_z; workers = NT)
    maxdiff_ez = maximum(abs.(HEM_threaded .- HEM_serial))

    # --- H_EC prep (crossprep) serial vs threaded ---
    cross_ws_z = _ensure_cm_cross_scratch!(cctx, wctx_z.ncolI, cctx.D, Lb_z)
    winner_pair_cross_hessian_fill!(wctx_z, cross_ws_z, obj_z, cctx.Bidx)
    QCScum_serial = copy(cross_ws_z.QCScum)
    winner_pair_cross_hessian_fill_threaded!(wctx_z, cross_ws_z, obj_z, cctx.Bidx; workers = NT)
    maxdiff_ec = maximum(abs.(cross_ws_z.QCScum .- QCScum_serial))

    maxdiff = max(maxdiff_ec, maxdiff_ez)
    note = @sprintf("H_EC_maxdiff=%.3e H_EZ_maxdiff=%.3e Hfull_norm=%.6e", maxdiff_ec, maxdiff_ez, hfull_norm)
    return (maxdiff, hfull_norm, note)
end

# ================================================================================================
# Orchestration: 3 points x runnable combos, strictly sequential.
# ================================================================================================
const CALIB_BUDGET = 20.0
const NONCALIB_BUDGET = 20.0
const TRAJ_BUDGET = 90.0

const POINTS = [
    ("calibration", W0_CALIB, CALIB_BUDGET),
    ("non_calibration", W0_NONCALIB, NONCALIB_BUDGET),
    ("solver_trajectory", W0_CALIB, TRAJ_BUDGET),
]

rows = NamedTuple[]   # point,combo,wall_s,knitro_status,n_eval,n_grad,kappa,hessian_check_maxdiff_vs_serial,notes
function record!(point, combo, wall_s, knitro_status, n_eval, n_grad, kappa, maxdiff, notes)
    push!(rows, (point = point, combo = combo, wall_s = wall_s, knitro_status = knitro_status,
        n_eval = n_eval, n_grad = n_grad, kappa = kappa, hessian_check_maxdiff_vs_serial = maxdiff, notes = notes))
end

for (point_label, w0, budget) in POINTS
    global rows
    for combo in COMBOS
        if !combo.runnable
            lp("\n--- SKIP point=", point_label, " combo=", combo.key, " -- ", combo.skip_reason)
            record!(point_label, combo.key, NaN, "SKIPPED", -1, -1, "n/a", NaN, combo.skip_reason)
            continue
        end
        workers_for_combo = combo.ec_threaded || combo.ez_threaded ? min(NT, 20) : 1
        set_cross_hessian_refs!(combo.ec_threaded, workers_for_combo)
        lp("\n", "="^100, "\n=== point=", point_label, " combo=", combo.key,
           " CROSS_HESSIAN_THREADED_DEFAULT[]=", CROSS_HESSIAN_THREADED_DEFAULT[],
           " CROSS_HESSIAN_WORKERS_DEFAULT[]=", CROSS_HESSIAN_WORKERS_DEFAULT[],
           " maxtime_real=", budget)
        run_id = "$(point_label)_$(combo.key)"
        local run, status_str, notes, maxdiff
        try
            run = run_cmzc_driver(w0, budget, run_id)
        catch e
            msg = sprint(showerror, e)
            lp("!!! DRIVER FAILED point=", point_label, " combo=", combo.key, " -- ", msg)
            record!(point_label, combo.key, NaN, "DRIVER_ERROR", -1, -1, "n/a", NaN, msg[1:min(end, 500)])
            continue
        end
        r = run.result
        status_str = string(r.knitro_status)
        kappa_str = hasproperty(r, :kappa) ? string(r.kappa) : "n/a"
        try
            maxdiff, hfull_norm, note = hessian_cross_check(run.handle)
            notes = note
        catch e
            maxdiff = NaN
            notes = "hessian_cross_check FAILED: " * sprint(showerror, e)[1:min(end, 300)]
        end
        record!(point_label, combo.key, run.wall, status_str, r.n_eval, r.n_grad, kappa_str, maxdiff, notes)
    end
end

# ================================================================================================
# Write CSV.
# ================================================================================================
csvpath = joinpath(D4X, "..", "..", "docs", "CM_MEANZC_HEC_HEZ_ISOLATED_GATE_2026-07-28.csv")
mkpath(dirname(csvpath))
open(csvpath, "w") do io
    println(io, "point,combo,wall_s,knitro_status,n_eval,n_grad,kappa,hessian_check_maxdiff_vs_serial,notes")
    for r in rows
        notes_escaped = replace(r.notes, "\"" => "'", "\n" => " ")
        println(io, "$(r.point),$(r.combo),$(r.wall_s),$(r.knitro_status),$(r.n_eval),$(r.n_grad),$(r.kappa),$(r.hessian_check_maxdiff_vs_serial),\"$(notes_escaped)\"")
    end
end
lp("Wrote ", csvpath)
lp("DONE.")
