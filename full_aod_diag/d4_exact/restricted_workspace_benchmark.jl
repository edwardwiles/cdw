# ============================================================================
# Phase B (2026-07-24) B8: before/after benchmark, real D=20/W=80,000/L=50.
#
# Rescoped per docs/CURRENT_CM_BASIS_AND_STORAGE_AUDIT_2026-07-24.md: flexible
# CM's own CM-grid moment construction/Hessian ALREADY has the Architecture
# B/C treatment in production (this script re-measures it only as a sanity
# check/reference point, NOT as a "before" for a change made in this task).
# The actual "before" vs "after" comparison this script exists to produce is
# for CM+mean/ZC and origin-ZC's mean/pair block:
#   BEFORE = wrap_moments_with_cm_meanzc_dense / wrap_moments_with_originzc_dense
#            (fresh G_tmp alloc every call, mean/pair centered block built via
#            an allocating temporary then copied into G)
#   AFTER  = wrap_moments_with_cm_meanzc / wrap_moments_with_originzc (current
#            production default after this task's edit): G_tmp cached across
#            calls, mean/pair centered block written directly into the G view
#            with zero intermediate allocation.
#
# Usage: julia --project=. full_aod_diag/d4_exact/phaseB_b8_benchmark.jl
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_basis_diagnostic.jl"))
using Printf, LinearAlgebra, Statistics, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const W = 80_000
const L = 50
const DRAW_SEED = 20260719
const CONTRASTS = :orthonormal
const N_REPS = 5

report_cm_basis_diagnostic()

lp(">>> [", now(), "] building D=20/W=", W, " real-data context (draw_seed=", DRAW_SEED, ")...")
t_ctx = @elapsed ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED)
lp(">>> context built in ", round(t_ctx, digits = 1), "s. D=", ctx.D, " bi=", ctx.bi)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]
x_free_calib = ctx.θ0_up[ctx.free_idx]

# Point B: same cold-verified CM incumbent d20_meanzc_release_gates.jl uses, if present.
const SEED_PATH = "/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/cold_verified_seed.jls"
points = Dict{String,Vector{Float64}}("A_calibration" => x_free_calib)
try
    if isfile(SEED_PATH)
        pe_tmp = build_pivot_elimination(ctx)
        seedB = deserialize(SEED_PATH)
        if seedB.W == W && seedB.draw_seed == DRAW_SEED && seedB.cm_L == L && seedB.contrasts == CONTRASTS
            points["B_cm_incumbent_delta1"] = x_free_from_w(seedB.w, pe_tmp)
            lp(">>> Point B loaded from cold-verified seed.")
        else
            lp(">>> Point B seed provenance mismatch -- skipping, using Point A only.")
        end
    else
        lp(">>> No Point B seed file found at ", SEED_PATH, " -- using Point A only.")
    end
catch e
    lp(">>> Point B load failed (", sprint(showerror, e), ") -- using Point A only. Not fatal: B8's headline",
       " comparison only needs one valid feasible point per family.")
end

rows = NamedTuple[]

function summarize_and_record!(rows, family, variant, point_label, W_actual, wall_times, deltas, peakrss_kb)
    prof_rows = prof_summary()
    total_moment = 0.0; total_fg = 0.0; total_hess = 0.0; total_solve = 0.0
    n_fg = 0; n_hess = 0; alloc_moment = 0.0
    for r in prof_rows
        if r.label == "inner_moment_build"
            total_moment = r.total_alloc_bytes > 0 || r.n > 0 ? sum(PROF_TIMES["inner_moment_build"]) : 0.0
            alloc_moment = sum(PROF_ALLOCS["inner_moment_build"])
            n_fg = r.n
        elseif r.label in ("inner_dual_fg_callback",)
            total_fg = sum(PROF_TIMES[r.label])
        elseif r.label in ("inner_dual_hessian_callback_archC", "inner_dual_hessian_callback")
            total_hess += sum(PROF_TIMES[r.label])
            n_hess += r.n
        elseif r.label in ("inner_knitro_dual_solve_arch", "inner_knitro_dual_solve")
            total_solve = sum(PROF_TIMES[r.label])
        end
    end
    push!(rows, (family = family, variant = variant, point = point_label, W = W_actual, n_reps = length(wall_times),
        median_wall_s = median(wall_times), min_wall_s = minimum(wall_times), max_wall_s = maximum(wall_times),
        total_moment_build_s = total_moment, moment_build_alloc_bytes = alloc_moment, n_moment_calls = n_fg,
        total_fg_s = total_fg, total_hess_s = total_hess, n_hess_calls = n_hess, total_knitro_solve_s = total_solve,
        Delta_dual_min = minimum(deltas), Delta_dual_max = maximum(deltas), peak_rss_kb = peakrss_kb))
    lp(@sprintf("  [%-10s/%-9s/%-22s] median_wall=%.3fs  moment_build=%.4fs (n=%d, %.2f MB)  fg=%.4fs  hess=%.4fs  knitro_solve=%.3fs  Delta=%.6g..%.6g",
        family, variant, point_label, median(wall_times), total_moment, n_fg, alloc_moment/1e6, total_fg, total_hess, total_solve,
        minimum(deltas), maximum(deltas)))
    return nothing
end

function peak_rss_kb()
    try
        return parse(Int, split(read(`grep VmHWM /proc/$(getpid())/status`, String))[2])
    catch
        return -1
    end
end

# ---------------------------------------------------------------------------
# Family 1: flexible CM (reference point -- Architecture A vs B, NOT part of
# this task's change; re-measured only to confirm the B1 audit's claim that
# archB is already the production default and already cheap).
# ---------------------------------------------------------------------------
for (plabel, xfree) in points
    for (variant, use_archB) in (("archA_dense", false), ("archB_cached", true))
        pcx = build_cm_production_context(ctx, CS; L = L, contrasts = CONTRASTS, probs = probs, use_archB_moments = use_archB)
        prof_reset!()
        wall = Float64[]; deltas = Float64[]
        for _ in 1:N_REPS
            t = @elapsed (base, verify) = archC_verified_state(xfree, pcx.ctx_cm, pcx.cctx)
            push!(wall, t); push!(deltas, verify.Delta_dual)
        end
        summarize_and_record!(rows, "flexCM", variant, plabel, size(ctx.U,1), wall, deltas, peak_rss_kb())
    end
end

# ---------------------------------------------------------------------------
# Family 2: CM+mean/ZC, K_mean=1, K_pair=1 (matches production release-gate config)
# ---------------------------------------------------------------------------
nu0_meanzc = [1.0]
for (plabel, xfree) in points
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 1, contrasts = CONTRASTS,
        meanzc_basis = :direct, probs = probs)
    for (variant, dense) in (("dense", true), ("cached", false))
        local obj_use
        if dense
            moments_dense! = wrap_moments_with_cm_meanzc_dense(ctx.obj.moments!, aug.ncore_econ, aug.CM,
                aug.Zraw_all, aug.Zpairraw_all; meanzc_basis = aug.meanzc_basis, refIndex1 = aug.refIndex1)
            obj0 = aug.obj_cm
            obj_use = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
                γ = obj0.γ, (moments!) = moments_dense!, moments_jacobian! = error,
                d = obj0.d, outer_constr_index = obj0.outer_constr_index,
                inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
                l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
                use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
                outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
                needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
        else
            obj_use = aug.obj_cm
        end
        ctx_cm = merge(ctx, (obj = obj_use,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        prof_reset!()
        wall = Float64[]; deltas = Float64[]
        for _ in 1:N_REPS
            t = @elapsed (base, verify) = archC_meanzc_verified_state(xfree, nu0_meanzc, ctx_cm, cctx)
            push!(wall, t); push!(deltas, verify.Delta_dual)
        end
        summarize_and_record!(rows, "CM+meanZC", variant, plabel, size(ctx.U,1), wall, deltas, peak_rss_kb())
    end
end

# ---------------------------------------------------------------------------
# Family 3: origin-ZC, K_mean=1, K_pair=1 (SharedByPowerLayout, matches K=1 production config)
# ---------------------------------------------------------------------------
layout1 = SharedByPowerLayout(1, 1)
nu0_oz = [mean(ctx.U .^ k) for k in 1:layout1.K_mean]   # feasible: literal data mean, matches d20_originzc_fixedpoint_gates.jl's own init convention
for (plabel, xfree) in points
    aug = build_originzc_augmented_obj(ctx, CS, layout1)
    for (variant, dense) in (("dense", true), ("cached", false))
        local obj_use
        if dense
            moments_dense! = wrap_moments_with_originzc_dense(ctx.obj.moments!, aug.ncore_econ, aug.Zraw_all, aug.Zpairraw_all, layout1)
            obj0 = aug.obj_cm
            obj_use = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
                γ = obj0.γ, (moments!) = moments_dense!, moments_jacobian! = error,
                d = obj0.d, outer_constr_index = obj0.outer_constr_index,
                inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
                l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
                use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
                outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
                needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
        else
            obj_use = aug.obj_cm
        end
        ctx_cm = merge(ctx, (obj = obj_use,))
        prof_reset!()
        wall = Float64[]; deltas = Float64[]
        for _ in 1:N_REPS
            t = @elapsed (base, verify) = archOZ_verified_state(xfree, nu0_oz, ctx_cm)
            push!(wall, t); push!(deltas, verify.Delta_dual)
        end
        summarize_and_record!(rows, "originZC", variant, plabel, size(ctx.U,1), wall, deltas, peak_rss_kb())
    end
end

lp()
lp("="^100)
lp("SUMMARY TABLE")
lp("="^100)
write_csv_rows(joinpath(@__DIR__, "..", "..", "docs", "phaseB_b8_benchmark_raw_2026-07-24.csv"), rows)
for r in rows
    lp(r)
end
lp(">>> DONE at ", now())
