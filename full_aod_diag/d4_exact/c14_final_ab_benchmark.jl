# Continuation 14, Task 5: ONE authoritative CM/no-CM A/B benchmark, same commit/KNITRO
# version(13.0.1)/draw set throughout (seed 20260719, matching every other c14 script and the
# canonical-rerun checkpoint this reuses). Follows timing_harness.jl's exact point-construction
# and scenario conventions (reused, not reinvented -- see that file's own header): calibration
# (gp0*1.01/zfree0), nearby = zfree + 0.05*dir (seed 777), distant/difficult = zfree + 1.0*dir
# (same dir), each representation (no-CM via evaluate_fullA_screened_ranged, CM-on via
# cm_production_value_v2) run cold / exact-cache-hit / nearby-warm / difficult-solve.
#
# Points (brief Task 5): (1) calibration, (2) latest unrestricted delta=1 candidate -- loaded
# READ-ONLY from gravity-fullA-d20-canonical-rerun's own checkpoint
# (canonical_rerun/checkpoints/d1_startB_canon_latest.jls, draw_seed=20260719 matching this
# script's own seed, best_feasible.w the accepted incumbent, kappa=0.07864 -- see
# canon_inspect_checkpoints.jl's own read-only inspection, run first to confirm this before
# committing to it as "the" candidate), (3) the hard CM L=50 candidate from
# c14_find_hard_cm_point.jl's fixture.
# Include order matches c10_d20_production_driver.jl EXACTLY (see c14_diag_screen_shortcut.jl's
# header for the full story: an earlier ad-hoc subset of these includes -- missing
# compressed_cc_inner.jl/lfix_buffer_reuse.jl/bandwidth_cache_policy.jl/dual_bank.jl -- silently
# produced a WRONG Delta_dual (reads as -0.0 instead of the correct 0.2308841490034606 at the
# calibration point) with no error thrown; confirmed fixed by cross-checking against
# timing_harness.jl, unmodified, which includes the full production chain and reproduces the
# correct historical value exactly).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "dual_bank.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

lp("=== c14_final_ab_benchmark === ", Dates.now())
W = 80000; DELTA = 1.0
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d", time()-t0, D, W))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
w_calib = vcat(gp0 * 1.01, zfree0)

snaps = nested_grid_sequence([10, 20, 50])
cfg = CMConfig(common_marginals = true, cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50],
               cm_basis = :cumulative, cm_hessian_backend = :structured, contrasts = :anchored)
t0 = time()
pcx = build_cm_production_context_v2(ctx, CS, cfg; L = 50)
lp(@sprintf(">>> CM production context (L=50) built in %.1fs", time()-t0))

# ---- point 2: latest unrestricted delta=1 candidate, read-only from the canonical-rerun worktree ----
struct D20Checkpoint
    schema::Int; run_id::String; label::String; branch::Symbol; find_smallest::Bool
    delta::Float64; W::Int; draw_seed::Int; g::Float64; zfree::Vector{Float64}
    logA_full::Matrix{Float64}; dual_warm_start::Vector{Float64}; bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any; n_eval::Int; knitro_iter::Int; wall_elapsed::Float64
    checkpoint_reason::Symbol; screen_counts::NamedTuple; verify_Delta_dual::Float64
    verify_gravity_value::Float64; verify_max_abs_moment_kkt_resid::Float64
    verify_moment_resid_norm::Float64; solver_state_note::String
end
const CANON_CKPT = "/bbkinghome/edav/gravity_robustness/gravity-fullA-d20-canonical-rerun/full_aod_diag/d4_exact/canonical_rerun/checkpoints/d1_startB_canon_latest.jls"
w_unrestricted_d1 = nothing
if isfile(CANON_CKPT)
    ckpt = deserialize(CANON_CKPT)::D20Checkpoint
    ckpt.draw_seed == 20260719 || error("canonical checkpoint draw_seed=$(ckpt.draw_seed) != this script's seed 20260719 -- ctx.U would NOT match, do not reuse")
    # NOTE: ckpt.best_feasible.w != vcat(ckpt.g, ckpt.zfree) (confirmed by direct inspection --
    # best_feasible was recorded at a possibly-different n_eval than the checkpoint's "current"
    # top-level g/zfree). The checkpoint's OWN verify_Delta_dual/verify_gravity_value fields were
    # computed from vcat(g, zfree) specifically (c10_d20_production_driver.jl's resume-validation
    # block: `x_free_from_w(vcat(g, zfree_start), pe)`), NOT from best_feasible.w -- use the SAME
    # vector so this script's own "cold" re-evaluation is checked against a value that is actually
    # reproducible. Using best_feasible.w instead was tried first and produced a genuinely garbage
    # point when reinterpreted through this worktree's pivot_expand (Aod entries ~1e7-1e11,
    # eventually causing a real inner-solve failure, nStatus=-300, under CM) -- not a subtle
    # numerical issue, a real wrong-vector bug, now fixed.
    w_unrestricted_d1 = vcat(ckpt.g, ckpt.zfree)
    lp(@sprintf(">>> loaded canonical unrestricted delta=1 candidate (read-only): kappa=%.6f n_eval=%d knitro_iter=%d verify_Delta_dual=%.6f",
        ckpt.best_feasible.kappa, ckpt.n_eval, ckpt.knitro_iter, ckpt.verify_Delta_dual))
else
    lp(">>> WARNING: canonical checkpoint not found at ", CANON_CKPT, " -- skipping point 2 (documented gap, not fabricated)")
end

# ---- point 3: hard CM L=50 candidate, from c14_find_hard_cm_point.jl's fixture ----
fixture_path = joinpath(OUTDIR, "hard_cm_point.jls")
isfile(fixture_path) || error("run c14_find_hard_cm_point.jl first")
fixture = deserialize(fixture_path)
w_hard_cm = fixture.hard_point.w
lp(@sprintf(">>> loaded hard CM point: %s  n_hess=%d  wall=%.1fs", fixture.hard_point.label, fixture.hard_point.n_hess, fixture.hard_point.wall))

points = [("calibration", w_calib)]
w_unrestricted_d1 !== nothing && push!(points, ("unrestricted_delta1_candidate", w_unrestricted_d1))
push!(points, ("hard_cm_l50_candidate", w_hard_cm))

Random.seed!(777)
dir_template = randn(length(zfree0)); dir_template ./= sqrt(sum(abs2, dir_template))

all_rows = NamedTuple[]

# ================= NO-CM representation =================
lp(""); lp("="^100); lp("NO-CM (unrestricted) representation"); lp("="^100)
cache_ur = oracle_cache_for(ctx)
for (label, w) in points
    lp("-"^100); lp("POINT (no-CM): ", label); lp("-"^100)
    xf = x_free_from_w2(w)
    zfree = w[2:end]

    ctx.obj.x .= NaN
    t0 = time()
    r_cold, m_cold = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = cache_ur, use_cache = true, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_cold = time() - t0
    lp(@sprintf("  [cold]            wall=%9.4fs  Delta_dual=%12.6f  inner_status=%d  screen_status=%s",
        t_cold, r_cold.Delta_dual, r_cold.inner_status, string(get(m_cold, :screen_status, missing))))
    push!(all_rows, (representation = "noCM", point = label, scenario = "cold", wall = t_cold, Delta_dual = r_cold.Delta_dual, inner_status = r_cold.inner_status))

    t0 = time()
    r_hit, m_hit = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = cache_ur, use_cache = true, warm = false, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_hit = time() - t0
    lp(@sprintf("  [exact-cache-hit] wall=%9.5fs  cache_hit=%s", t_hit, string(get(r_hit, :cache_hit, false))))
    push!(all_rows, (representation = "noCM", point = label, scenario = "exact_cache_hit", wall = t_hit, Delta_dual = r_hit.Delta_dual, inner_status = r_hit.inner_status))

    xf_near = x_free_from_w2(vcat(w[1], zfree .+ 0.05 .* dir_template))
    t0 = time()
    r_near, m_near = evaluate_fullA_screened_ranged(xf_near, ctx, rsc; moment_representation = :compressed,
        cache = nothing, use_cache = false, warm = true, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_near = time() - t0
    lp(@sprintf("  [nearby-warm]     wall=%9.4fs  Delta_dual=%12.6f  inner_status=%d", t_near, r_near.Delta_dual, r_near.inner_status))
    push!(all_rows, (representation = "noCM", point = label, scenario = "nearby_warm", wall = t_near, Delta_dual = r_near.Delta_dual, inner_status = r_near.inner_status))

    xf_far = x_free_from_w2(vcat(w[1], zfree .+ 1.0 .* dir_template))
    t0 = time()
    r_far, m_far = evaluate_fullA_screened_ranged(xf_far, ctx, rsc; moment_representation = :compressed,
        cache = nothing, use_cache = false, warm = true, tag = "", pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_far = time() - t0
    lp(@sprintf("  [difficult-solve] wall=%9.4fs  Delta_dual=%12.6f  inner_status=%d  screen_status=%s",
        t_far, r_far.Delta_dual, r_far.inner_status, string(get(m_far, :screen_status, missing))))
    push!(all_rows, (representation = "noCM", point = label, scenario = "difficult_solve", wall = t_far, Delta_dual = r_far.Delta_dual, inner_status = r_far.inner_status))
end

# ================= CM-on representation =================
# A point that is a perfectly good UNRESTRICTED solve (e.g. an accepted unrestricted delta=1
# incumbent, at the extreme boundary of that problem) is NOT guaranteed to be CC-solvable once the
# ADDITIONAL common-marginal restrictions are layered on top -- satisfying both simultaneously can
# genuinely push the inner dual problem to real infeasibility (nStatus=-300), exactly like Task 1's
# near-infeasible witness. This is a real, reportable A/B finding (some points aren't compatible
# with both representations), not a bug -- so every CM-on scenario call is wrapped to catch and
# record that outcome instead of crashing the whole benchmark.
lp(""); lp("="^100); lp("CM-ON (L=50) representation"); lp("="^100)
cache_cm = cm_oracle_cache_for(pcx)
dcs = hash(ctx.U)

function cm_scenario(label, scenario, xf; cache = nothing, use_cache = false)
    pcx.ctx_cm.obj.x .= NaN
    t0 = time()
    local K, base, ok, errmsg
    try
        K, base = cm_production_value_v2(xf, pcx; cache = cache, use_cache = use_cache, draw_checksum = dcs)
        ok = true; errmsg = ""
    catch e
        ok = false; K = NaN; base = nothing
        errmsg = sprint(showerror, e)
    end
    wall = time() - t0
    if ok
        lp(@sprintf("  [%-16s] wall=%9.4fs  Delta_dual=%12.6f  inner_status=%d", scenario, wall, -base.ζstar, base.inner_status))
        push!(all_rows, (representation = "CM_L50", point = label, scenario = scenario, wall = wall, Delta_dual = -base.ζstar, inner_status = base.inner_status))
    else
        lp(@sprintf("  [%-16s] wall=%9.4fs  CM-INFEASIBLE (inner solve failed): %s", scenario, wall, first(errmsg, 120)))
        push!(all_rows, (representation = "CM_L50", point = label, scenario = scenario, wall = wall, Delta_dual = NaN, inner_status = -300))
    end
    return ok
end

for (label, w) in points
    lp("-"^100); lp("POINT (CM-on): ", label); lp("-"^100)
    xf = x_free_from_w2(w)
    zfree = w[2:end]

    ok_cold = cm_scenario(label, "cold", xf; cache = cache_cm, use_cache = true)
    ok_cold && cm_scenario(label, "exact_cache_hit", xf; cache = cache_cm, use_cache = true)

    # nearby/difficult: no exact cache (genuinely new points); attempted regardless of whether
    # "cold" itself was CM-feasible, since these are DIFFERENT points (each stands on its own).
    xf_near = x_free_from_w2(vcat(w[1], zfree .+ 0.05 .* dir_template))
    cm_scenario(label, "nearby_warm", xf_near)

    xf_far = x_free_from_w2(vcat(w[1], zfree .+ 1.0 .* dir_template))
    cm_scenario(label, "difficult_solve", xf_far)
end

lp(""); lp("="^100); lp("SUMMARY TABLE"); lp("="^100)
lp(@sprintf("%-8s %-32s %-18s %10s %14s %6s", "repr", "point", "scenario", "wall_s", "Delta_dual", "status"))
for r in all_rows
    lp(@sprintf("%-8s %-32s %-18s %10.4f %14.6f %6d", r.representation, r.point, r.scenario, r.wall, r.Delta_dual, r.inner_status))
end

write_csv_rows(joinpath(OUTDIR, "final_ab_benchmark.csv"), all_rows)
lp(""); lp(">>> wrote ", joinpath(OUTDIR, "final_ab_benchmark.csv"))
lp("DONE_C14_FINAL_AB_BENCHMARK")
