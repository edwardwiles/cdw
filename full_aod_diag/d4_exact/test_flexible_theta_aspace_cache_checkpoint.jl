# Cache / DualBank / checkpoint-resume gates for the flexible-theta a-space production port
# (task §10/§11). Runs at D=4 rectangular scale (fast, deterministic) -- the cache/checkpoint
# MECHANICS being tested (fingerprint composition, exact-hit logic, schema rejection) are
# scale-independent; D=20 wall-clock behavior is covered separately by the D=20 gate script and
# the matched-comparison campaign's own checkpoint activity.
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_flexible_theta_aspace_cache_checkpoint.jl
using Random
using LinearAlgebra: norm

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_flexible_theta_A.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool; note::AbstractString = "")
    status = cond ? "PASS" : "FAIL"
    lp(rpad(status, 6), name, note == "" ? "" : "  ($note)")
    cond || push!(FAILURES, name)
end

ctx_fixed = d_exact_setup_scaled(D = 4, W = 4000, row_idx = 4)   # rectangular D=4/D_dest=3
theta_star = 1.0 / ctx_fixed.μHat
sigma = ctx_fixed.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 2 * theta_star
ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
ctx = merge(ctx, (pairwise = nothing, witness = nothing))
rsc = build_ranged_screen_context(ctx)
xy = precompute_aspace_XY(ctx)
Ddest = _flex_ddest(ctx)

x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
gp0 = x_free_fixed[1]
logA_full0 = log.(reshape(x_free_fixed[2:end], ctx.D, Ddest))
pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
w_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)

lp(">>> S10: cache fingerprint composition")
fp_flex = context_fingerprint_flexible_theta(ctx)
ctx_fixed_mode = merge(ctx, (trade_elasticity_mode = :fixed,))
fp_fixed_tag = context_fingerprint_flexible_theta(ctx_fixed_mode)
check("flexible-mode fingerprint differs from fixed-mode-tagged fingerprint (same ctx otherwise)", fp_flex != fp_fixed_tag)

ctx_narrower = make_flexible_theta(ctx_fixed; theta_lo = theta_min * 1.5, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
fp_narrower = context_fingerprint_flexible_theta(ctx_narrower)
check("different theta bounds -> different fingerprint", fp_flex != fp_narrower)

fp_flex_2 = context_fingerprint_flexible_theta(ctx)
check("SAME ctx -> IDENTICAL fingerprint (deterministic)", fp_flex == fp_flex_2)

lp(">>> S10: exact-point cache A/B/A -- solve at theta A, solve at theta B, return to theta A, require exact hit")
exact_cache = SafeExactCache()
sc = ScreenCounters()

theta_A = theta_star
theta_B = theta_star * 1.03
w_A = copy(w_start_a); w_A[1] = log(theta_A)
w_B = copy(w_start_a); w_B[1] = log(theta_B)

r_A1, _, d_A1 = screened_eval_flexible_A(w_A, ctx, rsc, sc, Ref(0), xy; warm = false, exact_cache = exact_cache)
check("A/B/A step 1 (theta A) feasible", r_A1.inner_status in FEASIBLE_CODES)
size_after_A1 = length(exact_cache)

r_B, _, d_B = screened_eval_flexible_A(w_B, ctx, rsc, sc, Ref(0), xy; warm = false, exact_cache = exact_cache)
check("A/B/A step 2 (theta B) feasible", r_B.inner_status in FEASIBLE_CODES)
size_after_B = length(exact_cache)
check("cache grew after a genuinely new (theta B) point", size_after_B > size_after_A1)

r_A2, _, d_A2 = screened_eval_flexible_A(w_A, ctx, rsc, sc, Ref(0), xy; warm = true, exact_cache = exact_cache)
size_after_A2 = length(exact_cache)
check("A/B/A step 3 (theta A again): cache size UNCHANGED (exact hit, no new solve)", size_after_A2 == size_after_B;
      note = "sizes: $size_after_A1 -> $size_after_B -> $size_after_A2")
check("A/B/A step 3: Delta_dual IDENTICAL to step 1 (exact cache hit, not a fresh solve)", r_A1.Delta_dual === r_A2.Delta_dual;
      note = "step1=$(r_A1.Delta_dual) step3=$(r_A2.Delta_dual)")
check("A/B/A step 3: xf IDENTICAL to step 1", d_A1.xf == d_A2.xf)
check("A/B/A cache_hit flag set on step 3", get(r_A2, :cache_hit, false) == true)

lp(">>> S10: identical theta+A+gp+draws+layout+valid-optimum -> exact cache hit (no re-solve), re-verified via a SECOND independent cache instance")
exact_cache2 = SafeExactCache()
r_x1, _, _ = screened_eval_flexible_A(w_A, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false, exact_cache = exact_cache2)
r_x2, _, _ = screened_eval_flexible_A(w_A, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = true, exact_cache = exact_cache2)
check("identical outer coordinates hit cache on 2nd call (fresh cache instance)", get(r_x2, :cache_hit, false) == true)
check("identical outer coordinates: Delta_dual bit-identical", r_x1.Delta_dual === r_x2.Delta_dual)

lp(">>> S11: checkpoint save/resume at theta_star and off theta_star")
ckpt_dir = mktempdir()
d_ckpt = decode_and_expand_flexible_A(w_A, ctx, xy)
r_ckpt, _ = screened_eval(d_ckpt.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
ckpt1 = D20CheckpointFlexA(CHECKPOINT_SCHEMA_FLEX_A, "test_run", "gateS11", :upper, true, 1.0, 4000, 20260719,
    w_A[2], copy(d_ckpt.z_nonpivot), pivot_expand_cheap(d_ckpt.z_nonpivot, d_ckpt.pgc, d_ckpt.mu),
    copy(ctx.obj.x), Dict{Int,Float64}(), nothing, 1, 0, 1.0, :stage_complete, as_namedtuple(ScreenCounters()),
    r_ckpt.Delta_dual, r_ckpt.gravity_value, r_ckpt.max_abs_moment_kkt_resid, 0.0, "test", :pseudorandom, "u", "t",
    "13.0.1", :exclude_row, ctx.row_idx, ctx.D_dest, :flexible, :theta_decoupled_aspace, w_A[1], d_ckpt.theta,
    theta_min, theta_max, copy(d_ckpt.a_nonpivot))
p1 = joinpath(ckpt_dir, "gateS11_flexA_latest.jls")
save_checkpoint_flexA(p1, ckpt1)
loaded1 = load_checkpoint_flexA(p1)
check("checkpoint round-trips at theta_star: eta_theta bit-identical", loaded1.eta_theta == ckpt1.eta_theta)
check("checkpoint round-trips at theta_star: a_nonpivot bit-identical", loaded1.a_nonpivot == ckpt1.a_nonpivot)
check("checkpoint round-trips at theta_star: logA_full bit-identical", loaded1.logA_full == ckpt1.logA_full)

d_ckpt_off = decode_and_expand_flexible_A(w_B, ctx, xy)
r_ckpt_off, _ = screened_eval(d_ckpt_off.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
ckpt2 = D20CheckpointFlexA(CHECKPOINT_SCHEMA_FLEX_A, "test_run", "gateS11off", :upper, true, 1.0, 4000, 20260719,
    w_B[2], copy(d_ckpt_off.z_nonpivot), pivot_expand_cheap(d_ckpt_off.z_nonpivot, d_ckpt_off.pgc, d_ckpt_off.mu),
    copy(ctx.obj.x), Dict{Int,Float64}(), nothing, 1, 0, 1.0, :stage_complete, as_namedtuple(ScreenCounters()),
    r_ckpt_off.Delta_dual, r_ckpt_off.gravity_value, r_ckpt_off.max_abs_moment_kkt_resid, 0.0, "test", :pseudorandom, "u", "t",
    "13.0.1", :exclude_row, ctx.row_idx, ctx.D_dest, :flexible, :theta_decoupled_aspace, w_B[1], d_ckpt_off.theta,
    theta_min, theta_max, copy(d_ckpt_off.a_nonpivot))
p2 = joinpath(ckpt_dir, "gateS11off_flexA_latest.jls")
save_checkpoint_flexA(p2, ckpt2)
loaded2 = load_checkpoint_flexA(p2)
check("checkpoint round-trips OFF theta_star: theta bit-identical", loaded2.theta == ckpt2.theta)
check("off-theta checkpoint theta != theta_star (genuinely off)", loaded2.theta != theta_star)

lp(">>> S11: deliberate mismatch rejections")
let threw = false
    try
        bad = deserialize_bad = ckpt1  # a D20CheckpointFlexA
        # Deliberate fixed-vs-flexible mismatch: construct a bogus file containing a plain Int (not
        # even a checkpoint struct) and confirm load_checkpoint_flexA rejects it loudly.
        bogus_path = joinpath(ckpt_dir, "bogus.jls")
        serialize(bogus_path, 42)
        load_checkpoint_flexA(bogus_path)
    catch e
        threw = true
    end
    check("load_checkpoint_flexA rejects a non-D20CheckpointFlexA file loudly (not silently)", threw)
end
let threw = false
    try
        bogus2 = D20CheckpointFlexA(CHECKPOINT_SCHEMA_FLEX_A + 1, ckpt1.run_id, ckpt1.label, ckpt1.branch, ckpt1.find_smallest,
            ckpt1.delta, ckpt1.W, ckpt1.draw_seed, ckpt1.g, ckpt1.zfree, ckpt1.logA_full, ckpt1.dual_warm_start,
            ckpt1.bandwidth_cache, ckpt1.best_feasible, ckpt1.n_eval, ckpt1.knitro_iter, ckpt1.wall_elapsed,
            ckpt1.checkpoint_reason, ckpt1.screen_counts, ckpt1.verify_Delta_dual, ckpt1.verify_gravity_value,
            ckpt1.verify_max_abs_moment_kkt_resid, ckpt1.verify_moment_resid_norm, ckpt1.solver_state_note,
            ckpt1.draw_design, ckpt1.draw_checksum_uniform, ckpt1.draw_checksum_transformed, ckpt1.knitro_version,
            ckpt1.destination_sample, ckpt1.row_idx, ckpt1.D_dest, ckpt1.trade_elasticity_mode, ckpt1.A_coordinate_mode,
            ckpt1.eta_theta, ckpt1.theta, ckpt1.theta_lo, ckpt1.theta_hi, ckpt1.a_nonpivot)
        p_bad_schema = joinpath(ckpt_dir, "bad_schema.jls")
        save_checkpoint_flexA(p_bad_schema, bogus2)
        load_checkpoint_flexA(p_bad_schema)
    catch e
        threw = true
    end
    check("load_checkpoint_flexA rejects a wrong-schema checkpoint loudly", threw)
end

lp(">>> S17 side-effect: DualBank record/select smoke (theta-blind by design, documented limitation)")
bank = DualBank(4)
r_bank1, _, d_bank1 = screened_eval_flexible_A(w_A, ctx, rsc, sc, Ref(0), xy; warm = false, bank = bank)
check("DualBank warm-start wiring: no crash on record/select through screened_eval_flexible_A", true)

lp(isempty(FAILURES) ? "ALL CACHE/CHECKPOINT GATES PASS" : "CACHE/CHECKPOINT FAILURES: $(join(FAILURES, "; "))")
flush(stdout)
