# ============================================================================
# Restricted-immutable-workspace production port (2026-07-24), task section 4:
# mismatch-rejection test for the new fingerprinted benchmark-seed checkpoint
# (cm_checkpoint_fingerprint.jl).
#
# Two things this test must show, live, at real D=20/W=80,000 scale:
#   1. The actual stale 2026-07-22 seed file (cm_campaign_2026-07-22/chain1/
#      delta_1.0/cold_verified_seed.jls) is explicitly REJECTED by
#      load_and_validate_benchmark_seed with an informative
#      BenchmarkSeedMismatch (not a bare BoundsError three frames away).
#   2. A freshly-written, correctly-fingerprinted seed at the SAME live
#      context round-trips successfully (save -> load -> x_free recovered
#      exactly).
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_checkpoint_fingerprint_mismatch.jl
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_checkpoint_fingerprint.jl"))
using Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
ok_all = true

lp(">>> [", now(), "] building D=20/W=80,000 real-data context (current default destination_sample)...")
ctx = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
lp(">>> ctx: D=", ctx.D, " D_dest=", _ctx_ddest(ctx), " destination_sample=",
   hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :unknown,
   " pe.other_idx length=", length(pe.other_idx))

# ---------------------------------------------------------------------------
# Test 1: the actual stale 2026-07-22 file must be explicitly rejected.
# ---------------------------------------------------------------------------
const STALE_PATH = "/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/cold_verified_seed.jls"
lp()
lp("--- Test 1: stale pre-omit-ROW checkpoint must be rejected explicitly ---")
if isfile(STALE_PATH)
    try
        load_and_validate_benchmark_seed(STALE_PATH, ctx, pe)
        lp("FAIL: expected BenchmarkSeedMismatch, got no error")
        global ok_all = false
    catch e
        if e isa BenchmarkSeedMismatch
            lp("PASS: stale checkpoint rejected with informative error:")
            lp(sprint(showerror, e))
        else
            lp("FAIL: expected BenchmarkSeedMismatch, got ", typeof(e), ": ", sprint(showerror, e))
            global ok_all = false
        end
    end
else
    lp("SKIP: stale file not present at ", STALE_PATH, " (nothing to reject -- not a failure of this test)")
end

# ---------------------------------------------------------------------------
# Test 2: a synthetic mismatched fingerprint (same everything except
# destination_sample/D_dest forced to the OLD :all_legacy/square convention)
# must also be rejected -- this exercises the validator directly, independent
# of whether the one stale file on disk happens to still exist.
# ---------------------------------------------------------------------------
lp()
lp("--- Test 2: synthetic pre-omit-ROW fingerprint (destination_sample=:all_legacy, D_dest=20) must be rejected ---")
x_free_calib = ctx.θ0_up[ctx.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], ctx.D, _ctx_ddest(ctx))), pe))
bad = RestrictedWorkspaceBenchmarkSeed(
    BENCHMARK_SEED_SCHEMA, string(now()), :calibration,
    size(ctx.U, 1), ctx.draw_design, ctx.draw_seed,
    ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
    :all_legacy, ctx.D, ctx.D, ctx.D * ctx.D, ctx.D * ctx.D - 1,   # WRONG: old square convention
    pe.pivot_lin, 0, 0, "none", :orthonormal, 0,
    w_calib, Float64[], 0.0, 1.0, string(now()))
bad_path = tempname()
serialize(bad_path, bad)
try
    load_and_validate_benchmark_seed(bad_path, ctx, pe)
    lp("FAIL: expected BenchmarkSeedMismatch, got no error")
    global ok_all = false
catch e
    if e isa BenchmarkSeedMismatch
        lp("PASS: synthetic pre-omit-ROW fingerprint rejected:")
        lp(sprint(showerror, e))
    else
        lp("FAIL: expected BenchmarkSeedMismatch, got ", typeof(e), ": ", sprint(showerror, e))
        global ok_all = false
    end
end
rm(bad_path; force = true)

# ---------------------------------------------------------------------------
# Test 3: a freshly-written, correctly-fingerprinted seed round-trips.
# ---------------------------------------------------------------------------
lp()
lp("--- Test 3: fresh correctly-fingerprinted seed round-trips exactly (serialization-only check) ---")
# Note: x_free_calib itself is NOT exactly on the pivot-gravity manifold (it's the raw calibrated
# point, satisfying gravity only to estimation precision, not to this pivot parameterization's
# machine-epsilon solved pivot cell) -- so the right in-process reference is x_free_from_w(w_calib,
# pe) computed directly (no serialization), isolating save/load precision from that unrelated
# manifold-snap discrepancy.
x_free_direct = x_free_from_w(w_calib, pe)
fresh_path = tempname()
save_benchmark_seed(fresh_path, ctx, pe, w_calib; family = :calibration, Delta_dual = 0.0, delta_budget = 1.0)
recovered = load_and_validate_benchmark_seed(fresh_path, ctx, pe; expected_family = :calibration)
max_diff = maximum(abs.(recovered.x_free .- x_free_direct))
if max_diff < 1e-12
    lp("PASS: round-trip exact (vs direct x_free_from_w(w_calib,pe)), max_diff=", max_diff)
else
    lp("FAIL: round-trip max_diff=", max_diff, " (expected < 1e-12)")
    global ok_all = false
end
rm(fresh_path; force = true)

lp()
lp(ok_all ? "ALL CHECKPOINT FINGERPRINT TESTS PASSED" : "SOME CHECKPOINT FINGERPRINT TESTS FAILED -- SEE ABOVE")
lp("DONE_CHECKPOINT_FINGERPRINT_TEST")
