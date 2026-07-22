# ============================================================================
# Closure task Phase 3F: verify the accepted-point checkpoint reuse (commit 9da9e89,
# cb_newpt! in c10_d20_production_driver.jl) does what it claims:
#   - reuses ONLY an exact full-vector match to the most recent complete function result
#   - rejects stale/approximate/mismatched states, falling back to a fresh solve
#   - (context fingerprint / CM config: satisfied TRIVIALLY here -- last_F_state is a
#     per-call closure-local Ref inside a SINGLE run_profile_checkpointed/
#     run_polish_checkpointed invocation, never persisted or shared across a different ctx/
#     process, so there is no cross-context risk to gate against -- documented, not an
#     oversight)
#   - increments a counter for avoided checkpoint-only inner solves (added THIS closure
#     task, n_checkpoint_reuse_hits -- the original commit had the reuse logic but no
#     counter, a real gap this fixes)
#
# Part 1 (PURE, no KNITRO): exact reproduction of the equality-gate idiom
# `(shared !== nothing && shared.w == w_now) ? shared.r : nothing`, covering exact-match,
# one-bit mismatch, and no-prior-state (nothing) cases.
# Part 2 (real D=20/W=80,000/delta=1 KNITRO run): confirms n_checkpoint_reuse_hits > 0 in a
# real short run, and that checkpoint save/load still round-trips correctly with the new
# field present (D20Checkpoint itself is untouched; only the FUNCTION RETURN NamedTuple
# gained a field).
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Printf

lp(xs...) = (println(xs...); flush(stdout))
n_pass = Ref(0); n_fail = Ref(0)
check(name, cond) = (cond ? (lp("  PASS: ", name); n_pass[] += 1) : (lp("  FAIL: ", name); n_fail[] += 1))

lp("=== Part 1: idiom-level exact-match/mismatch/nothing gate (PURE) ===")
function reuse_gate(shared, w_now)
    r_now = (shared !== nothing && shared.w == w_now) ? shared.r : nothing
    return r_now
end

w1 = [0.9878, 0.1, 0.2, 0.3]
shared_ok = (w = copy(w1), r = :the_real_result)
check("exact-match reuses the stored result", reuse_gate(shared_ok, w1) === :the_real_result)
check("exact-match with a freshly-copied (not === ) but ==-equal vector still reuses",
      reuse_gate(shared_ok, copy(w1)) === :the_real_result)

w2_onebit = copy(w1); w2_onebit[3] = nextfloat(w2_onebit[3])   # smallest possible perturbation
check("a one-ULP mismatch is treated as a DIFFERENT point (no false-positive reuse)",
      reuse_gate(shared_ok, w2_onebit) === nothing)

w3_different = [0.5, -0.1, 0.2, 0.3]
check("a genuinely different point falls back (returns nothing, not stale data)",
      reuse_gate(shared_ok, w3_different) === nothing)

check("no prior state (nothing) falls back safely", reuse_gate(nothing, w1) === nothing)

lp("\n=== Part 2: real D=20/W=80,000/delta=1 run -- n_checkpoint_reuse_hits fires ===")
fctx = build_fullA_context(W = 80000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
ctx = fctx.ctx; pe = fctx.pe; D = ctx.D
Aod_real = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe)
gF = ctx.θ0_up[3+D]
ckpt = mktempdir()
res = run_polish_checkpointed("c32_reuse", true, gF, zfree0; maxtime_real = 60.0,
    W_in = 80000, delta_in = 1.0, ckpt_dir = ckpt, checkpoint_interval_s = 200.0,
    reuse = fctx, price_cache_backend = :cplus)
lp("  n_eval=", res.n_eval, " n_checkpoint_reuse_hits=", res.n_checkpoint_reuse_hits,
   " knitro_status=", res.knitro_status)
check("n_checkpoint_reuse_hits field is present on the return NamedTuple", haskey(res, :n_checkpoint_reuse_hits))
check("at least one accepted iterate reused cb_F!'s result (n_checkpoint_reuse_hits > 0)",
      res.n_checkpoint_reuse_hits > 0)
check("reuse hits never exceed n_eval (sanity bound -- can't reuse more than were computed)",
      res.n_checkpoint_reuse_hits <= res.n_eval)

lp("\n  checkpoint round-trip (D20Checkpoint itself untouched by this task -- only the")
lp("  RETURN NamedTuple gained a field): load_checkpoint(", res.ckpt_path, ")")
ck = load_checkpoint(res.ckpt_path)
check("checkpoint file loads successfully post-change", ck !== nothing)
check("checkpoint schema unaffected (still the current D20Checkpoint schema)", ck.schema == CHECKPOINT_SCHEMA)

lp("\n>>> RESULT: ", n_pass[], "/", n_pass[] + n_fail[], " checks passed")
n_fail[] == 0 || error("c32_phase3f_checkpoint_reuse_validation.jl: $(n_fail[]) check(s) FAILED")
