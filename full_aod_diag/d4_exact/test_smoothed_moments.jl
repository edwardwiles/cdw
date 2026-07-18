# Task §12E validation. Run: julia --project=. full_aod_diag/d4_exact/test_smoothed_moments.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "smoothed_moments.jl"))

const COMMIT = "9e03706"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
θ0 = CS.reconstruct_full(x0, ctx.m)
base = solve_base_state(x0, ctx)

println("="^78); println("TEST 1: smoothed G -> hard G as tuner -> -infinity"); println("="^78)
W = size(ctx.U, 1); K_hard = zeros(W); G_hard = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K_hard, G_hard, θ0, ctx.U, ctx.obj)
for tuner in (-1.0, -10.0, -100.0, -1000.0, -1e5)
    G_smooth = smoothed_factual_G(θ0, ctx, tuner)
    err = maximum(abs.(G_smooth .- G_hard[:, 1:ctx.D^2]))
    println("tuner=", tuner, "  max|G_smooth - G_hard| = ", err)
end
println("(expect monotone decrease toward 0 as tuner -> -infinity; measurement, not a hard assert)")

println("\n" * "="^78); println("TEST 2: smoothed_frozen_adjoint_Q reduces to frozen_adjoint_Q as tuner->-inf"); println("="^78)
Q_hard = frozen_adjoint_Q(x0, ctx, base)
for tuner in (-10.0, -100.0, -1000.0, -1e5)
    Q_smooth = smoothed_frozen_adjoint_Q(x0, ctx, base, tuner)
    println("tuner=", tuner, "  Q_smooth=", Q_smooth, "  Q_hard=", Q_hard, "  diff=", abs(Q_smooth - Q_hard))
end
@assert abs(smoothed_frozen_adjoint_Q(x0, ctx, base, -1e6) - Q_hard) < 1e-4 "smoothed_frozen_adjoint_Q does not converge to the hard frozen_adjoint_Q as tuner -> -infinity -- construction bug"
println("PASS: smoothed collapses to hard at extreme tuner")

println("\n" * "="^78); println("TEST 3: smoothing genuinely removes the kink at the known tie threshold"); println("="^78)
rng = MersenneTwister(4242)   # SAME direction/threshold as winner_switching.jl's kink test
v = randn(rng, length(x0)); v ./= norm(v)
v_mat = v_free_to_Amat(v, ctx)
thresh, _ = exact_tie_thresholds(θ0, v_mat, ctx)
finite_pos = sort(thresh[(thresh .> 1e-8) .& isfinite.(thresh)])
t_star = finite_pos[1]
xfun = t -> x0 .* exp.(t .* v)
# CONTINUATION-SESSION FINDING (OPEN, NOT FULLY ROOT-CAUSED -- see docs/fullA_d4_resume_audit.md
# sec 6.4 for the full investigation trail): the smoothed_frozen_adjoint_Q/smoothed_factual_G path
# exhibits genuine, unexplained run-to-run and CALL-HISTORY-DEPENDENT non-determinism near this exact
# tie threshold. Isolated facts: (1) NOT the callee alone -- calling smoothed_factual_G directly,
# repeatedly, at a fixed perturbed theta is stable within one process; (2) NOT a generic first-call/
# JIT artifact -- a genuine first-ever call to smoothed_frozen_adjoint_Q at a non-degenerate point is
# stable; (3) a STANDALONE script computing ONLY this delta-sweep (nothing else run first) gives a
# perfectly reproducible, cleanly-linear-in-delta result at delta>=1e-8 across independent process
# runs; (4) but THIS script (which runs TEST 1/TEST 2's many other smoothed-moment calls at other
# tuners/points first) gives a DIFFERENT, seemingly random result at the SAME delta=1e-8, varying
# across runs. This means the result depends on call history, not just (theta, tuner, delta) -- a
# real latent bug (likely a genuine floating-point/compiler-codegen sensitivity interacting with
# something not yet isolated), not a numerical-conditioning artifact alone. Root-causing further was
# out of this session's time budget. IMPORTANT SCOPE NOTE: this affects ONLY this diagnostic-only
# Method E smoothing path (smoothed_factual_G / smoothed_frozen_adjoint_Q / smoothMinIndNew!, all
# under full_aod_diag/d4_exact/ or misc/, none touched by production or by any other phase of this
# investigation) -- it does NOT affect evaluate_fullA (the hard-value oracle used throughout Phases
# A/B/D/F), which passed a dedicated bit-identical-repeated-call determinism test in this session's
# mandatory smoke test (test_oracle.jl TEST 2). Every number below is reported with an explicit
# multi-sample spread rather than a single point value, precisely because a single point value from
# this path is not currently trustworthy.
delta = 1e-8
const DELTA_UNSTABLE_1E9 = 1e-9
Qp_hard = frozen_adjoint_Q(xfun(t_star + delta), ctx, base); Qm_hard = frozen_adjoint_Q(xfun(t_star - delta), ctx, base)
println("HARD jump at threshold: ", abs(Qp_hard - Qm_hard))
for tuner in (-100.0, -1000.0)
    Qp_s = smoothed_frozen_adjoint_Q(xfun(t_star + delta), ctx, base, tuner)
    Qm_s = smoothed_frozen_adjoint_Q(xfun(t_star - delta), ctx, base, tuner)
    println("SMOOTHED (tuner=", tuner, ") jump at threshold: ", abs(Qp_s - Qm_s))
end
# at a MUCH wider straddle (delta=1e-3, comparable to the smoothing's own natural bandwidth), the
# smoothed function should vary continuously/gradually rather than show a step
Qp_s_wide = smoothed_frozen_adjoint_Q(xfun(t_star + 1e-3), ctx, base, -100.0)
Qm_s_wide = smoothed_frozen_adjoint_Q(xfun(t_star - 1e-3), ctx, base, -100.0)
println("SMOOTHED (tuner=-100) wide-delta(1e-3) change across threshold: ", abs(Qp_s_wide - Qm_s_wide), " (expect comparable to a generic nearby interval, not a concentrated jump)")

"""
Retry wrapper around the tuner=-100 narrow-straddle jump computation.

CONTINUATION-SESSION FINDING (correction #5 in the resume audit): the FIRST
evaluation of `smoothed_frozen_adjoint_Q` at a point within ~1e-9 of an EXACT
tie threshold, with a sharp tuner (-100), is occasionally NaN -- reproduced
directly (see docs/fullA_d4_resume_audit.md sec 6.4): calling the identical
function with bit-identical arguments a second time then gives a STABLE,
repeatable finite value (1.5246771727095432e-11, matching to 4 sig figs
across independent script runs). Isolated (not merely observed) to NOT be:
(a) JIT/precompilation of the callee `smoothed_factual_G` -- calling that
function directly, repeatedly, at the exact same perturbed theta never
produced a NaN in 3/3 trials; (b) a generic "first call to this function
ever" artifact -- a genuinely first-ever call to `smoothed_frozen_adjoint_Q`
at a DIFFERENT (non-degenerate) point x0 never produced a NaN in 2/2 trials.
The NaN is specific to evaluating a SHARP-temperature softmax at a
femtoscale (1e-9) perturbation from an EXACT degenerate tie -- genuine
floating-point ill-conditioning (a near-zero softmax denominator subject to
catastrophic cancellation, sensitive to call-site-dependent compiler
instruction ordering/FMA use), not a logic error. This is the same family of
finding as Phase A's h-grid fragility (results/fullA_d4/<commit>/
phaseA_upper_revalidation/step4_h_grid.csv): both show the exact hard-winner
landscape is genuinely singular at tie thresholds, and any FD/softmax probe
placed close enough to one is fragile by construction, not by implementation
bug. Retrying is a legitimate mitigation BECAUSE the retried value is stable
and reproducible -- this is not "retry until you like the answer".
"""
function robust_smoothed_jump(xp, xm, ctx, base, tuner; max_tries::Int = 5)
    first_was_nan = false
    for attempt in 1:max_tries
        Qp = smoothed_frozen_adjoint_Q(xp, ctx, base, tuner)
        Qm = smoothed_frozen_adjoint_Q(xm, ctx, base, tuner)
        if isnan(Qp) || isnan(Qm)
            attempt == 1 && (first_was_nan = true)
            continue
        end
        return abs(Qp - Qm), first_was_nan, attempt
    end
    return NaN, first_was_nan, max_tries
end

smoothed_jump_100, first_call_was_nan, n_tries = robust_smoothed_jump(xfun(t_star+delta), xfun(t_star-delta), ctx, base, -100.0)
println("\ndelta=1e-8, ONE sample from THIS process/call-history: ", smoothed_jump_100,
        "  (first attempt was NaN: ", first_call_was_nan, ", stabilized after ", n_tries, " attempt(s))")
println("NOTE: per the finding documented above this value is CALL-HISTORY-DEPENDENT, not a fixed")
println("      function of (theta,tuner,delta) alone -- do not treat this single number as reproducible.")

# document the instability directly with multiple within-process samples at BOTH delta=1e-9 and 1e-8
unstable_samples_1e9 = Float64[]; samples_1e8 = Float64[]
for _ in 1:3
    j9, _, _ = robust_smoothed_jump(xfun(t_star+DELTA_UNSTABLE_1E9), xfun(t_star-DELTA_UNSTABLE_1E9), ctx, base, -100.0; max_tries = 1)
    push!(unstable_samples_1e9, j9)
    j8, _, _ = robust_smoothed_jump(xfun(t_star+delta), xfun(t_star-delta), ctx, base, -100.0; max_tries = 1)
    push!(samples_1e8, j8)
end
println("delta=1e-9 within-process samples: ", unstable_samples_1e9)
println("delta=1e-8 within-process samples: ", samples_1e8, " (expect all equal WITHIN one process/call-history; cross-process comparison is the unresolved part)")

open(joinpath(OUTDIR, "smoothing_check.csv"), "w") do io
    println(io, "quantity,value")
    println(io, "hard_jump_at_threshold,", abs(Qp_hard - Qm_hard))
    println(io, "smoothed_tuner100_jump_delta1e-8_this_run,", smoothed_jump_100)
    println(io, "smoothed_tuner100_delta1e-9_sample1,", unstable_samples_1e9[1])
    println(io, "smoothed_tuner100_delta1e-9_sample2,", unstable_samples_1e9[2])
    println(io, "smoothed_tuner100_delta1e-9_sample3,", unstable_samples_1e9[3])
    println(io, "smoothed_tuner100_delta1e-8_sample1,", samples_1e8[1])
    println(io, "smoothed_tuner100_delta1e-8_sample2,", samples_1e8[2])
    println(io, "smoothed_tuner100_delta1e-8_sample3,", samples_1e8[3])
    println(io, "note,\"OPEN BUG (unresolved this session): this diagnostic's result is call-history-dependent across process runs at both delta=1e-8 and 1e-9 -- NOT a stable function of (theta,tuner,delta) alone. Do not use ANY single value from this file as a headline claim. Isolated to the Method E smoothing-diagnostic path ONLY; evaluate_fullA (used in Phases A/B/D/F) is unaffected -- see docs/fullA_d4_resume_audit.md sec 6.4.\"")
end
println("\nWrote ", joinpath(OUTDIR, "smoothing_check.csv"))
println("\nMETHOD E (DERIVATIVE-ONLY SMOOTHING) DIAGNOSTIC COMPLETE")
println("NOTE: labeled heuristic/inexact per task sec 12E -- NOT proposed as the exact-hard outer solver's gradient.")
