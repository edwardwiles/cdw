# Phase 1: audit provenance of the 4.9823e8 AboveEvaluationCap certificate at the D20 h=0.5
# minus-side endpoint. Instruments EVERY raw objective callback value via the new opt-in
# MELITZ_OBJECTIVE_TRACE (backend_config.jl / cc_bundle.jl, 2026-07-30 addition).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

st = deserialize(joinpath(SCRATCH, "phase0_state.jls"))
theta0 = st.theta0; x0 = st.x0; calib = st.calib; focal = st.focal
D = st.D; nA = st.nA; nq = st.nq; b_q = st.b_q; stage = st.stage
println("Loaded phase0 state. Delta0=", st.Delta0, "  D=", D, "  |b_q|=", norm(b_q))

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end
obj = build_bundle()
ctx = obj.γ
@assert obj.lower_limit == -10.0
@assert obj.mode == :delta

theta_m = copy(theta0); theta_m[1+nA+1:end] .-= 0.5 .* b_q

println("\n" * "="^100); println("Instrumented MINUS-side (h=0.5) capped solve"); println("="^100); flush(stdout)

melitz_objective_trace_reset!()
MELITZ_OBJECTIVE_TRACE_ENABLED[] = true
obj.use_cached_x = false; obj.x .= NaN
t0 = time()
lfd_m = melitz_recover_lfd(obj, theta_m)
wall = time() - t0
MELITZ_OBJECTIVE_TRACE_ENABLED[] = false
trace = copy(MELITZ_OBJECTIVE_TRACE)

@printf("wall=%.4fs  nStatus=%d  lfd_ok=%s  n_callback_evals=%d\n", wall, lfd_m.nStatus, lfd_m.lfd_ok, length(trace))
println("obj.threshold_crossed[] = ", obj.threshold_crossed[])
println("obj.threshold_crossing_bound[] = ", obj.threshold_crossing_bound[])

# Report every callback value.
open(joinpath(OUTDIR, "melitz_negswitch_phase1_objective_trace_2026-07-30.csv"), "w") do io
    println(io, "call_index,f,crossed_this_call")
    for (i, (f, crossed)) in enumerate(trace)
        println(io, "$i,$(f),$(crossed)")
    end
end

first_cross_idx = findfirst(t -> t[2], trace)
println("first call index with f<=lower_limit(-10): ", first_cross_idx === nothing ? "NONE" : first_cross_idx)
if first_cross_idx !== nothing
    println("  f at that call = ", trace[first_cross_idx][1])
end
last_idx = length(trace)
if last_idx > 0
    println("LAST callback value: f=", trace[last_idx][1], "  crossed=", trace[last_idx][2])
end
# Check for any non-finite / sentinel-magnitude value entering the raw f trace itself
# (as opposed to the -KN_INFINITY RETURNED to KNITRO, which is a separate, deliberate sentinel
# never entering this trace since we log f BEFORE the lower_limit branch substitutes anything).
n_nonfinite = count(t -> !isfinite(t[1]), trace)
n_at_floatmax = count(t -> abs(t[1]) >= floatmax(Float64)/2, trace)
println("n_nonfinite f values in trace = ", n_nonfinite)
println("n |f|>=floatmax/2 in trace = ", n_at_floatmax)
println("max |f| in trace = ", maximum(abs.(first.(trace))))
println("min f (most negative) in trace = ", minimum(first.(trace)))
argmin_i = argmin(first.(trace))
println("index of most-negative f = ", argmin_i, "  value=", trace[argmin_i][1])

# Now run through the TYPED classifier (fixed code) on the SAME theta_m, fresh session, and
# report the returned certificate, confirming it matches -f at the LAST crossing call.
melitz_objective_trace_reset!()
MELITZ_OBJECTIVE_TRACE_ENABLED[] = true
session_m = MelitzInnerSession(obj, ctx, policy_cap)
r_m = solve_melitz_delta!(session_m, theta_m, policy_cap)
MELITZ_OBJECTIVE_TRACE_ENABLED[] = false
trace2 = copy(MELITZ_OBJECTIVE_TRACE)
println("\nTyped classifier result: ", typeof(r_m))
println(r_m)
if hasproperty(r_m, :certified_lower_bound)
    println("certified_lower_bound = ", r_m.certified_lower_bound)
    println("source = ", r_m.source)
end
last2 = trace2[end]
println("Last callback of THIS run: f=", last2[1], "  -f=", -last2[1], "  crossed=", last2[2])
@assert !isempty(trace2)
if hasproperty(r_m, :certified_lower_bound)
    println("match -f(last call) == certified_lower_bound ? ", -last2[1] == r_m.certified_lower_bound)
end

serialize(joinpath(SCRATCH, "phase1_trace.jls"), (trace=trace, trace2=trace2, r_m=r_m, theta_m=theta_m,
    nStatus=lfd_m.nStatus))
println("\nDONE PHASE 1")
