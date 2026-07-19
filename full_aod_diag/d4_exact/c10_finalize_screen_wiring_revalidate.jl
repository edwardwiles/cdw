# ============================================================================
# Continuation 10, Section 9 (finalize-architecture): RE-validation of
# c10_screen_wiring_validate.jl (workstream c10-prod-wiring's own screen
# false-positive check), adapted in two ways for this task:
#   1. moment_representation switched from :dense to :compressed throughout --
#      the ORIGINAL script exercised oracle_fast.jl's dense tail only, never
#      touching infeasibility_screen.jl::evaluate_fullA_screened_compressed /
#      compressed_live.jl's Hessian callback -- i.e. never the code this
#      task actually modified (the structured moment materialization + BLAS
#      KKT/moment-residual swap both only fire in :compressed mode). This
#      version exercises exactly the driver's own production configuration.
#   2. includes structured_moment_build.jl (new dependency compressed_live.jl
#      / infeasibility_screen.jl now have).
# Otherwise IDENTICAL logic/assertions to the original (Class 1 feasible
# points all pass, Class 2 adversarial large-perturbation points get
# cross-checked against the ground-truth full winner-scan for false
# positives, timing comparison) -- reusing that workstream's script, not
# reimplementing it, per this investigation's "attribute reused code"
# discipline.
# ============================================================================
# ============================================================================
# Continuation 10, Section 5 validation: confirms the NEWLY-WIRED default
# screening path (ctx.pairwise/ctx.witness built once inside d20_real_setup,
# consumed via evaluate_fullA_screened in c10_d20_production_driver.jl) is
# correct -- zero false positives -- and reports realistic rejection counts
# + timing at real D=20/W=80,000. This is a WIRING check: Continuation 9
# already validated the underlying pairwise/witness/winner-scan MATH
# (docs/fullA_D20_infeasibility_screening_report.md, 0 false positives across
# D=4/6/8/10/20); this script re-confirms that holds when those structures
# come from ctx.pairwise/ctx.witness (context-embedded, built once) instead
# of being constructed ad hoc per-script as Continuation 9's own validation
# scripts did.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using Random, Printf, Statistics
using LinearAlgebra: norm

t0 = time()
ctx = d20_real_setup(W = 80000, find_smallest = true)
println("ctx build wall=", round(time()-t0, digits=1), "s")
println("screen_setup_wall = ", ctx.screen_setup_wall)
@assert ctx.pairwise !== nothing "ctx.pairwise not built -- build_screen default must be true"
@assert ctx.witness !== nothing "ctx.witness not built -- build_screen default must be true"
D = ctx.D; D2 = D^2
pe = build_pivot_elimination(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]

function screen_and_check(xf, label)
    result, meta = evaluate_fullA_screened(xf, ctx; moment_representation = :compressed, cache = nothing,
        use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    st = meta.screen_status
    return (label = label, screen_status = st, Delta_dual = result.Delta_dual, inner_status = result.inner_status)
end

println("\n=== Class 1: feasible points (calibration + small perturbations) -- expect ALL screen_passed ===")
Random.seed!(20260719)
n_pass = 0; n_reject = 0
rows = NamedTuple[]
push!(rows, screen_and_check(x_free_from_w(vcat(gp0, zfree0)), "calibration"))
for i in 1:9
    step = randn(D2-1); step ./= norm(step); step .*= 0.5   # small step, per c9's own "typical local step" scale
    push!(rows, screen_and_check(x_free_from_w(vcat(gp0, zfree0 .+ step)), "small_pert_$i"))
end
for r in rows
    global n_pass, n_reject
    r.screen_status === :screen_passed ? (n_pass += 1) : (n_reject += 1)
    println("  ", r.label, ": screen_status=", r.screen_status, " Delta_dual=", r.Delta_dual)
end
println("Class 1 totals: passed=", n_pass, " rejected=", n_reject, " (expect rejected=0)")
@assert n_reject == 0 "UNEXPECTED: a small local perturbation was screen-rejected -- investigate before trusting the screen"

println("\n=== Class 2: large adversarial perturbations -- force some genuine rejections, cross-check vs ground truth (full_scan) ===")
Random.seed!(31337)
n_pairwise = 0; n_witness = 0; n_winner = 0; n_passed = 0
n_checked = 0; n_false_positive = 0
for i in 1:40
    step = randn(D2-1); step ./= norm(step); step .*= (8.0 + 8.0*rand())   # large step, per c9's D=20 finding (needed ~16 before any rejection)
    xf = x_free_from_w(vcat(gp0, zfree0 .+ step))
    result, meta = evaluate_fullA_screened(xf, ctx; moment_representation = :compressed, cache = nothing,
        use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    st = meta.screen_status
    global n_pairwise, n_witness, n_winner, n_passed, n_checked, n_false_positive
    if st === :pairwise_certified_infeasible
        n_pairwise += 1
    elseif st === :witness_certified_infeasible
        n_witness += 1
    elseif st === :winner_scan_infeasible
        n_winner += 1
    else
        n_passed += 1
    end
    # FALSE-POSITIVE CHECK: for every REJECTED point, cross-check against the ground-truth
    # full-scan winner construction (full_scan=true, natural order, no early exit) -- a true
    # rejection must ALSO be winner-infeasible under a full, unhurried scan. A pairwise or
    # witness rejection that the full scan calls feasible would be a genuine false positive.
    if st !== :screen_passed
        n_checked += 1
        θ_full = CS.reconstruct_full(xf, ctx.m)
        Pmat = target_shares(ctx)
        wres_full = screen_hard_winners(θ_full, ctx, Pmat; order = 1:D, full_scan = true)
        if wres_full.feasible
            n_false_positive += 1
            println("  *** FALSE POSITIVE at trial ", i, ": screen said ", st, " but full_scan says feasible ***")
        end
    end
end
println("Class 2 (40 large-perturbation trials): pairwise=", n_pairwise, " witness=", n_witness,
        " winner=", n_winner, " passed=", n_passed, " checked_for_false_positive=", n_checked,
        " FALSE_POSITIVES=", n_false_positive)
@assert n_false_positive == 0 "FALSE POSITIVE DETECTED -- screen wiring is unsafe, do not adopt as default until fixed"

println("\n=== Timing: screen-reject vs full real (dense) solve ===")
Random.seed!(12345)
# One feasible (small-step) point: time the FULL screened-and-passed evaluation (screen cost is
# a small ADDITION on top of the real solve here, since the screen passes and the real solve runs).
xf_feas = x_free_from_w(vcat(gp0, zfree0 .+ 0.1 .* randn(D2-1)))
t_feas = @elapsed evaluate_fullA_screened(xf_feas, ctx; moment_representation = :compressed, cache = nothing,
    use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
@printf("  feasible point (screen passes, real dense solve runs): %.4fs\n", t_feas)

# Find one genuinely rejected point via a large step (matching Class 2's own construction) and
# time the screen-only rejection path (no moments, no KNITRO) directly.
found_reject = false
local xf_rej, stage_rej
for i in 1:20
    step = randn(D2-1); step ./= norm(step); step .*= 16.0
    xf_try = x_free_from_w(vcat(gp0, zfree0 .+ step))
    _, meta_try = evaluate_fullA_screened(xf_try, ctx; moment_representation = :compressed, cache = nothing,
        use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    if meta_try.screen_status !== :screen_passed
        global found_reject, xf_rej, stage_rej
        found_reject = true; xf_rej = xf_try; stage_rej = meta_try.screen_status
        break
    end
end
if found_reject
    t_rej = @elapsed evaluate_fullA_screened(xf_rej, ctx; moment_representation = :compressed, cache = nothing,
        use_cache = false, warm = false, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    @printf("  rejected point (stage=%s): %.6fs  (%.1fx cheaper than the feasible-point solve above)\n",
            string(stage_rej), t_rej, t_feas / max(t_rej, 1e-9))
else
    println("  no rejected point found in 20 tries at step=16 -- consistent with Class 2's own low hit rate; timing comparison skipped")
end

println("\nSCREEN WIRING VALIDATION: PASS (0 false positives across ", n_checked, " checked rejections)")
println("DONE_SCREEN_VALIDATE")
