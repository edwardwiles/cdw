# ============================================================================
# Regression test for the driver crash found on the canonical D=20 rerun:
# c10_d20_production_driver.jl's screened_eval unconditionally reads
# screen_meta.worst_o/worst_d for EVERY rejection stage, but
# evaluate_fullA_screened_ranged's :pairwise_certified_infeasible branch
# (fast_range_screen.jl) did not return those fields, causing a FieldError
# inside a live KNITRO callback (observed once in 8 canonical-rerun runs,
# delta=1 Start A, n_eval=120, 1174.7s into an 1800s budget).
#
# This test constructs a genuinely pairwise-certified-infeasible point (same
# adversarial construction as test_infeasibility_screen.jl's test 9, applied
# to evaluate_fullA_screened_ranged instead of evaluate_fullA_screened) and
# asserts screen_meta carries worst_o/worst_d matching the underlying
# pairwise_certificate result -- i.e. that screened_eval's field access
# cannot FieldError on this branch.
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_pairwise_screen_meta_ranged.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))

using Random, Test

ctx = d4_exact_setup(find_smallest = true)
D = ctx.D
pe = build_pivot_elimination(ctx)
zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

pc = precompute_pairwise_M(ctx)
Pmat = target_shares(ctx)
rsc = build_ranged_screen_context(ctx)   # matches production's RangedScreenContext construction

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

println("== Regression: evaluate_fullA_screened_ranged's pairwise branch carries worst_o/worst_d ==")
Random.seed!(9999)   # same seed as test_infeasibility_screen.jl's test 9, same adversarial recipe
found_infeasible = false
local xf_bad, pres_bad
for i in 1:200
    step = 3.0 + 8.0 * rand()
    dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
    w = vcat(gp0, zfree0 .+ step .* dir)
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    if pres.infeasible
        global found_infeasible = true
        global xf_bad = xf
        global pres_bad = pres
        break
    end
end

if found_infeasible
    r_bad, meta_bad = evaluate_fullA_screened_ranged(xf_bad, ctx, rsc; moment_representation = :compressed,
                                                       cache = nothing, use_cache = false, warm = false,
                                                       pairwise = pc)
    check("screen_status == :pairwise_certified_infeasible", meta_bad.screen_status == :pairwise_certified_infeasible)
    check("screen_meta has worst_o field", hasproperty(meta_bad, :worst_o))
    check("screen_meta has worst_d field", hasproperty(meta_bad, :worst_d))
    check("worst_o matches producing pairwise_certificate", meta_bad.worst_o == pres_bad.worst_o)
    check("worst_d matches producing pairwise_certificate", meta_bad.worst_d == pres_bad.worst_d)

    # This is the exact field-access pattern from c10_d20_production_driver.jl's
    # screened_eval (the :pairwise branch push!) -- must not throw FieldError.
    driver_access_ok = try
        _ = (stage = :pairwise, o = meta_bad.worst_o, d = meta_bad.worst_d, n_eval = 0)
        true
    catch e
        println("    driver-pattern access threw: ", e)
        false
    end
    check("driver's screened_eval field-access pattern does not throw", driver_access_ok)
else
    println("  (no pairwise-certified infeasible point found in 200 tries at this seed -- skipping, not a failure)")
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail regression check(s) failed")
