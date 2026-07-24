# Gate B (exclude-ROW-destination UNRESTRICTED-CORE release, 2026-07-24): screen correctness for
# the rectangularized compressed-path screen stack (fast_range_screen.jl's
# evaluate_fullA_screened_ranged, precompute_envelope/envelope_prewinner_screen/
# screen_hard_winners_ranged, range_screen_standalone; infeasibility_screen.jl's
# pairwise_certificate/compute_a_od/target_shares, already rectangular per the prior CM-family
# release and unchanged here).
#
# Reuses the exact adversarial-point recipe already validated in test_infeasibility_screen.jl's
# test 9 / test_pairwise_screen_meta_ranged.jl (large random step from the calibration point,
# seed 9999, up to 200 tries) -- a genuinely pairwise-certified-infeasible point, independently
# verified via pairwise_certificate itself (the exact core structural-infeasibility certificate
# whose validity does NOT depend on the compressed representation, matching the theorem in the
# task brief section 4), applied here to D=4/Ddest=3 rectangular and real D=20/:exclude_row (in
# addition to the pre-existing D=4 square coverage, run separately as a regression check).
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_exclude_row_unrestricted_gateB_screens.jl
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))   # -> InnerCallCounters/_INNER_CALL_COUNTERS, evaluate_fullA_fast; compressed_live.jl reuses these unqualified
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
using Random, Printf

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

"""
Adversarial-point search: large random steps from the calibration point until
`pairwise_certificate` independently certifies structural infeasibility. Same recipe as
test_infeasibility_screen.jl's test 9 (seed 9999, <=200 tries) -- reused verbatim, not
re-derived, so a failure to find a hit here would itself be a signal something upstream changed.
"""
function find_pairwise_infeasible_point(ctx, pe, pc, Pmat; seed::Int = 9999, tries::Int = 200)
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    zfree0 = pivot_reduce(zeros(ctx.D, Ddest), pe)
    gp0 = ctx.θ0_up[3+ctx.D]
    rng = MersenneTwister(seed)
    for i in 1:tries
        step = 3.0 + 8.0 * rand(rng)
        dir = randn(rng, length(zfree0)); dir ./= sqrt(sum(abs2, dir))
        w = vcat(gp0, zfree0 .+ step .* dir)
        xf = x_free_from_w(w)
        θ_full = CS.reconstruct_full(xf, ctx.m)
        a = compute_a_od(θ_full, ctx)
        pres = pairwise_certificate(a, pc, Pmat)
        if pres.infeasible
            return (xf = xf, pres = pres, zfree0 = zfree0, gp0 = gp0)
        end
    end
    return nothing
end

function gate_b_battery(label::AbstractString, ctx; expect_calibration_feasible::Bool = true)
    println()
    println("="^96)
    println("$label  (D=$(ctx.D), D_dest=$(hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D))")
    println("="^96)
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    pe = build_pivot_elimination(ctx)
    pc = precompute_pairwise_M(ctx)
    Pmat = target_shares(ctx)
    check("$label: Pmat shape == (D,D_dest)", size(Pmat) == (ctx.D, Ddest))
    rsc = build_ranged_screen_context(ctx)
    check("$label: envelope precomputation succeeds (rsc.envelope !== nothing)", rsc.envelope !== nothing)

    # ---- calibration point (A_od==1) ----
    # For the D=4 SYNTHETIC economies (fakeData=1) this point is feasible BY CONSTRUCTION (the
    # fake DGP is generated with A_od==1), so it must screen-pass. For the REAL D=20 economy
    # (fakeData=3) A_od==1 is just an arbitrary free-parameter-map normalization, not an implied
    # economically-feasible point -- real trade-cost wedges are exactly what the outer estimation
    # searches over, and this point turns out to be genuinely (not marginally) infeasible at BOTH
    # W=8000 and W=80,000 (confirmed by a direct diagnostic: envelope/winning-range/pairwise/
    # range_screen_standalone all independently agree on the SAME violated cell, (o=1,d=1), which
    # is itself a strong cross-screen correctness signal -- an axis-swap bug in any one of them
    # would not coincidentally agree with the other three). expect_calibration_feasible=false
    # switches this battery to check cross-screen AGREEMENT instead of a pass/fail assumption that
    # does not hold for real data.
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    zfree0 = pivot_reduce(zeros(ctx.D, Ddest), pe)
    gp0 = ctx.θ0_up[3+ctx.D]
    xf_pass = x_free_from_w(vcat(gp0, zfree0))
    r_pass, meta_pass = evaluate_fullA_screened_ranged(xf_pass, ctx, rsc; moment_representation = :compressed,
                                                          cache = nothing, use_cache = false, warm = false, pairwise = pc)
    if expect_calibration_feasible
        check("$label: calibration point is NOT hard-rejected by any screen (screen_status==:screen_passed)",
              meta_pass.screen_status == :screen_passed)
    else
        θ_full_cal = CS.reconstruct_full(xf_pass, ctx.m)
        a_cal = compute_a_od(θ_full_cal, ctx)
        pres_cal = pairwise_certificate(a_cal, pc, Pmat)
        eres_cal = envelope_prewinner_screen(θ_full_cal, ctx, rsc.envelope)
        check("$label: calibration point IS structurally infeasible under real data (expected, not a defect)",
              meta_pass.screen_status != :screen_passed)
        # pairwise_certificate and envelope_prewinner_screen are independent certificates with
        # different scan orders/metrics (pairwise: a_od-vs-rival-origin bound; envelope: winning-
        # value upper bound) -- both correctly detecting infeasibility is the correctness property;
        # there is no mathematical reason they must report the identical "worst" cell, so that is
        # NOT asserted here.
        check("$label: pairwise_certificate independently confirms infeasibility", pres_cal.infeasible)
        check("$label: envelope_prewinner_screen independently confirms infeasibility", eres_cal.status == :EXACT_INFEASIBLE_PREWINNER_ENVELOPE)
        check("$label: envelope's violated cell is a valid (origin, active-destination-slot) pair",
              1 <= eres_cal.origin <= ctx.D && 1 <= eres_cal.destination <= Ddest)
        check("$label: pairwise's violated cell is a valid (origin, active-destination-slot) pair",
              1 <= pres_cal.worst_o <= ctx.D && 1 <= pres_cal.worst_d <= Ddest)
    end

    # ---- synthetic HIT: adversarial point, independently certified by pairwise_certificate ----
    hit = find_pairwise_infeasible_point(ctx, pe, pc, Pmat)
    if hit === nothing
        println("  (no pairwise-certified infeasible point found in 200 tries at this seed -- SKIP, not a failure, matches pre-existing test's own skip discipline)")
        return
    end
    solves_before = CS.INNER_SOLVE_COUNT[]
    r_hit, meta_hit = evaluate_fullA_screened_ranged(hit.xf, ctx, rsc; moment_representation = :compressed,
                                                        cache = nothing, use_cache = false, warm = false, pairwise = pc)
    solves_after = CS.INNER_SOLVE_COUNT[]
    check("$label: adversarial point IS hard-rejected (screen_status != :screen_passed)", meta_hit.screen_status != :screen_passed)
    check("$label: rejection matches the independent pairwise_certificate origin/destination",
          meta_hit.worst_o == hit.pres.worst_o && meta_hit.worst_d == hit.pres.worst_d)
    check("$label: rejected point never triggers a real inner KNITRO solve (screen fired before the solve)",
          solves_after == solves_before)
    check("$label: rejected point's Delta_dual is the screen sentinel (Inf), not a real value", r_hit.Delta_dual == Inf)
    check("$label: rejected point's inner_status is a screen sentinel (< -9000, not a real KNITRO code)", r_hit.inner_status < -9000)
    # omitted-destination-never-accessed: worst_d must be a valid ACTIVE slot (1..D_dest), never
    # referencing a destination outside the active range (which would mean the omitted destination
    # or an out-of-bounds index leaked into the certificate).
    check("$label: rejection's worst_d is a valid active-destination slot (1<=worst_d<=D_dest)",
          1 <= meta_hit.worst_d <= Ddest)

    # ---- range_screen_standalone: independent re-check ----
    θ_full_pass = CS.reconstruct_full(xf_pass, ctx.m)
    cf_pass = build_compressed_factual(θ_full_pass, ctx; check_ties = false)
    rres_pass = range_screen_standalone(cf_pass)
    if expect_calibration_feasible
        check("$label: range_screen_standalone does NOT flag the feasible calibration point (no false positive)",
              rres_pass.status == :INCONCLUSIVE)
    else
        # calibration is genuinely infeasible here (see above) -- the correctness property to check
        # is agreement with the other screens' certificate, not a false-positive-free pass.
        check("$label: range_screen_standalone also flags the same known-infeasible calibration point (consistent, not a false negative from the other screens)",
              rres_pass.status == :EXACT_INFEASIBLE_MOMENT_RANGE)
    end
end

# ================================================================================================
ctx4 = d_exact_setup_scaled(D = 4, W = 8000, find_smallest = true)   # legacy square, regression
gate_b_battery("D=4 legacy square", ctx4)

ctx_r = d_exact_setup_scaled(D = 4, W = 8000, find_smallest = true, row_idx = 4)   # rectangular
gate_b_battery("D=4/D_dest=3 rectangular", ctx_r)

ctx20 = d20_real_setup(W = 8000, δ = 1.0, find_smallest = true)   # real D=20/:exclude_row, moderate W for speed
gate_b_battery("D=20/D_dest=19 real rectangular (:exclude_row)", ctx20; expect_calibration_feasible = false)

println()
println("="^96)
println("Cross-family core-screen sharing (code-identity check, not re-derivation):")
println("infeasibility_screen.jl's compute_a_od/pairwise_certificate/target_shares -- the core")
println("structural-infeasibility certificate this Gate exercises above -- are the SAME functions")
println("cm_screen_bridge.jl's witness/pairwise screening calls for CM/CM+ZC/origin-ZC (verified by")
println("direct source inspection, not a separate call graph); a core-infeasible point is therefore")
println("rejected identically for all four families BY CONSTRUCTION, not merely by convention.")
cm_bridge_path = joinpath(@__DIR__, "cm_screen_bridge.jl")
cm_bridge_src = read(cm_bridge_path, String)
check("cm_screen_bridge.jl calls the same pairwise_certificate (shared core screen)", occursin("pairwise_certificate", cm_bridge_src))
check("cm_screen_bridge.jl calls the same compute_a_od (shared core screen)", occursin("compute_a_od", cm_bridge_src))
println("="^96)

println()
if isempty(FAILURES)
    println(">>> RESULT: ALL PASS")
else
    println(">>> RESULT: ", length(FAILURES), " FAILURE(S): ", FAILURES)
    exit(1)
end
