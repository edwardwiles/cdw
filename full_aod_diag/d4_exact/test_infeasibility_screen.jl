# ============================================================================
# Correctness battery for infeasibility_screen.jl. Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_infeasibility_screen.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))   # Continuation 10 Section 9: structured dense-materialize, used by compressed_live.jl / infeasibility_screen.jl
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))

using Random, Test

ctx = d4_exact_setup(find_smallest = true)
D = ctx.D
pe = build_pivot_elimination(ctx)
zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

pc = precompute_pairwise_M(ctx)
Pmat = target_shares(ctx)
println("Pmat: min=", minimum(Pmat), " max=", maximum(Pmat), " n_positive=", count(>(0), Pmat), " / ", length(Pmat))

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

# ---------------------------------------------------------------------------
# 1. Known-feasible points: registry candidates must NEVER be screen-rejected
# ---------------------------------------------------------------------------
println("\n== 1. Known-feasible D=4 registry points: zero false positives ==")

feasible_ws = Dict(
    "calibration" => vcat(gp0, zfree0),
    "upper_maxit15_productfd_control" => [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069, 0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011, 1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663, 0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306],
    "upper_maxit40_headline" => [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927],
    "lower_stalled_maxit15" => [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385],
    "upper_lfixcomposite_sr1_60s" => [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845],
    "lower_lfixcomposite_fast_sr1_300s" => [0.9967391744173478, 0.33826763364911505, 0.2756097423805949, 0.3168080759212972, 0.28291467586724917, 1.124214996356822, 1.0586966677239436, 1.0353970705480537, 1.0651065385723435, 0.7972795304932372, 0.7498744164437179, 0.797300832305142, 0.7520482961532734, 1.464240420901755, 1.3955928233005424, 1.4031151818251653],
    "lower_v2" => [0.9973649883022927, 0.4191333995096165, 0.3278704261228879, 0.34822377242086583, 0.3266848028818515, 1.1414170966875377, 1.299758470774316, 1.005176411943463, 1.0769193031619044, 0.8872003122885062, 0.7853545670122154, 0.8376830845864721, 0.7594387919364166, 1.7457822520932191, 1.4007037442409944, 1.5130549509169442],
)

for (name, w) in feasible_ws
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    order = order_destinations(pres, D)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = order)
    check("$name: pairwise cert does NOT reject", !pres.infeasible)
    check("$name: winner-scan does NOT reject", wres.feasible)
end

# ---------------------------------------------------------------------------
# 2. Random small perturbations around calibration (should mostly stay
#    feasible -- these are the kind of points an outer-loop optimizer visits)
# ---------------------------------------------------------------------------
println("\n== 2. Random small perturbations (|step| in [0.01,0.3]): expect mostly feasible, zero false positives ==")
Random.seed!(777)
n_small_infeasible_pairwise = 0
n_small_infeasible_winner = 0
n_small_total = 60
for i in 1:n_small_total
    step = 0.01 + 0.29 * rand()
    dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
    w = vcat(gp0, zfree0 .+ step .* dir)
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    order = order_destinations(pres, D)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = order)
    pres.infeasible && (global n_small_infeasible_pairwise += 1)
    !wres.feasible && (global n_small_infeasible_winner += 1)
    if pres.infeasible && !wres.feasible
        # both agree
    elseif pres.infeasible && wres.feasible
        check("perturb $i: pairwise says infeasible but winner-scan says FEASIBLE (would be a FALSE POSITIVE)", false)
    end
end
println("  small perturbations: pairwise-infeasible=$(n_small_infeasible_pairwise)/$(n_small_total), winner-infeasible=$(n_small_infeasible_winner)/$(n_small_total)")
check("no small-perturbation false positive (pairwise infeasible but winner-scan feasible)", true)  # already checked above via individual asserts

# ---------------------------------------------------------------------------
# 3. Random LARGE perturbations: generate genuine zero-winner infeasible
#    points, cross-validate pairwise certificate against exact ground truth
#    (full_scan winner construction) -- core "zero false positive" test.
# ---------------------------------------------------------------------------
println("\n== 3. Random LARGE perturbations: pairwise-certified infeasible points must ALL be confirmed by exact full-scan ==")
Random.seed!(2024)
n_large = 200
n_pairwise_infeasible = 0
n_winner_infeasible = 0
n_disagree = 0
for i in 1:n_large
    step = 2.0 + 8.0 * rand()
    dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
    w = vcat(gp0, zfree0 .+ step .* dir)
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    wres_full = screen_hard_winners(θ_full, ctx, Pmat; order = 1:D, full_scan = true)
    if pres.infeasible
        global n_pairwise_infeasible += 1
        if wres_full.feasible
            global n_disagree += 1
            check("LARGE perturb $i: pairwise certified infeasible but EXACT full-scan says FEASIBLE -- FALSE POSITIVE", false)
        end
    end
    !wres_full.feasible && (global n_winner_infeasible += 1)
end
println("  large perturbations: pairwise-infeasible=$(n_pairwise_infeasible)/$(n_large), exact-winner-infeasible=$(n_winner_infeasible)/$(n_large), disagreements(FALSE POSITIVES)=$(n_disagree)")
check("zero false positives: every pairwise-certified-infeasible point IS exactly infeasible", n_disagree == 0)
check("pairwise catches a nonzero fraction of the genuine infeasible points found (expected false negatives OK, but >0 catch rate)", n_pairwise_infeasible > 0)

# ---------------------------------------------------------------------------
# 4. Order-independence at feasible points (Section 1's explicit requirement:
#    "Order changes must not affect output values at feasible points")
# ---------------------------------------------------------------------------
println("\n== 4. Order-independence at feasible points ==")
xf_cal = x_free_from_w(vcat(gp0, zfree0))
θ_full_cal = CS.reconstruct_full(xf_cal, ctx.m)
wres_natural = screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D)
Random.seed!(31337)
perm = randperm(D)
wres_perm = screen_hard_winners(θ_full_cal, ctx, Pmat; order = perm)
check("order-independence: feasible flag matches", wres_natural.feasible == wres_perm.feasible)
check("order-independence: winner matrix bit-identical", wres_natural.winner == wres_perm.winner)
check("order-independence: wval matrix bit-identical", wres_natural.wval == wres_perm.wval)
check("order-independence: win_counts bit-identical", wres_natural.win_counts == wres_perm.win_counts)

# ---------------------------------------------------------------------------
# 5. Early-exit vs full-scan consistency on infeasible points
# ---------------------------------------------------------------------------
println("\n== 5. Early-exit vs full-scan agree on feasible/infeasible decision ==")
Random.seed!(555)
n_check = 40
n_mismatch = 0
for i in 1:n_check
    step = 2.0 + 8.0 * rand()
    dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
    w = vcat(gp0, zfree0 .+ step .* dir)
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    order = order_destinations(pres, D)
    w_early = screen_hard_winners(θ_full, ctx, Pmat; order = order, full_scan = false)
    w_full = screen_hard_winners(θ_full, ctx, Pmat; order = order, full_scan = true)
    if w_early.feasible != w_full.feasible
        global n_mismatch += 1
    end
end
check("early-exit / full-scan decision agreement (0 mismatches over $n_check trials)", n_mismatch == 0)

# ---------------------------------------------------------------------------
# 6. Tie-safety: deliberately construct an exact price tie and confirm the
#    screen's win-count does NOT under-count / false-positive on it.
# ---------------------------------------------------------------------------
println("\n== 6. Tie-safety: deliberate exact price tie must not cause a false positive ==")
# Force origins 1 and 2 to have IDENTICAL Aod_theta[.,d] for one destination d=1,
# and set every OTHER origin's A very low at d=1 so only 1 and 2 can ever win --
# if the tie-unsafe (first-index-only) win count were used, origin 2 would show
# zero wins at d=1 (origin 1 always "wins" the tie), a false positive.
w_tie = copy(vcat(gp0, zfree0))
xf_tie = x_free_from_w(w_tie)
θ_full_tie = CS.reconstruct_full(xf_tie, ctx.m)
Aoff = ctx.Aod_offset
# reshape indices: Aod_θ[o,d] is at position Aod_offset + o + (d-1)*D (column-major, matches reshape(...,(D,D)))
idx(o, d) = Aoff + o + (d - 1) * D
θ_full_tie[idx(2, 1)] = θ_full_tie[idx(1, 1)]   # exact bit-identical Aod_theta[1,1] and Aod_theta[2,1]
a_tie = compute_a_od(θ_full_tie, ctx)
println("  a[1,1]=", a_tie[1,1], "  a[2,1]=", a_tie[2,1], "  bit-identical=", a_tie[1,1] == a_tie[2,1])
wres_tie = screen_hard_winners(θ_full_tie, ctx, Pmat; order = 1:D, full_scan = true)
check("tie test: origin 1 has >0 wins at destination 1", wres_tie.win_counts[1,1] > 0)
check("tie test: origin 2 has >0 wins at destination 1 (would be 0 under a tie-UNSAFE count)", wres_tie.win_counts[2,1] > 0)

# ---------------------------------------------------------------------------
# 7. Extreme-draw witness (Section 3): correctness cross-check against the
#    exact winner-scan ground truth, at a handful of (o,d) pairs.
# ---------------------------------------------------------------------------
println("\n== 7. Extreme-draw witness correctness (D=4) ==")
ew = build_extreme_draw_witness(ctx)
Bmat = hard_score_B(ctx)
n_witness_checked = 0
n_witness_mismatch = 0
for (name, w) in feasible_ws
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = 1:D, full_scan = true)
    for d in 1:D, o in 1:D
        Pmat[o,d] > 0 || continue
        global n_witness_checked += 1
        exists, s, ntested, csize = query_witness(o, d, a, Bmat, ew)
        ground_truth = wres.win_counts[o,d] > 0
        if exists != ground_truth
            global n_witness_mismatch += 1
            check("witness($name,o=$o,d=$d): exists=$exists vs ground_truth(win_counts>0)=$ground_truth", false)
        end
    end
end
println("  witness checked $(n_witness_checked) (o,d) pairs across $(length(feasible_ws)) points, mismatches=$(n_witness_mismatch)")
check("extreme-draw witness matches exact ground truth on all feasible-point (o,d) pairs", n_witness_mismatch == 0)

# also check witness on the large-perturbation infeasible set
Random.seed!(2024)
n_witness_infeas_checked = 0
n_witness_infeas_mismatch = 0
for i in 1:30
    step = 2.0 + 8.0 * rand()
    dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
    w = vcat(gp0, zfree0 .+ step .* dir)
    xf = x_free_from_w(w)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    a = compute_a_od(θ_full, ctx)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = 1:D, full_scan = true)
    for d in 1:D, o in 1:D
        Pmat[o,d] > 0 || continue
        global n_witness_infeas_checked += 1
        exists, s, ntested, csize = query_witness(o, d, a, Bmat, ew)
        ground_truth = wres.win_counts[o,d] > 0
        if exists != ground_truth
            global n_witness_infeas_mismatch += 1
        end
    end
end
println("  witness checked $(n_witness_infeas_checked) (o,d) pairs across 30 large-perturbation points, mismatches=$(n_witness_infeas_mismatch)")
check("extreme-draw witness matches exact ground truth on large-perturbation (o,d) pairs too", n_witness_infeas_mismatch == 0)

# ---------------------------------------------------------------------------
# 8. Integration: evaluate_fullA_screened must match evaluate_fullA_fast
#    EXACTLY at every feasible point (dense), and evaluate_fullA_screened
#    with :compressed must match dense to the SAME tolerance the existing
#    :compressed mode already achieves.
# ---------------------------------------------------------------------------
println("\n== 8. Integration: evaluate_fullA_screened matches evaluate_fullA_fast at feasible points ==")
dd_match(a, b; atol) = (isnan(a) && isnan(b)) ? true : isapprox(a, b; atol = atol)

for (name, w) in feasible_ws
    xf = x_free_from_w(w)
    r_direct, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, use_cache = false, warm = false)
    r_screened, meta = evaluate_fullA_screened(xf, ctx; moment_representation = :dense, cache = nothing, use_cache = false, warm = false)
    check("$name: screened matches direct inner_status", r_direct.inner_status == r_screened.inner_status)
    check("$name: screened matches direct Delta_dual (or both NaN)", dd_match(r_direct.Delta_dual, r_screened.Delta_dual; atol=1e-12))
    check("$name: screened matches direct winner_hash", r_direct.winner_hash == r_screened.winner_hash)
    check("$name: screen_status == :screen_passed", meta.screen_status == :screen_passed)

    r_screened_c, meta_c = evaluate_fullA_screened(xf, ctx; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false)
    check("$name: screened-compressed matches direct Delta_dual (tol 1e-8, or both NaN)", dd_match(r_direct.Delta_dual, r_screened_c.Delta_dual; atol=1e-8))
    check("$name: screened-compressed winner_hash matches", r_direct.winner_hash == r_screened_c.winner_hash)
end

println("\n== 9. Integration: infeasible point returns structured status, Delta_dual=Inf, no KNITRO call ==")
Random.seed!(9999)
found_infeasible = false
local xf_bad
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
        break
    end
end
if found_infeasible
    solves_before = CS.INNER_SOLVE_COUNT[]
    r_bad, meta_bad = evaluate_fullA_screened(xf_bad, ctx; moment_representation = :dense, cache = nothing, use_cache = false, warm = false)
    solves_after = CS.INNER_SOLVE_COUNT[]
    check("infeasible point: Delta_dual == Inf", r_bad.Delta_dual == Inf)
    check("infeasible point: inner_status is a screen sentinel (not a real KNITRO code)", r_bad.inner_status < -9000)
    check("infeasible point: screen_status == :pairwise_certified_infeasible", meta_bad.screen_status == :pairwise_certified_infeasible)
    check("infeasible point: zero NEW inner KNITRO solves performed", solves_after == solves_before)
else
    println("  (no pairwise-certified infeasible point found in 200 tries at this seed -- skipping test 9, not a failure)")
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
println("============================================================")
n_fail == 0 || error("$n_fail correctness check(s) FAILED -- see above")
