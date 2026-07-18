# ============================================================================
# Continuation 8, workstream C: high-g branch (THE more important unfinished
# part per the standing brief). Traces profile_Delta(g) = min_A Delta(g,A)
# from the benchmark g_F=0.960965 (known near-zero, doc#1's addendum) OUT
# PAST g=0.99 (where doc#1's grid stopped, Delta~=0.299) toward g_hi=1.0,
# using the CURRENT lower incumbent's own (g,A) point
# (g=0.9967391744173478, kappa=0.005428799948779983) as a MANDATORY start --
# per the brief, NOT optional -- plus continuation from the feasible interior
# and calibration/upper-incumbent seeds. Classifies the upper feasibility
# boundary into one of 4 cases (regular crossing / feasibility wall / outer-
# search failure / combination), with evidence, rather than assuming.
#
# Does NOT initialize only at g=1.0 and infer the branch from its failure --
# builds up from g_F with real intermediate points, per the brief's explicit
# instruction.
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf, Statistics

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const HIGHG_OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, "c8_gammabranch_highg_sweep")
mkpath(HIGHG_OUTDIR)
const MOMENT_REPR = :compressed
const G_F = 0.9609650007465950   # benchmark Frechet value, doc#1's addendum

"Cheap single-start continuation evaluation."
function eval_cont(g::Float64, zf_start::Vector{Float64}; maxtime::Float64 = 15.0)
    profile_delta_at_gamma_c8(g, zf_start, ctx, pe; moment_repr = MOMENT_REPR, maxtime_real = maxtime, hessopt_tag = "sr1")
end

rows = NamedTuple[]   # g, Delta, feasible(inner), knitro_status, source, zfree
function record!(g, res, source)
    has_sol = res.best_zfree !== nothing && isfinite(res.best_Delta)
    push!(rows, (g = g, Delta = has_sol ? res.best_Delta : NaN, feasible = has_sol,
                 knitro_status = res.knitro_status, n_eval = res.n_eval, wall = res.wall,
                 source = source, zfree = has_sol ? copy(res.best_zfree) : nothing))
    return has_sol
end

println("="^78)
println("PHASE 1: coarse continuation g_F=$G_F -> 0.99, step 0.005 (cross-check vs doc#1's grid)")
println("="^78)
g_coarse = collect(G_F:0.005:0.990)
zf_cur = copy(ZFREE_INCUMBENT_C8)
# warm the continuation AT g_F first via a short multistart (calib+incumbent), since g_F is far from
# the incumbent's own basin -- established practice from gamma_profile_multistart.jl.
ms_gF = multistart_profile_at_g(G_F, ZFREE_INCUMBENT_C8, ctx, pe; moment_repr = MOMENT_REPR, maxtime_per_start = 15.0, seed = 90001)
if ms_gF.best !== nothing
    zf_cur = copy(ms_gF.best.zfree)
    @printf("  g_F multistart best: Delta=%.6e (kind=%s)\n", ms_gF.best.Delta, ms_gF.best.kind)
end
for g in g_coarse
    global zf_cur
    res = eval_cont(g, zf_cur; maxtime = 15.0)
    ok = record!(g, res, "coarse_continuation")
    @printf("  g=%.6f  Delta=%.6e  Delta-delta=%+.4e  status=%d n_eval=%d%s\n",
        g, res.best_Delta, res.best_Delta - ctx.δ, res.knitro_status, res.n_eval, ok ? "" : "  <== NO FEASIBLE SOLVE")
    ok && (zf_cur = copy(res.best_zfree))
end

println("\n", "="^78)
println("PHASE 2: fine continuation 0.99 -> 1.0, step 0.001, stop early on repeated failure")
println("="^78)
g_fine_candidates = collect(0.990:0.001:1.000)
n_consec_fail = 0
last_feasible_g = g_coarse[end]
last_feasible_zf = zf_cur
first_failing_g = NaN
for g in g_fine_candidates
    global zf_cur, n_consec_fail, last_feasible_g, last_feasible_zf, first_failing_g
    res = eval_cont(g, zf_cur; maxtime = 15.0)
    ok = record!(g, res, "fine_continuation")
    @printf("  g=%.6f  Delta=%.6e  Delta-delta=%+.4e  status=%d n_eval=%d%s\n",
        g, res.best_Delta, res.best_Delta - ctx.δ, res.knitro_status, res.n_eval, ok ? "" : "  <== NO FEASIBLE SOLVE")
    if ok
        zf_cur = copy(res.best_zfree)
        last_feasible_g = g; last_feasible_zf = copy(res.best_zfree)
        n_consec_fail = 0
    else
        n_consec_fail += 1
        isnan(first_failing_g) && (first_failing_g = g)
        if n_consec_fail >= 3
            println("  3 consecutive continuation failures -- stopping fine sweep, will bisect the transition")
            break
        end
    end
end

println("\n", "="^78)
println("PHASE 3: mandatory start -- the lower incumbent's own (g,A) point, evaluated directly + via reopt")
println("="^78)
xf_lower = x_free_from_w(W_LOWER_INCUMBENT)
r_lower_raw = evaluate_fullA(xf_lower, ctx; cache = nothing, warm = false)
@printf("  RAW (no reopt): g=%.10f  Delta_dual=%.10f  Delta-delta=%.4e  inner_status=%d\n",
    r_lower_raw.gamma_focal_prime, r_lower_raw.Delta_dual, r_lower_raw.Delta_dual - ctx.δ, r_lower_raw.inner_status)
res_lower_reopt = eval_cont(G_LOWER_INCUMBENT, ZFREE_LOWER_INCUMBENT; maxtime = 30.0)
ok_lower = record!(G_LOWER_INCUMBENT, res_lower_reopt, "lower_incumbent_mandatory_start")
@printf("  REOPT from own A: Delta=%.10f  Delta-delta=%+.4e  status=%d n_eval=%d\n",
    res_lower_reopt.best_Delta, res_lower_reopt.best_Delta - ctx.δ, res_lower_reopt.knitro_status, res_lower_reopt.n_eval)
relL2_from_own = ok_lower ? norm(Aod_vec_c8(res_lower_reopt.best_zfree) .- Aod_vec_c8(ZFREE_LOWER_INCUMBENT)) / norm(Aod_vec_c8(ZFREE_LOWER_INCUMBENT)) : NaN
@printf("  relL2(A_reopt - A_lower_incumbent) = %.4e  (0 => lower incumbent's own A IS already the constrained minimizer)\n", relL2_from_own)

println("\n", "="^78)
println("PHASE 4: multistart robustness check bracketing the transition (continuation, incumbent, calib, LOWER-INCUMBENT anchors)")
println("="^78)
# candidate set: last clearly feasible fine point, the lower incumbent's own g, and (if found) the first failing g
g_ms_candidates = sort(unique(filter(!isnan, [last_feasible_g, G_LOWER_INCUMBENT, first_failing_g, 0.999, 0.9995])))
anchors_highg = [("incumbent", ZFREE_INCUMBENT_C8), ("calib", ZFREE_CALIB_C8), ("lower_incumbent", ZFREE_LOWER_INCUMBENT)]
ms_highg_rows = NamedTuple[]
for g in g_ms_candidates
    ms = multistart_profile_at_g(g, last_feasible_zf, ctx, pe; anchors = anchors_highg, moment_repr = MOMENT_REPR,
                                  maxtime_per_start = 15.0, seed = 92000 + round(Int, g*1e7))
    feas = filter(r -> r.feasible, ms.all)
    deltas = [r.Delta for r in feas]
    @printf("  g=%.10f  n_feasible=%d/%d  min=%s  max=%s  best_start=%s\n",
        g, length(feas), length(ms.all),
        isempty(deltas) ? "NaN" : @sprintf("%.6e", minimum(deltas)),
        isempty(deltas) ? "NaN" : @sprintf("%.6e", maximum(deltas)),
        ms.best === nothing ? "NONE" : ms.best.kind)
    for s in ms.all
        @printf("      [%-20s] feasible=%-5s Delta=%s status=%d n_eval=%d\n", s.kind, s.feasible,
            s.feasible ? @sprintf("%.6e", s.Delta) : "NaN", s.knitro_status, s.n_eval)
    end
    push!(ms_highg_rows, (g = g, ms = ms))
end

println("\n", "="^78)
println("PHASE 5: independent LP primal-feasibility certificate at key points")
println("="^78)
# LP-check at: g=1.0 exactly (doc#1 says inner-infeasible there, confirm independently), the lower
# incumbent's own point (should be FEASIBLE -- sanity check on the LP machinery itself), and the
# first continuation-failing point (if any) using the LAST successful A as the test point.
lp_test_points = Tuple{String,Vector{Float64}}[]
push!(lp_test_points, ("lower_incumbent (sanity: should be FEASIBLE)", xf_lower))
push!(lp_test_points, ("g=1.0 with calib A (doc#1: inner-infeasible)", x_free_from_w(vcat(1.0, ZFREE_CALIB_C8))))
push!(lp_test_points, ("g=1.0 with lower-incumbent A", x_free_from_w(vcat(1.0, ZFREE_LOWER_INCUMBENT))))
if !isnan(first_failing_g)
    push!(lp_test_points, ("first_failing_g with last-feasible A", x_free_from_w(vcat(first_failing_g, last_feasible_zf))))
end
lp_results = NamedTuple[]
for (label, xf) in lp_test_points
    lp = lp_feasibility_check_c8(xf, label)
    @printf("  [%-45s] classification=%s  feasible_exact=%s  phase1_max_resid=%s\n",
        label, lp.classification, lp.feasible_exact, isnan(lp.phase1_max_resid) ? "NaN" : @sprintf("%.4e", lp.phase1_max_resid))
    push!(lp_results, lp)
end

# ---- write outputs ----
open(joinpath(HIGHG_OUTDIR, "highg_sweep_rows.csv"), "w") do io
    println(io, "g,Delta,feasible,knitro_status,n_eval,wall,source")
    for r in rows
        println(io, r.g, ",", r.Delta, ",", r.feasible, ",", r.knitro_status, ",", r.n_eval, ",", r.wall, ",", r.source)
    end
end
open(joinpath(HIGHG_OUTDIR, "highg_multistart.csv"), "w") do io
    println(io, "g,start_kind,Delta,feasible,knitro_status,n_eval,wall")
    for row in ms_highg_rows, s in row.ms.all
        println(io, row.g, ",", s.kind, ",", s.Delta, ",", s.feasible, ",", s.knitro_status, ",", s.n_eval, ",", s.wall)
    end
end
open(joinpath(HIGHG_OUTDIR, "lp_feasibility_checks.csv"), "w") do io
    println(io, "label,classification,feasible_exact,phase1_max_resid")
    for lp in lp_results
        println(io, "\"", lp.label, "\",", lp.classification, ",", lp.feasible_exact, ",", lp.phase1_max_resid)
    end
end
println("\nWrote:")
println("  ", joinpath(HIGHG_OUTDIR, "highg_sweep_rows.csv"))
println("  ", joinpath(HIGHG_OUTDIR, "highg_multistart.csv"))
println("  ", joinpath(HIGHG_OUTDIR, "lp_feasibility_checks.csv"))

println("\n", "="^78)
println("SUMMARY (high-g branch)")
println("="^78)
println("last_feasible_g (fine continuation) = ", last_feasible_g)
println("first_failing_g (fine continuation) = ", first_failing_g)
println("lower incumbent g=", G_LOWER_INCUMBENT, " Delta(reopt)=", res_lower_reopt.best_Delta, " relL2-from-own-A=", relL2_from_own)
