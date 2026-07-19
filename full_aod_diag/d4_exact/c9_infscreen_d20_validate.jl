# ============================================================================
# Continuation 9 (branch c9-infeasibility-screen): D=20 real-data validation
# + benchmark of infeasibility_screen.jl.
#
# Sections:
#  A. Setup (W=80000), zero-false-positive check on the 4 known-feasible
#     outer points from docs/fullA_D20_W80k_microbenchmark.md (calibration,
#     gravity-tangent, upper branch, lower branch).
#  B. RETROACTIVE CHECK on the historical W=8000 "-300 everywhere" finding
#     (docs/fullA_D20_production_path_audit.md / continuation9_interim_handoff.md
#     Phase 1): does the exact screen certify these points structurally
#     feasible (positive win counts everywhere) or infeasible (zero-winner)?
#  C. Pairwise-certificate / winner-scan timing at W=80000 (draw-free +
#     draw-based costs), vs the existing dense value-callback cost.
#  D. Random A-perturbation sweep: false-positive audit (pairwise vs exact
#     full-scan) and false-negative rate, at D=20 scale, PURE SCREEN cost
#     only (no KNITRO) so a large trial count is affordable.
#  E. Realistic infeasible-workload benchmark (task spec Section 5): time to
#     rejection / work avoided across the 5 requested workload classes.
#  F. Extreme-draw witness (Section 3): preprocessing time, memory, query
#     time, fraction resolved without full winner construction, at
#     W=80,000. W=800,000 probed separately for memory/preprocessing ONLY
#     (c9_infscreen_witness_w800k.jl) per the standing safety discipline.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_infscreen_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_infscreen_d20_validate.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end
const FEASIBLE_CODES = (0, -100, -101, -103)
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted) / n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end], mean_s = μ)
end

# ============================================================================
# SECTION A: setup + known-feasible zero-false-positive check
# ============================================================================
logprint("\n", "="^90); logprint("SECTION A: setup (W=80000) + known-feasible zero-FP check"); logprint("="^90)

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
logprint("d20_real_setup(W=80000) wall=", round(t_setup, digits=2), "s  VmHWM=", round(vmhwm_kb()/1e6, digits=2), " GB")
D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)

t0 = time()
pc = precompute_pairwise_M(ctx)
t_pc_build = time() - t0
Pmat = target_shares(ctx)
logprint("precompute_pairwise_M: wall=", round(t_pc_build, digits=4), "s   Pmat: n_positive=", count(>(0), Pmat), " / ", length(Pmat), "  min(nonzero)=", minimum(filter(>(0), vec(Pmat))))

# ---- 4 known outer points, reconstructed exactly as c9_w80k_microbenchmark.jl did ----
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
w_nat = vcat(xf_nat[1], zfree_nat)
Random.seed!(9001)
dir2 = randn(length(w_nat) - 1); dir2 ./= norm(dir2)
w2 = copy(w_nat); w2[2:end] .+= 0.02 .* dir2
xf2 = x_free_from_w(w2, pe)

gp_hi = ctx.bounds.γp_hi; gp_lo = ctx.bounds.γp_lo
xf3 = vcat(clamp(gp0 * 1.01, gp_lo, gp_hi), xf_nat[2:end])   # upper branch, offset=0.01 (per W80k doc, first try worked)
xf4 = vcat(clamp(gp0 * 0.99, gp_lo, gp_hi), xf_nat[2:end])   # lower branch, offset=0.01

known_points = [
    ("calibration", xf_nat),
    ("gravity_tangent", xf2),
    ("upper_branch_offset01", xf3),
    ("lower_branch_offset01", xf4),
]

fp_rows = NamedTuple[]
for (name, xf) in known_points
    θ_full = CS.reconstruct_full(xf, ctx.m)
    t0 = time(); a = compute_a_od(θ_full, ctx); t_a = time() - t0
    t0 = time(); pres = pairwise_certificate(a, pc, Pmat); t_pw = time() - t0
    order = order_destinations(pres, D)
    t0 = time(); wres = screen_hard_winners(θ_full, ctx, Pmat; order = order); t_ws = time() - t0
    logprint("  ", name, ": pairwise.infeasible=", pres.infeasible, " (worst_slack=", round(pres.worst_slack, digits=4),
             ")  winner_scan.feasible=", wres.feasible, "  t_a=", round(t_a*1000,digits=2), "ms  t_pairwise=",
             round(t_pw*1000,digits=3), "ms  t_winnerscan=", round(t_ws*1000,digits=1), "ms")
    push!(fp_rows, (point=name, pairwise_infeasible=pres.infeasible, winner_scan_feasible=wres.feasible,
                     worst_slack=pres.worst_slack, t_a_ms=t_a*1000, t_pairwise_ms=t_pw*1000, t_winnerscan_ms=t_ws*1000))
end
write_csv_rows(joinpath(OUTDIR, "sectionA_known_feasible.csv"), fp_rows)
n_fp_A = count(r -> r.pairwise_infeasible || !r.winner_scan_feasible, fp_rows)
logprint("SECTION A false positives on known-feasible D=20 points: ", n_fp_A, " / ", length(fp_rows))

# ============================================================================
# SECTION B: RETROACTIVE historical W=8000 "-300 everywhere" check
# ============================================================================
logprint("\n", "="^90); logprint("SECTION B: retroactive W=8000 check (historical -300 finding)"); logprint("="^90)

t0 = time()
ctx8k = d20_real_setup(W = 8000)
t_setup8k = time() - t0
logprint("d20_real_setup(W=8000) wall=", round(t_setup8k, digits=2), "s")
gp0_8k = ctx8k.θ0_up[3+D]
xf_nat_8k = ctx8k.θ0_up[ctx8k.free_idx]
pe8k = build_pivot_elimination(ctx8k)
pc8k = precompute_pairwise_M(ctx8k)
Pmat8k = target_shares(ctx8k)

zfree_nat_8k = pivot_reduce(log.(reshape(xf_nat_8k[2:end], D, D)), pe8k)
w_nat_8k = vcat(xf_nat_8k[1], zfree_nat_8k)
Random.seed!(9001)
dir2_8k = randn(length(w_nat_8k) - 1); dir2_8k ./= norm(dir2_8k)
w2_8k = copy(w_nat_8k); w2_8k[2:end] .+= 0.02 .* dir2_8k
xf2_8k = x_free_from_w(w2_8k, pe8k)
gp_hi8k = ctx8k.bounds.γp_hi; gp_lo8k = ctx8k.bounds.γp_lo
xf3_8k = vcat(clamp(gp0_8k * 1.01, gp_lo8k, gp_hi8k), xf_nat_8k[2:end])
xf4_8k = vcat(clamp(gp0_8k * 0.99, gp_lo8k, gp_hi8k), xf_nat_8k[2:end])

points_8k = [("calibration_W8000", xf_nat_8k), ("gravity_tangent_W8000", xf2_8k),
             ("upper_branch_W8000", xf3_8k), ("lower_branch_W8000", xf4_8k)]

retro_rows = NamedTuple[]
for (name, xf) in points_8k
    θ_full = CS.reconstruct_full(xf, ctx8k.m)
    a = compute_a_od(θ_full, ctx8k)
    pres = pairwise_certificate(a, pc8k, Pmat8k)
    order = order_destinations(pres, D)
    wres = screen_hard_winners(θ_full, ctx8k, Pmat8k; order = order, full_scan = true)
    n_zero_win_pairs = sum((Pmat8k .> 0) .& (wres.win_counts .== 0))
    logprint("  ", name, ": pairwise.infeasible=", pres.infeasible, "  winner_scan.feasible=", wres.feasible,
             "  n_zero_win_positive_share_pairs=", n_zero_win_pairs, " / ", count(>(0), Pmat8k))
    # ALSO run the REAL dense oracle at W=8000 to confirm it is indeed nStatus=-300 as historically found
    t0 = time()
    r_real = evaluate_fullA_fast(xf, ctx8k; cache = nothing, use_cache = false, warm = false)[1]
    t_real = time() - t0
    logprint("    REAL dense solve: inner_status=", r_real.inner_status, "  Delta_dual=", r_real.Delta_dual,
             "  wall=", round(t_real, digits=2), "s")
    push!(retro_rows, (point=name, pairwise_infeasible=pres.infeasible, winner_scan_feasible=wres.feasible,
                        n_zero_win_positive_share_pairs=n_zero_win_pairs, n_positive_share_pairs=count(>(0), Pmat8k),
                        real_inner_status=r_real.inner_status, real_Delta_dual=r_real.Delta_dual, real_wall_s=t_real))
end
write_csv_rows(joinpath(OUTDIR, "sectionB_w8000_retroactive.csv"), retro_rows)

all_structurally_feasible_8k = all(r -> r.winner_scan_feasible, retro_rows)
all_real_300_8k = all(r -> r.real_inner_status == -300, retro_rows)
logprint("\n*** RETROACTIVE FINDING: W=8000 historical -300 points -- structurally feasible (exact screen)=",
         all_structurally_feasible_8k, "; ALL real dense solves returned -300=", all_real_300_8k)
if all_structurally_feasible_8k && all_real_300_8k
    logprint("*** CONFIRMS: the W=8000 '-300 everywhere' phenomenon is NOT the zero-winner structural")
    logprint("*** infeasibility this screen targets -- every positive-target-share (o,d) pair DOES have")
    logprint("*** >=1 winning draw at W=8000 (exact screen says feasible), yet KNITRO's inner CC dual solve")
    logprint("*** still reports -300 (unbounded/infeasible). This is a genuinely DIFFERENT failure mode --")
    logprint("*** plausibly a too-few-effective-draws / near-degenerate-moment-system numerical issue, NOT")
    logprint("*** the structural zero-winner case. REFUTES the 'plausible but not confirmed' hypothesis in")
    logprint("*** the task brief, at least for these 4 representative points.")
elseif !all_structurally_feasible_8k
    logprint("*** CONFIRMS the hypothesis for at least one point: a zero-winner structural infeasibility")
    logprint("*** WAS detected by the exact screen among these W=8000 points.")
end

# ============================================================================
# SECTION C: pairwise / winner-scan timing at W=80000 (calibration point)
# ============================================================================
logprint("\n", "="^90); logprint("SECTION C: screen timing at W=80000 vs the dense value-callback baseline"); logprint("="^90)

θ_full_cal = CS.reconstruct_full(xf_nat, ctx.m)
# warm-up
for _ in 1:2
    compute_a_od(θ_full_cal, ctx)
    pairwise_certificate(compute_a_od(θ_full_cal, ctx), pc, Pmat)
    screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D)
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end

N = 8
t_pcbuild = [( t0=time(); precompute_pairwise_M(ctx); time()-t0 ) for _ in 1:3]   # only 3 -- this is the expensive O(D^2 W) one-time cost
t_a = [( t0=time(); compute_a_od(θ_full_cal, ctx); time()-t0 ) for _ in 1:N]
t_pw = [( t0=time(); pairwise_certificate(compute_a_od(θ_full_cal, ctx), pc, Pmat); time()-t0 ) for _ in 1:N]
t_order = [( t0=time(); order_destinations(pairwise_certificate(compute_a_od(θ_full_cal, ctx), pc, Pmat), D); time()-t0 ) for _ in 1:N]
t_ws = [( t0=time(); screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D); time()-t0 ) for _ in 1:N]
t_dense_value = [( t0=time(); evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense); time()-t0 ) for _ in 1:N]

for (label, times) in [("pairwise_M_build(one-time)", t_pcbuild), ("compute_a_od", t_a), ("pairwise_certificate(with a)", t_pw),
                        ("order_destinations", t_order), ("screen_hard_winners(full_scan=false, feasible->all D)", t_ws),
                        ("evaluate_fullA_fast(dense, warm, REFERENCE)", t_dense_value)]
    r = stats_row(label, times)
    logprint(@sprintf("  %-52s N=%d median=%.4fms  min=%.4fms  max=%.4fms", label, r.n, r.median_s*1000, r.min_s*1000, r.max_s*1000))
end
write_csv_rows(joinpath(OUTDIR, "sectionC_timing.csv"),
    [(component="pairwise_M_build", stats_row("x",t_pcbuild)...), (component="compute_a_od", stats_row("x",t_a)...),
     (component="pairwise_certificate", stats_row("x",t_pw)...), (component="order_destinations", stats_row("x",t_order)...),
     (component="screen_hard_winners_feasible", stats_row("x",t_ws)...), (component="dense_value_reference", stats_row("x",t_dense_value)...)])

logprint("\nSaved logs/CSVs to ", OUTDIR)
logprint("c9_infscreen_d20_validate.jl SECTION A-C complete at ", now())
