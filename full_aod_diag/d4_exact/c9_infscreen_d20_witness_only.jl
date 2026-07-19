# ============================================================================
# Standalone rerun of c9_infscreen_d20_workloads.jl's Section F (extreme-draw
# witness benchmark, W=80000) after fixing a dead-code bug that crashed the
# first run AFTER Sections D and E had already completed successfully and
# saved their CSVs (see results/fullA_d4/<commit>/c9_infscreen_d20_workloads/).
# Kept as a separate small script rather than re-running the whole (expensive,
# real-KNITRO) D+E workload benchmark a second time.
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
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_infscreen_d20_workloads")
mkpath(OUTDIR)
println("c9_infscreen_d20_witness_only.jl starting at ", now()); flush(stdout)

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

t0 = time()
ctx = d20_real_setup(W = 80000)
println("d20_real_setup(W=80000) wall=", round(time()-t0, digits=2), "s  VmHWM=", round(vmhwm_kb()/1e6,digits=2), " GB"); flush(stdout)
D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
Pmat = target_shares(ctx)

println("\n", "="^90); println("SECTION F: extreme-draw witness benchmark, W=80000"); println("="^90); flush(stdout)

vmhwm_before_witness = vmhwm_kb()
t0 = time()
ew = build_extreme_draw_witness(ctx)
t_build = time() - t0
vmhwm_after_witness = vmhwm_kb()
mem_pairs = sizeof(ew.pairs[1,2].idx) + sizeof(ew.pairs[1,2].val)
mem_total_est = mem_pairs * D * (D-1)
println("build_extreme_draw_witness: wall=", round(t_build, digits=2), "s  VmHWM before=", round(vmhwm_before_witness/1e6,digits=2),
        " GB  after=", round(vmhwm_after_witness/1e6,digits=2), " GB")
println("estimated structure memory (D*(D-1) sorted pairs x (Int32 idx + Float64 val)): ", round(mem_total_est/1e9, digits=3), " GB")
flush(stdout)

Bmat = hard_score_B(ctx)
θ_full_cal = CS.reconstruct_full(xf_nat, ctx.m)
a_cal = compute_a_od(θ_full_cal, ctx)
wres_cal = screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D, full_scan = true)

n_queries = 0
query_times = Float64[]
candidate_sizes = Int[]
for d in 1:D, o in 1:D
    global n_queries
    Pmat[o,d] > 0 || continue
    t0 = time()
    exists, s, ntested, csize = query_witness(o, d, a_cal, Bmat, ew)
    tq = time() - t0
    push!(query_times, tq); push!(candidate_sizes, csize)
    n_queries += 1
    ground_truth = wres_cal.win_counts[o,d] > 0
    @assert exists == ground_truth "witness mismatch at o=$o d=$d"
end
n_resolved_cheap = count(<(size(ctx.U,1)), candidate_sizes)
println("query_witness: N=", n_queries, " queries at calibration point")
println("  query time: median=", round(median(query_times)*1e6,digits=1), "us  mean=", round(mean(query_times)*1e6,digits=1),
        "us  max=", round(maximum(query_times)*1e6,digits=1), "us")
println("  candidate-set size (best rival's count): median=", median(candidate_sizes), "  mean=", round(mean(candidate_sizes),digits=1),
        "  (out of W=", size(ctx.U,1), ")")
println("  fraction with candidate-set < W: ", round(100*n_resolved_cheap/n_queries, digits=1), "%")
flush(stdout)

t_ws_ref = @elapsed screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D)
t_per_query = median(query_times)
n_positive_pairs = count(>(0), Pmat)
cost_full_scan_per_point = t_ws_ref
cost_witness_per_point = n_positive_pairs * t_per_query
breakeven_points = t_build / max(cost_full_scan_per_point - cost_witness_per_point, 1e-12)
println("\nBreak-even analysis (W=80000):")
println("  one-time build cost: ", round(t_build, digits=2), "s")
println("  full destination-major winner-scan (screen_hard_winners), feasible-path cost: ", round(cost_full_scan_per_point*1000, digits=2), "ms/outer-point")
println("  witness-based per-outer-point cost (querying all $n_positive_pairs positive-share pairs): ",
        round(cost_witness_per_point*1000, digits=2), "ms/outer-point")
if cost_witness_per_point < cost_full_scan_per_point
    println("  witness IS cheaper per-point once built; break-even after ~", round(breakeven_points, digits=1), " outer-point evaluations")
else
    println("  witness querying ALL positive-share pairs is NOT cheaper per-point than the direct destination-major scan at D=20/W=80000")
    println("  (its real value is resolving a SMALL SUBSET of (o,d) pairs cheaply -- e.g. as a destination-ordering oracle -- not replacing the full scan)")
end
flush(stdout)

write_csv_rows(joinpath(OUTDIR, "sectionF_witness_benchmark.csv"),
    [(D=D, W=size(ctx.U,1), build_wall_s=t_build, mem_est_GB=mem_total_est/1e9,
      n_queries=n_queries, query_median_us=median(query_times)*1e6, query_mean_us=mean(query_times)*1e6,
      candidate_size_median=median(candidate_sizes), frac_resolved_lt_W=n_resolved_cheap/n_queries,
      full_scan_ms=cost_full_scan_per_point*1000, witness_allpairs_ms=cost_witness_per_point*1000,
      breakeven_outer_points=breakeven_points)])

println("\nc9_infscreen_d20_witness_only.jl DONE at ", now())
