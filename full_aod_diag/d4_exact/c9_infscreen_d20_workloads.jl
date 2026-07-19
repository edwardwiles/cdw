# ============================================================================
# Continuation 9 (branch c9-infeasibility-screen): D=20 real-data, W=80000.
# Sections D (random A-perturbation false-positive/false-negative sweep,
# pure-screen cost only) + E (realistic infeasible-workload benchmark, task
# spec Section 5) + F (extreme-draw witness benchmark at W=80000).
# Run AFTER c9_infscreen_d20_validate.jl (separate process, own context) --
# kept as a separate file so a failure/rerun of one doesn't waste the other's
# already-completed work, matching this investigation's small/reviewable-
# script convention.
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
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_infscreen_d20_workloads")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c9_infscreen_d20_workloads.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

t0 = time()
ctx = d20_real_setup(W = 80000)
logprint("d20_real_setup(W=80000) wall=", round(time()-t0, digits=2), "s  VmHWM=", round(vmhwm_kb()/1e6,digits=2), " GB")
D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
Pmat = target_shares(ctx)
pc = precompute_pairwise_M(ctx)
n_free_A = length(zfree_nat)
logprint("D=", D, "  n_free_A(pivot-reduced)=", n_free_A, "  n_positive_share_pairs=", count(>(0), Pmat), "/", length(Pmat))

# warm-up (untimed)
θ_full_cal = CS.reconstruct_full(xf_nat, ctx.m)
screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D)
pairwise_certificate(compute_a_od(θ_full_cal, ctx), pc, Pmat)

# ============================================================================
# SECTION D: random A-perturbation sweep -- pure screen cost only (no KNITRO)
# ============================================================================
logprint("\n", "="^90); logprint("SECTION D: random A-perturbation sweep (pure screen, no KNITRO)"); logprint("="^90)

struct PerturbTrial
    step::Float64
    xf::Vector{Float64}
    θ_full::Vector{Float64}
    pairwise_infeasible::Bool
    winner_feasible::Bool
    worst_slack::Float64
    t_pairwise_s::Float64
    t_winnerscan_s::Float64
    stage::Int
end

Random.seed!(20260719)
STEPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_PER_STEP = 25
trials = PerturbTrial[]
for step in STEPS
    for i in 1:N_PER_STEP
        dir = randn(n_free_A); dir ./= norm(dir)
        w = vcat(gp0, zfree_nat .+ step .* dir)
        xf = x_free_from_w(w, pe)
        θ_full = CS.reconstruct_full(xf, ctx.m)
        t0 = time(); a = compute_a_od(θ_full, ctx); pres = pairwise_certificate(a, pc, Pmat); t_pw = time() - t0
        order = order_destinations(pres, D)
        t0 = time(); wres = screen_hard_winners(θ_full, ctx, Pmat; order = order); t_ws = time() - t0
        push!(trials, PerturbTrial(step, xf, θ_full, pres.infeasible, wres.feasible, pres.worst_slack, t_pw, t_ws, wres.stage))
    end
    n_this = count(t -> t.step == step, trials)
    n_pw_infeas = count(t -> t.step == step && t.pairwise_infeasible, trials)
    n_wc_infeas = count(t -> t.step == step && !t.winner_feasible, trials)
    logprint(@sprintf("  step=%5.1f  N=%d  pairwise_infeasible=%d  winner_infeasible=%d", step, n_this, n_pw_infeas, n_wc_infeas))
end

n_false_positive = count(t -> t.pairwise_infeasible && t.winner_feasible, trials)
n_total = length(trials)
n_pw_infeas_total = count(t -> t.pairwise_infeasible, trials)
n_wc_infeas_total = count(t -> !t.winner_feasible, trials)
n_false_negative = count(t -> !t.pairwise_infeasible && !t.winner_feasible, trials)
logprint("\nSECTION D SUMMARY (N=$n_total total, D=20/W=80000):")
logprint("  pairwise-certified infeasible: $n_pw_infeas_total / $n_total")
logprint("  exact winner-scan infeasible:  $n_wc_infeas_total / $n_total")
logprint("  FALSE POSITIVES (pairwise infeasible but exact winner-scan feasible): $n_false_positive")
logprint("  false negatives (pairwise feasible but exact winner-scan infeasible): $n_false_negative  (expected/acceptable, catch rate=",
         round(100*(n_wc_infeas_total>0 ? n_pw_infeas_total/n_wc_infeas_total : NaN), digits=1), "%)")

# early-exit savings: for genuinely infeasible points, how many of the D destinations were scanned
# before rejection vs the full D?
infeas_trials = filter(t -> !t.winner_feasible, trials)
if !isempty(infeas_trials)
    stages = [t.stage for t in infeas_trials]
    logprint("  early-exit stage among exact-infeasible points: median=", median(stages), "/", D,
             "  mean=", round(mean(stages), digits=1), "/", D, "  (lower = more work saved)")
end

write_csv_rows(joinpath(OUTDIR, "sectionD_perturbation_sweep.csv"),
    [(step=t.step, pairwise_infeasible=t.pairwise_infeasible, winner_feasible=t.winner_feasible,
      worst_slack=t.worst_slack, t_pairwise_ms=t.t_pairwise_s*1000, t_winnerscan_ms=t.t_winnerscan_s*1000,
      stage=t.stage) for t in trials])
logprint("zero false positives confirmed: ", n_false_positive == 0)
n_false_positive == 0 || error("SECTION D found $n_false_positive FALSE POSITIVE(S) -- STOP, this is a correctness bug")

# ============================================================================
# SECTION E: realistic infeasible-workload benchmark (task spec Section 5)
# ============================================================================
logprint("\n", "="^90); logprint("SECTION E: realistic workload benchmark (5 classes)"); logprint("="^90)

# Class 1: feasible points (calibration + gravity-tangent + small perturbations)
class1 = NamedTuple[]
push!(class1, (name="calibration", xf=xf_nat))
Random.seed!(9001)
dir2 = randn(n_free_A); dir2 ./= norm(dir2)
w2 = vcat(gp0, zfree_nat .+ 0.02 .* dir2)
push!(class1, (name="gravity_tangent", xf=x_free_from_w(w2, pe)))
for i in 1:3
    dir = randn(n_free_A); dir ./= norm(dir)
    w = vcat(gp0, zfree_nat .+ (0.05 + 0.1*i) .* dir)
    push!(class1, (name="small_perturb_$i", xf=x_free_from_w(w, pe)))
end

# Class 2: pairwise-certified zero-winner points (pick 3 from Section D's step=8/16 trials)
class2_src = filter(t -> t.pairwise_infeasible, trials)
class2 = [(name="pairwise_certified_$(i)", xf=t.xf) for (i,t) in enumerate(class2_src[1:min(3,end)])]

# Class 3: zero-winner points NOT caught by pairwise certificate (from Section D)
class3_src = filter(t -> !t.pairwise_infeasible && !t.winner_feasible, trials)
class3 = [(name="uncaught_zerowinner_$(i)", xf=t.xf) for (i,t) in enumerate(class3_src[1:min(3,end)])]
logprint("Class 3 (uncaught zero-winner) candidates found: ", length(class3_src))

# Class 4: positive-winner-count but moment-infeasible (nStatus != feasible despite screen pass)
# -- constructed via the historical W=8000 retroactive finding (Section B of
# c9_infscreen_d20_validate.jl already tests this directly); here we ALSO probe
# whether any MODERATE (screen-passing) A-perturbation at W=80000 itself produces a
# genuine KNITRO numerical failure (a from-first-principles example at W=80000).
class4_candidates = filter(t -> !t.pairwise_infeasible && t.winner_feasible, trials)
Random.seed!(4242)
shuffle!(class4_candidates)
class4 = NamedTuple[]

# Class 5: feasible points OUTSIDE the divergence budget (small delta context)
ctx_smalldelta = d20_real_setup(W = 80000, δ = 0.001)
class5 = [(name="outside_budget_calibration", xf=xf_nat, ctx=ctx_smalldelta)]

FEASIBLE_CODES = (0, -100, -101, -103)

function bench_class(class_name, pts; ctx_override = nothing, screen_only = false)
    rows = NamedTuple[]
    for p in pts
        use_ctx = haskey(p, :ctx) ? p.ctx : (ctx_override === nothing ? ctx : ctx_override)
        θ_full = CS.reconstruct_full(p.xf, use_ctx.m)
        Pmat_l = target_shares(use_ctx)
        pc_l = use_ctx === ctx ? pc : precompute_pairwise_M(use_ctx)

        t0 = time()
        a = compute_a_od(θ_full, use_ctx)
        pres = pairwise_certificate(a, pc_l, Pmat_l)
        t_pairwise = time() - t0

        rejected_at_pairwise = pres.infeasible
        wres = nothing
        t_winnerscan = 0.0
        if !rejected_at_pairwise
            order = order_destinations(pres, use_ctx.D)
            t0 = time()
            wres = screen_hard_winners(θ_full, use_ctx, Pmat_l; order = order)
            t_winnerscan = time() - t0
        end
        rejected_at_winner = wres !== nothing && !wres.feasible
        t_screen_total = t_pairwise + t_winnerscan

        t_full = NaN; real_status = missing; real_delta = NaN
        if !rejected_at_pairwise && !rejected_at_winner
            t0 = time()
            r_full, _ = evaluate_fullA_fast(p.xf, use_ctx; cache = nothing, use_cache = false, warm = false)
            t_full = time() - t0
            real_status = r_full.inner_status
            real_delta = r_full.Delta_dual
        end

        t_screened_total = t_screen_total + (isnan(t_full) ? 0.0 : t_full)
        savings_pct = rejected_at_pairwise || rejected_at_winner ?
            100.0 * (1.0 - t_screen_total / (t_screen_total + 1.86)) :   # 1.86s = W80k doc's own warm dense-value reference
            NaN

        row = (class = class_name, name = p.name, rejected_at_pairwise = rejected_at_pairwise,
               rejected_at_winner = rejected_at_winner, t_pairwise_ms = t_pairwise*1000,
               t_winnerscan_ms = t_winnerscan*1000, t_screen_total_ms = t_screen_total*1000,
               t_full_solve_s = t_full, real_inner_status = real_status, real_Delta_dual = real_delta,
               screen_stage = wres === nothing ? 0 : wres.stage)
        push!(rows, row)
        logprint("  [$class_name] ", p.name, ": rejected_pairwise=", rejected_at_pairwise,
                 "  rejected_winner=", rejected_at_winner, "  t_screen=", round(t_screen_total*1000,digits=2), "ms",
                 rejected_at_pairwise || rejected_at_winner ? "" : "  t_full_solve=$(round(t_full,digits=3))s status=$real_status Delta=$real_delta")
    end
    return rows
end

rows1 = bench_class("1_feasible", class1)
rows2 = bench_class("2_pairwise_certified_infeasible", class2)
rows3 = bench_class("3_uncaught_zerowinner", class3)
rows5 = bench_class("5_feasible_outside_budget", class5)

# class 4: search among screen-passing perturbations for a genuine KNITRO numerical failure
logprint("\n  Class 4 search: screen-passing points, probing for a genuine KNITRO -300/failure at W=80000...")
class4_found = NamedTuple[]
for (i, t) in enumerate(class4_candidates[1:min(6, end)])
    t0 = time()
    r_full, _ = evaluate_fullA_fast(t.xf, ctx; cache = nothing, use_cache = false, warm = false)
    twall = time() - t0
    logprint("    candidate $i (step=$(t.step)): inner_status=", r_full.inner_status, " Delta_dual=", r_full.Delta_dual, " wall=", round(twall,digits=2), "s")
    if !(r_full.inner_status in FEASIBLE_CODES)
        push!(class4_found, (name = "class4_realfail_$(i)", xf = t.xf, status = r_full.inner_status, wall = twall))
    end
end
logprint("  Class 4: found ", length(class4_found), " genuine screen-pass-but-KNITRO-fails example(s) among ", min(6,length(class4_candidates)), " probed at W=80000.")

write_csv_rows(joinpath(OUTDIR, "sectionE_workload_benchmark.csv"), vcat(rows1, rows2, rows3, rows5))
logprint("\nSECTION E complete. Class sizes: 1(feasible)=", length(rows1), " 2(pairwise-infeasible)=", length(rows2),
         " 3(uncaught-zerowinner)=", length(rows3), " 4(real-KNITRO-fail-found)=", length(class4_found),
         " 5(outside-budget)=", length(rows5))

# ============================================================================
# SECTION F: extreme-draw witness benchmark (Section 3 of the spec), W=80000
# ============================================================================
logprint("\n", "="^90); logprint("SECTION F: extreme-draw witness benchmark, W=80000"); logprint("="^90)

vmhwm_before_witness = vmhwm_kb()
t0 = time()
ew = build_extreme_draw_witness(ctx)
t_build = time() - t0
vmhwm_after_witness = vmhwm_kb()
mem_pairs = sizeof(ew.pairs[1,2].idx) + sizeof(ew.pairs[1,2].val)
mem_total_est = mem_pairs * D * (D-1)
logprint("build_extreme_draw_witness: wall=", round(t_build, digits=2), "s  VmHWM before=", round(vmhwm_before_witness/1e6,digits=2),
         " GB  after=", round(vmhwm_after_witness/1e6,digits=2), " GB  delta=", round((vmhwm_after_witness-vmhwm_before_witness)/1e6,digits=2), " GB")
logprint("estimated structure memory (D*(D-1) sorted pairs x (Int32 idx + Float64 val)): ", round(mem_total_est/1e9, digits=3), " GB")

Bmat = hard_score_B(ctx)
θ_full_cal = CS.reconstruct_full(xf_nat, ctx.m)
a_cal = compute_a_od(θ_full_cal, ctx)
wres_cal = screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D, full_scan = true)

n_queries = 0
query_times = Float64[]
candidate_sizes = Int[]
for d in 1:D, o in 1:D
    Pmat[o,d] > 0 || continue
    global n_queries
    t0 = time()
    exists, s, ntested, csize = query_witness(o, d, a_cal, Bmat, ew)
    tq = time() - t0
    push!(query_times, tq); push!(candidate_sizes, csize)
    n_queries += 1
    ground_truth = wres_cal.win_counts[o,d] > 0
    @assert exists == ground_truth "witness mismatch at o=$o d=$d"
end
# fraction resolved touching FEWER draws than a full O(W) scan would need
n_resolved_cheap = count(<(size(ctx.U,1)), candidate_sizes)
logprint("query_witness: N=", n_queries, " queries at calibration point")
logprint("  query time: median=", round(median(query_times)*1e6,digits=1), "us  mean=", round(mean(query_times)*1e6,digits=1),
         "us  max=", round(maximum(query_times)*1e6,digits=1), "us")
logprint("  candidate-set size (best rival's count): median=", median(candidate_sizes), "  mean=", round(mean(candidate_sizes),digits=1),
         "  (out of W=", size(ctx.U,1), ")")
logprint("  fraction with candidate-set < W (i.e. avoided a full O(W) scan): ", round(100*n_resolved_cheap/n_queries, digits=1), "%")

# break-even: how many (o,d) queries would need to be answered before the witness's one-time
# preprocessing cost pays for itself vs just running screen_hard_winners (destination-major, full)?
t_ws_ref = @elapsed screen_hard_winners(θ_full_cal, ctx, Pmat; order = 1:D)
t_per_query = median(query_times)
n_positive_pairs = count(>(0), Pmat)
cost_full_scan_per_point = t_ws_ref
cost_witness_per_point = n_positive_pairs * t_per_query   # if querying every positive-share pair
breakeven_points = t_build / max(cost_full_scan_per_point - cost_witness_per_point, 1e-12)
logprint("\nBreak-even analysis (W=80000):")
logprint("  one-time build cost: ", round(t_build, digits=2), "s")
logprint("  full destination-major winner-scan (screen_hard_winners), feasible-path cost: ", round(cost_full_scan_per_point*1000, digits=2), "ms/outer-point")
logprint("  witness-based per-outer-point cost (querying all $n_positive_pairs positive-share pairs): ",
         round(cost_witness_per_point*1000, digits=2), "ms/outer-point")
if cost_witness_per_point < cost_full_scan_per_point
    logprint("  witness IS cheaper per-point once built; break-even after ~", round(breakeven_points, digits=1), " outer-point evaluations")
else
    logprint("  witness querying ALL positive-share pairs is NOT cheaper per-point than the direct destination-major scan at D=20/W=80000")
    logprint("  (its real value is resolving a SMALL SUBSET of (o,d) pairs cheaply -- e.g. as a destination-ordering oracle -- not replacing the full scan)")
end

write_csv_rows(joinpath(OUTDIR, "sectionF_witness_benchmark.csv"),
    [(D=D, W=size(ctx.U,1), build_wall_s=t_build, mem_est_GB=mem_total_est/1e9,
      n_queries=n_queries, query_median_us=median(query_times)*1e6, query_mean_us=mean(query_times)*1e6,
      candidate_size_median=median(candidate_sizes), frac_resolved_lt_W=n_resolved_cheap/n_queries,
      full_scan_ms=cost_full_scan_per_point*1000, witness_allpairs_ms=cost_witness_per_point*1000,
      breakeven_outer_points=breakeven_points)])

logprint("\nAll sections complete. Saved to ", OUTDIR)
logprint("c9_infscreen_d20_workloads.jl DONE at ", now())
