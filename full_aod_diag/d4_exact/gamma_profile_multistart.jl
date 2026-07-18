# ============================================================================
# Continuation 6, Priority 1: investigate the NON-MONOTONIC dip-then-rise in
# the (post-tie-bug-fix) profile_Delta(g) = min_A Delta(g,A) curve.
#
# The coarse gamma_profile.jl run (results/fullA_d4/4f696b7/...111444) shows,
# using a SINGLE continuation-warm-started path:
#     g=0.9356  Delta=0.0973
#     g=0.9571  Delta=0.0021   <-- interior minimum
#     g=0.9785  Delta=0.0733   <-- RISES again
# Question: is min_A Delta(g,A) genuinely multi-basin (so the single continuation
# path is reporting whichever local optimum it lands in, not the true profile),
# or is the rise a warm-start/continuation artifact that multistart dissolves?
#
# This driver REUSES gamma_profile.jl's own `profile_delta_at_gamma` (the exact
# same per-point KNITRO A-block minimizer + validated lfix_composite gradient),
# but at each refined g runs K INDEPENDENT starts:
#   s1 continuation (prev g's multistart-best),  s2 incumbent zfree,
#   s3 calibration-base zfree,  s4/s5 incumbent+radius*randn,
#   s6/s7 calibration+radius*randn,  s8/s9 uniform-box draws.
# Then min over starts = multistart estimate of the TRUE profile at each g, and
# the terminal A_od solutions are clustered to count distinct local basins.
#
# NOTE on inner warm-state: evaluate_fullA(warm=true) shares ctx.obj.arg1 (the
# inner CC dual) across calls -- this is the production DUAL_WARM_MODE=persist
# path, kept identical to the coarse run for apples-to-apples comparison. The
# inner solve is the convex fixed-dual program, so warm state only affects inner
# speed, never which OUTER (A) basin KNITRO descends into from a given init.
# ============================================================================
ENV["GP_RUN_GRID"] = "0"   # load gamma_profile.jl's functions/ctx/pe without running its own grid
include(joinpath(@__DIR__, "gamma_profile.jl"))
using Random, Statistics

const MS_RUN_ID = "gamma_profile_multistart_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const MS_OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, MS_RUN_ID)
mkpath(MS_OUTDIR)

# ---- reference start points -------------------------------------------------
const ZFREE_INCUMBENT = [0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
# calibration-base A_od in reduced coords (same construction as check_multistart_isolate.jl)
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
const ZFREE_CALIB = pivot_reduce(log.(Aod_theta0), pe)

# A_od characterization (16-dim, the actual economic object) from reduced zfree
Aod_vec(zfree) = vec(exp.(pivot_expand(zfree, pe)))

# Delta achieved by the start of a given `kind` at a g point (NaN if that start was infeasible)
function feas_delta_of(feas, kind::String)
    for f in feas
        f.kind == kind && return f.Delta
    end
    return NaN
end

# ---- start generation (K independent starts at each g) ----------------------
function make_starts(zf_continuation::Vector{Float64}, rng::AbstractRNG)
    starts = Tuple{String,Vector{Float64}}[]
    push!(starts, ("continuation", copy(zf_continuation)))
    push!(starts, ("incumbent",    copy(ZFREE_INCUMBENT)))
    push!(starts, ("calib",        copy(ZFREE_CALIB)))
    push!(starts, ("incumbent+0.5r", ZFREE_INCUMBENT .+ 0.5 .* randn(rng, n)))
    push!(starts, ("incumbent+1.5r", ZFREE_INCUMBENT .+ 1.5 .* randn(rng, n)))
    push!(starts, ("calib+0.5r",     ZFREE_CALIB     .+ 0.5 .* randn(rng, n)))
    push!(starts, ("calib+1.5r",     ZFREE_CALIB     .+ 1.5 .* randn(rng, n)))
    push!(starts, ("uniform_pm2_a", (rand(rng, n) .* 4.0 .- 2.0)))   # kind strings MUST be comma-free (unquoted CSV field)
    push!(starts, ("uniform_pm2_b", (rand(rng, n) .* 4.0 .- 2.0)))
    return starts
end

# ---- refined grid: 0.94..0.99 every 0.005, plus the two coarse anchor points -
const MAXTIME_PER_START = parse(Float64, get(ENV, "MS_MAXTIME_PER_START", "15.0"))
g_base = collect(0.940:0.005:0.990)
g_anchors = [0.9570543833857178, 0.978527191692859]   # the coarse dip & rise points, verbatim
g_grid = sort(unique(vcat(g_base, g_anchors)))
if haskey(ENV, "MS_SMOKE_G")          # smoke test: run ONLY this single g point
    g_grid = [parse(Float64, ENV["MS_SMOKE_G"])]
end
println("Refined g grid ($(length(g_grid)) points): ", g_grid)
println("K=9 starts per point, maxtime_per_start=$(MAXTIME_PER_START)s, threads=$(Threads.nthreads())")
println("OUTDIR = ", MS_OUTDIR); flush(stdout)

per_start_rows = NamedTuple[]
summary_rows   = NamedTuple[]
zf_cont = copy(ZFREE_INCUMBENT)   # continuation seed for the first g point

for (gi, g) in enumerate(g_grid)
    rng = MersenneTwister(7000 + round(Int, g * 1e6))
    starts = make_starts(zf_cont, rng)
    println("\n", "="^78)
    @printf("g = %.7f  (%d/%d)\n", g, gi, length(g_grid)); flush(stdout)
    feas = NamedTuple[]   # feasible terminal solutions at this g
    for (si, (kind, zf0)) in enumerate(starts)
        res = profile_delta_at_gamma(g, zf0, ctx, pe; maxtime_real = MAXTIME_PER_START, hessopt_tag = "sr1")
        has_sol = res.best_zfree !== nothing && isfinite(res.best_Delta)
        aod = has_sol ? Aod_vec(res.best_zfree) : fill(NaN, D^2)
        @printf("  [s%d %-16s] status=%d n_eval=%3d wall=%5.1fs best_Delta=%.6e\n",
                si, kind, res.knitro_status, res.n_eval, res.wall, res.best_Delta); flush(stdout)
        push!(per_start_rows, (g = g, start_id = si, start_kind = kind,
              knitro_status = res.knitro_status, n_eval = res.n_eval, wall = res.wall,
              best_Delta = res.best_Delta, best_gravity = res.best_gravity, best_kkt = res.best_kkt,
              aod = aod, zfree = has_sol ? copy(res.best_zfree) : fill(NaN, n)))
        if has_sol
            push!(feas, (Delta = res.best_Delta, aod = aod, zfree = copy(res.best_zfree), kind = kind))
        end
    end

    # ---- cluster feasible terminal solutions into basins (relative L2 on A_od) ----
    clusters = NamedTuple[]   # each: (rep_aod, best_Delta, members::Vector{Int})
    for (k, f) in enumerate(feas)
        placed = false
        for c in clusters
            reld = norm(f.aod .- c.rep_aod) / max(norm(c.rep_aod), 1e-12)
            if reld < 5e-3
                push!(c.members, k)
                placed = true
                break
            end
        end
        placed || push!(clusters, (rep_aod = f.aod, best_Delta = Ref(f.Delta), members = Int[k]))
    end
    # per-cluster best Delta
    for c in clusters
        c.best_Delta[] = minimum(feas[k].Delta for k in c.members)
    end
    n_feas = length(feas)
    ms_min = n_feas == 0 ? NaN : minimum(f.Delta for f in feas)
    ms_max = n_feas == 0 ? NaN : maximum(f.Delta for f in feas)
    n_basins = length(clusters)
    # basin whose members achieve the overall min
    cont_delta = feas_delta_of(feas, "continuation")
    inc_delta  = feas_delta_of(feas, "incumbent")

    @printf("  --> feasible=%d/9  multistart_min_Delta=%.6e (max=%.6e)  n_basins=%d  cont=%.6e inc=%.6e\n",
            n_feas, ms_min, ms_max, n_basins, cont_delta, inc_delta); flush(stdout)
    for (ci, c) in enumerate(clusters)
        @printf("      basin %d: nmembers=%d best_Delta=%.6e  Delta-range=[%.3e,%.3e]\n",
            ci, length(c.members), c.best_Delta[],
            minimum(feas[k].Delta for k in c.members), maximum(feas[k].Delta for k in c.members))
    end

    push!(summary_rows, (g = g, n_feasible = n_feas, multistart_min_Delta = ms_min,
          multistart_max_Delta = ms_max, n_basins = n_basins,
          continuation_Delta = cont_delta, incumbent_Delta = inc_delta,
          min_minus_delta = ms_min - ctx.δ))

    # continuation seed for next g = THIS g's overall best zfree (if any)
    if n_feas > 0
        bidx = argmin([f.Delta for f in feas])
        global zf_cont = copy(feas[bidx].zfree)
    end
end

# ---- write outputs ----------------------------------------------------------
open(joinpath(MS_OUTDIR, "multistart_per_start.csv"), "w") do io
    println(io, "g,start_id,start_kind,knitro_status,n_eval,wall,best_Delta,best_gravity,best_kkt,",
                join(["aod$i" for i in 1:D^2], ","), ",", join(["zf$i" for i in 1:n], ","))
    for r in per_start_rows
        print(io, r.g, ",", r.start_id, ",", r.start_kind, ",", r.knitro_status, ",", r.n_eval, ",",
                  r.wall, ",", r.best_Delta, ",", r.best_gravity, ",", r.best_kkt)
        for v in r.aod;   print(io, ",", v); end
        for v in r.zfree; print(io, ",", v); end
        println(io)
    end
end
open(joinpath(MS_OUTDIR, "multistart_summary.csv"), "w") do io
    println(io, "g,n_feasible,multistart_min_Delta,multistart_max_Delta,n_basins,continuation_Delta,incumbent_Delta,min_minus_delta")
    for r in summary_rows
        println(io, r.g, ",", r.n_feasible, ",", r.multistart_min_Delta, ",", r.multistart_max_Delta, ",",
                    r.n_basins, ",", r.continuation_Delta, ",", r.incumbent_Delta, ",", r.min_minus_delta)
    end
end

println("\n", "="^78)
println("MULTISTART SUMMARY (g, min_Delta, max_Delta, n_basins, continuation_Delta):")
for r in summary_rows
    @printf("  g=%.7f  min=%.6e  max=%.6e  basins=%d  cont=%.6e  %s\n",
        r.g, r.multistart_min_Delta, r.multistart_max_Delta, r.n_basins, r.continuation_Delta,
        (isfinite(r.continuation_Delta) && r.continuation_Delta > 1.05*r.multistart_min_Delta) ? "<== continuation ABOVE multistart-min" : "")
end
println("\nWrote:")
println("  ", joinpath(MS_OUTDIR, "multistart_per_start.csv"))
println("  ", joinpath(MS_OUTDIR, "multistart_summary.csv"))
