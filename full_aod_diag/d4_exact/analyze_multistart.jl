# Post-analysis of gamma_profile_multistart.jl output. Reads multistart_per_start.csv,
# reconstructs per-g feasible-Delta distributions and A_od basin clusters, and prints the
# refined profile table + the dip/rise verdict evidence. Usage:
#   julia analyze_multistart.jl <run_dir>
using DelimitedFiles, Printf, LinearAlgebra, Statistics

run_dir = length(ARGS) >= 1 ? ARGS[1] : error("pass the multistart run dir")
raw, hdr = readdlm(joinpath(run_dir, "multistart_per_start.csv"), ',', header = true)
hdr = vec(hdr)
col(name) = findfirst(==(name), hdr)
gcol, dcol, kindcol, statuscol = col("g"), col("best_Delta"), col("start_kind"), col("knitro_status")
aod_cols = [col("aod$i") for i in 1:16]

# Robustness guard: the ORIGINAL run's per_start CSV wrote two uniform-start kinds containing a literal
# comma ("uniform[-2,2]#N"), which shifts columns for those rows under readdlm. Restrict to the
# comma-free structured kinds (all feasible starts of interest live here); a re-run with the fixed
# source names (uniform_pm2_a/b) is fully clean and this filter is then a no-op superset.
const CLEAN_KINDS = ["continuation","incumbent","calib","incumbent+0.5r","incumbent+1.5r",
                     "calib+0.5r","calib+1.5r","uniform_pm2_a","uniform_pm2_b"]
keep = [r for r in 1:size(raw,1) if string(raw[r, kindcol]) in CLEAN_KINDS]
raw = raw[keep, :]

gs = Float64.(raw[:, gcol])
deltas = [x isa Number ? Float64(x) : (x == "NaN" ? NaN : parse(Float64, string(x))) for x in raw[:, dcol]]
kinds = string.(raw[:, kindcol])
aods = [ [ (raw[r, c] isa Number ? Float64(raw[r,c]) : NaN) for c in aod_cols ] for r in 1:size(raw,1) ]

ug = sort(unique(gs))
CLUSTER_REL = 5e-3

println("="^100)
@printf("%-11s %5s %12s %12s %7s  %-45s\n", "g", "nfeas", "ms_min_Δ", "cont_Δ", "basins", "sorted feasible Δ (per start)")
println("="^100)
summary = NamedTuple[]
for g in ug
    idx = findall(i -> gs[i] == g, 1:length(gs))
    fidx = [i for i in idx if isfinite(deltas[i])]
    fdel = deltas[fidx]
    order = sortperm(fdel)
    fidx_s = fidx[order]; fdel_s = fdel[order]
    # cluster feasible A_od solutions
    clusters = Vector{Int}[]  # each: list of row indices
    reps = Vector{Float64}[]
    for i in fidx_s
        placed = false
        for (ci, rep) in enumerate(reps)
            if norm(aods[i] .- rep) / max(norm(rep), 1e-12) < CLUSTER_REL
                push!(clusters[ci], i); placed = true; break
            end
        end
        placed || (push!(clusters, [i]); push!(reps, aods[i]))
    end
    cont_i = findfirst(i -> kinds[i] == "continuation" && isfinite(deltas[i]), idx)
    cont_d = cont_i === nothing ? NaN : deltas[idx[findfirst(i->kinds[i]=="continuation", idx)]]
    # continuation delta: look it up directly
    controw = findfirst(i -> kinds[i]=="continuation", idx)
    cont_d = controw === nothing ? NaN : deltas[idx[controw]]
    ms_min = isempty(fdel) ? NaN : minimum(fdel)
    dstr = join([@sprintf("%.4e", d) for d in fdel_s], " ")
    @printf("%-11.7f %5d %12.5e %12.5e %7d  %s\n", g, length(fidx), ms_min, cont_d, length(clusters), dstr)
    push!(summary, (g=g, nfeas=length(fidx), ms_min=ms_min, cont_d=cont_d, nbasin=length(clusters),
                    basin_best=[minimum(deltas[c]) for c in clusters]))
end

println("\n", "="^100)
println("DIP vs RISE evidence (interior region):")
println("="^100)
for g in ug
    (0.955 <= g <= 0.985) || continue
    s = summary[findfirst(x->x.g==g, summary)]
    bb = sort(s.basin_best)
    @printf("  g=%.7f  ms_min_Δ=%.5e   basin bests=[%s]\n", g, s.ms_min,
            join([@sprintf("%.4e",b) for b in bb], ", "))
end

# does the multistart-min curve itself still rise after its interior minimum?
finite = [s for s in summary if isfinite(s.ms_min)]
gmin_i = argmin([s.ms_min for s in finite])
gmin = finite[gmin_i]
println("\nInterior minimum of the MULTISTART-MIN curve: g=", gmin.g, "  ms_min_Δ=", gmin.ms_min)
rises_after = [s for s in finite if s.g > gmin.g && s.ms_min > 1.5*gmin.ms_min]
println("Points ABOVE 1.5x the interior min at larger g (multistart-confirmed rise): ",
        [(s.g, round(s.ms_min, sigdigits=4)) for s in rises_after])
println("\nContinuation-vs-multistart gaps > 5% (single-path landed in a worse basin):")
for s in finite
    if isfinite(s.cont_d) && s.cont_d > 1.05*s.ms_min
        @printf("  g=%.7f  cont_Δ=%.5e  ms_min_Δ=%.5e  ratio=%.2f\n", s.g, s.cont_d, s.ms_min, s.cont_d/s.ms_min)
    end
end
