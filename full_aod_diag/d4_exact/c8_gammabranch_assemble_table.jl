# ============================================================================
# Continuation 8, workstream C: assemble the machine-readable deliverable
# table from the low-g bracket run and the high-g sweep run's CSV outputs.
# One row per DISTINCT g tested (across BOTH branches), reporting best/
# second-best multistart Delta, which start produced the best, exact (inner
# CC dual) feasibility, A-solution stored in a companion JLD2 keyed by row
# index, runtime, and a local-stationarity diagnostic (reusing
# external_stationarity_check_c8, itself a verbatim reproduction of
# stationarity_check.jl's formula -- see c8_gammabranch_core.jl header) at a
# small set of headline points.
#
# NOTE: deliberately avoids the CSV.jl/DataFrames.jl packages (not in this
# worktree's Project.toml, and this investigation's convention -- see every
# OTHER script in this directory -- is plain comma-split parsing of its own
# simple, comma-free-field CSVs rather than adding new package dependencies
# for a one-off assembly step).
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using JLD2, Printf

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const LOWG_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", "d6e3b05", "c8_gammabranch_lowg_bracket")
const HIGHG_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", "f6790e0", "c8_gammabranch_highg_sweep")
const HIGHG_REFINE_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", "128f260", "c8_gammabranch_highg_refine")
const OUT_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8)
mkpath(OUT_DIR)

"Minimal comma-split CSV reader for THIS investigation's own simple, comma-free-field CSVs (header + rows of plain values, no quoting needed except a few LP-check labels handled separately)."
function read_simple_csv(path::String)
    lines = readlines(path)
    isempty(lines) && return (header = String[], rows = Vector{String}[])
    header = split(lines[1], ",")
    rows = [split(l, ",") for l in lines[2:end] if !isempty(l)]
    return (header = String.(header), rows = rows)
end
col_idx(header, name) = findfirst(==(name), header)

"Group a per-start CSV (g,start_kind,Delta,feasible,knitro_status,n_eval,wall[,...]) by g -> (best,second_best,best_kind,n_feasible,n_total)."
function summarize_multistart_csv(path::String)
    isfile(path) || return Dict{Float64,NamedTuple}()
    h, rows = read_simple_csv(path)
    ig, ik, iD, ifz = col_idx(h,"g"), col_idx(h,"start_kind"), col_idx(h,"Delta"), col_idx(h,"feasible")
    by_g = Dict{Float64,Vector{Tuple{String,Float64,Bool}}}()
    for r in rows
        g = parse(Float64, r[ig]); kind = r[ik]; feasible = r[ifz] == "true"
        Δ = feasible ? parse(Float64, r[iD]) : NaN
        push!(get!(by_g, g, Tuple{String,Float64,Bool}[]), (kind, Δ, feasible))
    end
    out = Dict{Float64,NamedTuple}()
    for (g, entries) in by_g
        feas = filter(e -> e[3], entries)
        if isempty(feas)
            out[g] = (best = NaN, second_best = NaN, best_kind = "NONE", n_feasible = 0, n_total = length(entries))
        else
            ord = sortperm([e[2] for e in feas])
            best_kind, best = feas[ord[1]][1], feas[ord[1]][2]
            second = length(ord) >= 2 ? feas[ord[2]][2] : best
            out[g] = (best = best, second_best = second, best_kind = best_kind, n_feasible = length(feas), n_total = length(entries))
        end
    end
    return out
end

lowg_ms = summarize_multistart_csv(joinpath(LOWG_DIR, "multistart_at_root.csv"))
highg_ms = summarize_multistart_csv(joinpath(HIGHG_DIR, "highg_multistart.csv"))

# ---- build the row set: every g with a multistart summary (both branches), tagged by branch ----
rows = NamedTuple[]
for (g, s) in lowg_ms
    push!(rows, (branch = "low_g", g = g, best_Delta = s.best, second_best_Delta = s.second_best,
                 best_start_kind = s.best_kind, n_feasible = s.n_feasible, n_total_starts = s.n_total))
end
for (g, s) in highg_ms
    push!(rows, (branch = "high_g", g = g, best_Delta = s.best, second_best_Delta = s.second_best,
                 best_start_kind = s.best_kind, n_feasible = s.n_feasible, n_total_starts = s.n_total))
end
sort!(rows, by = r -> r.g)

# ---- also fold in the low-g Stage-1 bisection-only g's (single-continuation-start, not full
# multistart) so the table covers the FULL set of g's actually evaluated, not just the multistart
# subset -- tagged with n_total_starts=1 so the "which of 9" columns read honestly. ----
if isfile(joinpath(LOWG_DIR, "bisection_trace.csv"))
    h, brows = read_simple_csv(joinpath(LOWG_DIR, "bisection_trace.csv"))
    ig, ifv = col_idx(h,"g"), col_idx(h,"Delta_minus_delta")
    existing_g = Set(r.g for r in rows)
    for r in brows
        g = parse(Float64, r[ig])
        (g in existing_g) && continue
        Δ = parse(Float64, r[ifv]) + ctx.δ
        push!(rows, (branch = "low_g", g = g, best_Delta = Δ, second_best_Delta = NaN,
                      best_start_kind = "continuation_only", n_feasible = 1, n_total_starts = 1))
        push!(existing_g, g)
    end
end
if isfile(joinpath(HIGHG_DIR, "highg_sweep_rows.csv"))
    h, hrows = read_simple_csv(joinpath(HIGHG_DIR, "highg_sweep_rows.csv"))
    ig, iD, ifz, isrc = col_idx(h,"g"), col_idx(h,"Delta"), col_idx(h,"feasible"), col_idx(h,"source")
    existing_g = Set(r.g for r in rows)
    for r in hrows
        g = parse(Float64, r[ig])
        (g in existing_g) && continue
        feasible = r[ifz] == "true"
        Δ = feasible ? parse(Float64, r[iD]) : NaN
        push!(rows, (branch = "high_g", g = g, best_Delta = Δ, second_best_Delta = NaN,
                      best_start_kind = r[isrc], n_feasible = feasible ? 1 : 0, n_total_starts = 1))
        push!(existing_g, g)
    end
end
for fname in ("fine_robust_profile.csv", "highg_bisection_trace.csv")
    fpath = joinpath(HIGHG_REFINE_DIR, fname)
    isfile(fpath) || continue
    h, frows = read_simple_csv(fpath)
    ig, iD, ik = col_idx(h,"g"), col_idx(h,"Delta"), col_idx(h,"best_kind")
    existing_g = Set(r.g for r in rows)
    for r in frows
        g = parse(Float64, r[ig])
        (g in existing_g) && continue
        Δstr = r[iD]
        Δ = (Δstr == "NaN") ? NaN : parse(Float64, Δstr)
        push!(rows, (branch = "high_g", g = g, best_Delta = Δ, second_best_Delta = NaN,
                      best_start_kind = "best_of_3:" * r[ik], n_feasible = isfinite(Δ) ? 1 : 0, n_total_starts = 3))
        push!(existing_g, g)
    end
end
sort!(rows, by = r -> r.g)

# ---- recompute each row's A-solution + exact feasibility + runtime by RE-EVALUATING at (g, warm-
# start) via a single profile_delta_at_gamma_c8 call. IMPORTANT: within each branch, rows are
# processed in ASCENDING g order with the warm-start CARRIED FORWARD from the previous row's own
# converged A (true continuation) -- NOT re-started fresh from a fixed anchor every time. This matters
# specifically for the high-g branch, where c8_gammabranch_highg_refine.jl established that a fixed
# generic start (e.g. the lower incumbent's own A) reliably lands in a WORSE basin than a properly
# continued one -- re-deriving A-solutions from a fixed anchor here would silently under-report the
# true profile this deliverable is supposed to document. ----
final_rows = NamedTuple[]
Aod_jld2 = Dict{String,Any}()
zfree_low_g_root = nothing
zfree_lower_incumbent_confirmed = nothing
zf_chain = Dict("low_g" => copy(ZFREE_INCUMBENT_C8), "high_g" => copy(ZFREE_LOWER_INCUMBENT))
for (i, r) in enumerate(rows)
    zf0 = zf_chain[r.branch]
    res = profile_delta_at_gamma_c8(r.g, zf0, ctx, pe; moment_repr = :compressed, maxtime_real = 20.0, hessopt_tag = "sr1")
    has_sol = res.best_zfree !== nothing && isfinite(res.best_Delta)
    has_sol && (zf_chain[r.branch] = copy(res.best_zfree))   # carry forward within this branch's sorted-g chain
    key = "row$(i)_g$(round(r.g, digits=8))"
    Aod_jld2[key] = has_sol ? Aod_vec_c8(res.best_zfree) : fill(NaN, D^2)
    if has_sol && r.branch == "low_g" && abs(r.g - 0.892635789502) < 1e-6
        global zfree_low_g_root = copy(res.best_zfree)
    end
    if has_sol && r.branch == "high_g" && abs(r.g - G_LOWER_INCUMBENT) < 1e-9
        global zfree_lower_incumbent_confirmed = copy(res.best_zfree)
    end
    push!(final_rows, (row = i, branch = r.branch, g = r.g,
        best_Delta_multistart = r.best_Delta, second_best_Delta_multistart = r.second_best_Delta,
        best_start_kind = r.best_start_kind, n_feasible_of_9 = r.n_feasible, n_total_starts = r.n_total_starts,
        exact_feasible = has_sol, Delta_minus_delta = has_sol ? res.best_Delta - ctx.δ : NaN,
        knitro_status = res.knitro_status, runtime_wall = res.wall, a_solution_key = key))
end

open(joinpath(OUT_DIR, "c8_gammabranch_profile_table.csv"), "w") do io
    println(io, "row,branch,g,best_Delta_multistart,second_best_Delta_multistart,best_start_kind,n_feasible_of_9,n_total_starts,exact_feasible,Delta_minus_delta,knitro_status,runtime_wall_s,a_solution_key")
    for r in final_rows
        println(io, r.row, ",", r.branch, ",", r.g, ",", r.best_Delta_multistart, ",", r.second_best_Delta_multistart, ",",
                    r.best_start_kind, ",", r.n_feasible_of_9, ",", r.n_total_starts, ",", r.exact_feasible, ",",
                    r.Delta_minus_delta, ",", r.knitro_status, ",", r.runtime_wall, ",", r.a_solution_key)
    end
end
jldsave(joinpath(OUT_DIR, "c8_gammabranch_a_solutions.jld2"); Aod_jld2 = Aod_jld2)
println("Wrote ", joinpath(OUT_DIR, "c8_gammabranch_profile_table.csv"), " (", length(final_rows), " rows)")
println("Wrote ", joinpath(OUT_DIR, "c8_gammabranch_a_solutions.jld2"), " (", length(Aod_jld2), " A-solutions)")

# ---- local-stationarity diagnostic at headline points ----
println("\n", "="^78)
println("Local-stationarity diagnostic (external_stationarity_check_c8) at headline points")
println("="^78)
headline = Tuple{String,Vector{Float64}}[]
if zfree_low_g_root !== nothing
    push!(headline, ("low_g_root", vcat(0.892635789502, zfree_low_g_root)))
end
push!(headline, ("high_g_lower_incumbent_registry_point", W_LOWER_INCUMBENT))
if zfree_lower_incumbent_confirmed !== nothing
    push!(headline, ("high_g_lower_incumbent_reconfirmed", vcat(G_LOWER_INCUMBENT, zfree_lower_incumbent_confirmed)))
end
stat_rows = NamedTuple[]
for (label, w) in headline
    sc = external_stationarity_check_c8(w, ctx, pe; find_smallest = true,
            w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, n)), w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, n)))
    @printf("  [%-40s] Delta=%.10f  Delta-delta=%+.4e  eta=%.6f  eta_nonneg=%s  residual_rel=%.4e  n_active_bounds=%d  n_nonfinite_probes=%d\n",
        label, sc.Delta, sc.Delta_minus_delta, sc.eta, sc.eta_nonneg, sc.residual_relative, sc.n_active_bounds, sc.n_nonfinite_probes)
    push!(stat_rows, (label = label, Delta = sc.Delta, Delta_minus_delta = sc.Delta_minus_delta, eta = sc.eta,
                       eta_nonneg = sc.eta_nonneg, residual_relative = sc.residual_relative,
                       n_active_bounds = sc.n_active_bounds, n_nonfinite_probes = sc.n_nonfinite_probes))
end
open(joinpath(OUT_DIR, "c8_gammabranch_stationarity.csv"), "w") do io
    println(io, "label,Delta,Delta_minus_delta,eta,eta_nonneg,residual_relative,n_active_bounds,n_nonfinite_probes")
    for r in stat_rows
        println(io, r.label, ",", r.Delta, ",", r.Delta_minus_delta, ",", r.eta, ",", r.eta_nonneg, ",", r.residual_relative, ",", r.n_active_bounds, ",", r.n_nonfinite_probes)
    end
end
println("Wrote ", joinpath(OUT_DIR, "c8_gammabranch_stationarity.csv"))
