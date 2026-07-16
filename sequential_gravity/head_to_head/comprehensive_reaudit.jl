# ============================================================================
# Comprehensive re-audit (2026-07-16, post recover_lfd fix): re-check EVERY
# saved result from the official 4-method comparison (LC/LU/GC/GU, up to 9
# each) with the FIXED seq_gravcol (which now correctly rejects points whose
# inner CC dual solve failed with a rejected nStatus, e.g. -300=UNBOUNDED) and
# a direct trade-share check. Simplified per the user's guidance: verify
# gravity DIRECTLY on the fixed A_od (gravity_residual, no linearized-moment
# reconstruction needed).
#
# 3 of these (LC T2/rand1, LC T3/warm, GC T2/warm) are ALREADY CONFIRMED
# spurious via the fixed recover_lfd -- this checks whether the bug corrupted
# any of the other ~31 saved results too.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra, DelimitedFiles

function check_point(θsol::Vector{Float64}, budget::Float64)
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=budget, maxit=100, tol=5e-4)
    if !ok
        return (ok=false, worst_share_err=NaN, status="REJECTED_BY_FIXED_RECOVER_LFD")
    end
    log_x = build_log_x(Uσ, θsol[1])
    logp = log.(p)
    worst_err = 0.0
    for dd in 1:D
        ρ_dd = dd == focal ? 0.0 : ρ
        model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=ρ_dd)
        err = maximum(abs.(model_shares .- λData[:, dd]))
        worst_err = max(worst_err, err)
    end
    status = worst_err < 1e-4 ? "GENUINE" : "STILL_SPURIOUS_SOMEHOW"
    (ok=true, worst_share_err=worst_err, status=status)
end

TARGETS_BUDGET = Dict("T1" => 0.1, "T2" => 1.0, "T3" => 2.0)
STARTS = Dict("T1" => ["Astar", "rand1", "rand2"], "T2" => ["Astar", "rand1", "warm"], "T3" => ["Astar", "rand1", "warm"])
ALREADY_KNOWN_BAD = Set([("LC","T2","rand1"), ("LC","T3","warm"), ("GC","T2","warm")])

rows = NamedTuple[]
for method in ["lc", "lu", "gc", "gu"]
    for tname in ["T1", "T2", "T3"]
        for sname in STARTS[tname]
            METHOD = uppercase(method)
            if (METHOD, tname, sname) in ALREADY_KNOWN_BAD
                @printf("[%s %s/%s] already confirmed spurious -- skipping re-check\n", METHOD, tname, sname)
                push!(rows, (method=METHOD, target=tname, start=sname, status="ALREADY_CONFIRMED_SPURIOUS", worst_share_err=NaN))
                continue
            end
            path = joinpath(@__DIR__, "out_$method", "$(method)_$(tname)_$(sname).jld2")
            isfile(path) || continue
            d = JLD2.load(path)
            get(d, "done", false) == true || continue
            if method == "lc"
                θsol = d["best_feasible_theta"]
                θsol === nothing && continue
                θsol = Float64.(θsol)
            elseif method == "gc"
                θsol = Float64.(d["theta_star"])
            else  # lu, gu
                Aod = method == "lu" ? d["best_feasible_Aod"] : d["Acol_best"]
                gt = d["gammap_target"]
                (Aod === nothing || !isfinite(gt)) && continue
                θsol = vcat(θr0[1:2], gt, Float64.(Aod))
            end
            # LU/GU are UNCONSTRAINED (fix gp, minimize delta* -- no budget to violate); only
            # LC/GC have a genuine divergence-BUDGET feasibility criterion. Using the nominal
            # target budget as delta for LU/GU here would wrongly reject their own legitimate
            # achieved delta* (e.g. LU's real T1 answer, 0.1147, exceeds T1's nominal 0.1
            # "budget" by construction -- that's not a bug, LU was never subject to that budget).
            budget = (method == "lu" || method == "gu") ? Inf : TARGETS_BUDGET[tname]
            @printf("[%s %s/%s] checking...\n", METHOD, tname, sname); flush(stdout)
            t0 = time()
            v = check_point(θsol, budget)
            wall = time() - t0
            @printf("  -> ok=%s worst_share_err=%s  [%s]  wall=%.1fs\n",
                    v.ok, v.ok ? @sprintf("%.3e", v.worst_share_err) : "n/a", v.status, wall)
            flush(stdout)
            push!(rows, (method=METHOD, target=tname, start=sname, status=v.status, worst_share_err=v.worst_share_err))
        end
    end
end

println("\n" * "="^90); println(">>> COMPREHENSIVE RE-AUDIT SUMMARY"); println("="^90)
@printf("%-6s %-4s %-6s %-28s %14s\n", "method", "tgt", "start", "status", "worst_shr_err")
for r in rows
    @printf("%-6s %-4s %-6s %-28s %14s\n", r.method, r.target, r.start, r.status,
            isnan(r.worst_share_err) ? "n/a" : @sprintf("%.3e", r.worst_share_err))
end
n_bad = count(r -> r.status in ("REJECTED_BY_FIXED_RECOVER_LFD", "ALREADY_CONFIRMED_SPURIOUS", "STILL_SPURIOUS_SOMEHOW"), rows)
n_good = count(r -> r.status == "GENUINE", rows)
@printf("\nTOTAL: %d GENUINE, %d SPURIOUS/rejected, out of %d checked\n", n_good, n_bad, length(rows))

open(joinpath(@__DIR__, "comprehensive_reaudit_results.csv"), "w") do io
    writedlm(io, ["method" "target" "start" "status" "worst_share_err"], ',')
    for r in rows
        writedlm(io, [[r.method r.target r.start r.status r.worst_share_err]], ',')
    end
end
println("\nCOMPREHENSIVE_REAUDIT DONE")
