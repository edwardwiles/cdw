# ============================================================================
# Systematic trade-share verification (2026-07-16) of EVERY saved GC and GU
# result, following the discovery that GC's T2/warm point (kappa=0.1055)
# satisfies gravity (aggregate, demeaned -- can mask localized error) and the
# divergence budget (depends only on p, not on destination shares at all) but
# FAILS actual trade-share matching by 6.2% and the CC first-order moment by
# a large margin. Verified this is a real problem, not a check bug (the same
# methodology passes cleanly, to ~1e-6/1e-14, on a known-good LC point).
#
# This checks whether that failure is an isolated outlier (plausibly tied to
# GC's T2/warm's unusually large relΔA=6.96) or a broader pattern across
# GC/GU's other results, which tend toward much larger A_od movement than the
# local (LC/LU) methods ever explore.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra, DelimitedFiles

function verify_point(θsol::Vector{Float64}, budget::Float64, saved_kappa, saved_relΔA)
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
    if !ok
        return (ok=false, R_mean=NaN, div_p=NaN, worst_share_err=NaN, Gmom_err=NaN, within_budget=false)
    end
    gr = gravity_residual(umat, logτ, logw, σ)
    div_p = divergence_of(p)
    log_x = build_log_x(Uσ, θsol[1])
    logp = log.(p)
    worst_err = 0.0
    for dd in 1:D
        ρ_dd = dd == focal ? 0.0 : ρ
        model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=ρ_dd)
        err = maximum(abs.(model_shares .- λData[:, dd]))
        worst_err = max(worst_err, err)
    end
    Kchk = zeros(W); Gchk = zeros(W, D + 1)
    EK_moments_focal_norm_directgp!(Kchk, Gchk, θsol, U, (γ=γ,))
    Gmom_err = maximum(abs.(sum(p .* Gchk[:, 1:D], dims=1)))
    (ok=true, R_mean=gr.R_mean, div_p=div_p, worst_share_err=worst_err, Gmom_err=Gmom_err,
     within_budget = div_p <= budget * (1 + 1e-6) + 1e-10)
end

TARGETS_BUDGET = Dict("T1" => 0.1, "T2" => 1.0, "T3" => 2.0)
STARTS = Dict("T1" => ["Astar", "rand1", "rand2"], "T2" => ["Astar", "rand1", "warm"], "T3" => ["Astar", "rand1", "warm"])

rows = NamedTuple[]
for method in ["gc", "gu"]
    for tname in ["T1", "T2", "T3"]
        for sname in STARTS[tname]
            path = joinpath(@__DIR__, "out_$method", "$(method)_$(tname)_$(sname).jld2")
            isfile(path) || continue
            d = JLD2.load(path)
            get(d, "done", false) == true || continue
            if method == "gc"
                θsol = Float64.(d["theta_star"])
                saved_kappa = d["kappa"]; saved_feasible = d["feasible"]
            else
                θsol = vcat(θr0[1:2], d["gammap_target"], Float64.(d["Acol_best"]))
                saved_kappa = gp2kappa(d["gammap_target"]); saved_feasible = d["delta_star_best"] < 1e3
            end
            saved_relΔA = d["relDeltaA"]
            budget = TARGETS_BUDGET[tname]
            @printf("\n[%s %s/%s] saved: kappa=%.6f feasible=%s relDeltaA=%.3f -- verifying...\n",
                    uppercase(method), tname, sname, saved_kappa, saved_feasible, saved_relΔA)
            flush(stdout)
            t0 = time()
            v = verify_point(θsol, budget, saved_kappa, saved_relΔA)
            wall = time() - t0
            status = !v.ok ? "RESOLVE_FAILED" :
                     (v.worst_share_err < 1e-4 && v.within_budget) ? "GENUINE" : "SPURIOUS"
            @printf("  -> ok=%s R_mean=%.3e div_p=%.4f(budget=%.2f) worst_share_err=%.3e Gmom_err=%.3e  [%s]  wall=%.1fs\n",
                    v.ok, v.R_mean, v.div_p, budget, v.worst_share_err, v.Gmom_err, status, wall)
            flush(stdout)
            push!(rows, (method=uppercase(method), target=tname, start=sname, saved_kappa=saved_kappa,
                         saved_feasible=saved_feasible, relΔA=saved_relΔA, ok=v.ok, R_mean=v.R_mean,
                         div_p=v.div_p, budget=budget, worst_share_err=v.worst_share_err,
                         Gmom_err=v.Gmom_err, status=status))
        end
    end
end

println("\n" * "="^100); println(">>> VERIFY_ALL_GC_GU SUMMARY"); println("="^100)
@printf("%-6s %-4s %-6s %10s %8s %10s %14s %14s %10s\n",
        "method","tgt","start","kappa","relΔA","div_p","worst_shr_err","Gmom_err","status")
for r in rows
    @printf("%-6s %-4s %-6s %10.6f %8.3f %10.4f %14.3e %14.3e %10s\n",
            r.method, r.target, r.start, r.saved_kappa, r.relΔA, r.div_p, r.worst_share_err, r.Gmom_err, r.status)
end

n_spurious = count(r -> r.status == "SPURIOUS", rows)
n_genuine = count(r -> r.status == "GENUINE", rows)
n_failed = count(r -> r.status == "RESOLVE_FAILED", rows)
@printf("\nTOTAL: %d genuine, %d SPURIOUS (looked feasible, isn't), %d failed to re-solve, out of %d checked\n",
        n_genuine, n_spurious, n_failed, length(rows))

open(joinpath(@__DIR__, "verify_all_gc_gu_results.csv"), "w") do io
    writedlm(io, ["method" "target" "start" "saved_kappa" "relDeltaA" "div_p" "budget" "worst_share_err" "Gmom_err" "status"], ',')
    for r in rows
        writedlm(io, [[r.method r.target r.start r.saved_kappa r.relΔA r.div_p r.budget r.worst_share_err r.Gmom_err r.status]], ',')
    end
end
println("\nVERIFY_ALL_GC_GU DONE")
