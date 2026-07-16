# ============================================================================
# Spot-check (2026-07-16, post recover_lfd fix): re-verify the handful of
# lu_multistart50 points anyone would actually cite -- each target's BEST
# (lowest achieved delta*) point, plus T3's second (only other) feasible
# point -- using the same direct-gravity-residual check as
# comprehensive_reaudit.jl. Not a full 150-point re-audit (lower priority
# per HANDOFF_2026-07-16_recover_lfd_bug.md); this covers the specific
# numbers already reported to the user.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf

function check_point(θsol::Vector{Float64})
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
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

TO_CHECK = [("T1", "pt07"), ("T2", "pt20"), ("T3", "pt40"), ("T3", "pt45")]

rows = NamedTuple[]
for (tname, pname) in TO_CHECK
    path = joinpath(@__DIR__, "out_lu_multistart50", "lu_ms_$(tname)_$(pname).jld2")
    @assert isfile(path) "missing $path"
    d = JLD2.load(path)
    Aod = d["best_feasible_Aod"]
    gt = d["gammap_target"]
    @assert Aod !== nothing && isfinite(gt)
    θsol = vcat(θr0[1:2], gt, Float64.(Aod))
    @printf("[LU-MS %s/%s] original best_feasible_delta_star=%.6g -- checking...\n", tname, pname, d["best_feasible_delta_star"])
    flush(stdout)
    t0 = time()
    v = check_point(θsol)
    wall = time() - t0
    @printf("  -> ok=%s worst_share_err=%s  [%s]  wall=%.1fs\n",
            v.ok, v.ok ? @sprintf("%.3e", v.worst_share_err) : "n/a", v.status, wall)
    flush(stdout)
    push!(rows, (target=tname, point=pname, orig_delta_star=d["best_feasible_delta_star"], status=v.status, worst_share_err=v.worst_share_err))
end

println("\n" * "="^90); println(">>> LU_MULTISTART50 SPOTCHECK SUMMARY"); println("="^90)
for r in rows
    @printf("%-4s %-6s orig_delta*=%-10.6g status=%-28s worst_shr_err=%s\n", r.target, r.point, r.orig_delta_star, r.status,
            isnan(r.worst_share_err) ? "n/a" : @sprintf("%.3e", r.worst_share_err))
end
println("\nSPOTCHECK_LU_MULTISTART50 DONE")
