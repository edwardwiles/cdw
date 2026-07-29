# q-bandwidth convergence campaign (2026-07-29), Phase 10: limited real-D20 confirmation.
# NOT an outer campaign -- single verified real-D20 base point (delta~0.5, seed=1, from
# Phase 3), W in {80000, 320000, 1280000}, the 2 D4-shortlisted policies, 5 representative
# individual q coordinates, 4 dense q directions. 20 Julia threads / BLAS=1 per src/melitz/CLAUDE.md.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random

const OUTDIR = joinpath(REPO, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
base_theta_q = theta_q_rows[("realD20_seed1_W80000", 0.5)]

function load_realD20_calib()
    real_dir = joinpath(REPO, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()

results_coord = NamedTuple[]
results_dir = NamedTuple[]

for W in (80_000, 320_000, 1_280_000)
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1,
        inner_loop_opt=joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    ctx = obj.γ
    sorted_ctx = ctx.sorted_tail_ctx
    D = ctx.D
    nA = D^2 - 1
    theta0 = copy(base_theta_q)
    nq = length(theta0) - 1 - nA

    obj.use_cached_x = false; obj.x .= NaN
    t0 = time()
    lfd0 = melitz_recover_lfd(obj, theta0)
    solve_t = time() - t0
    if !lfd0.lfd_ok
        @printf("[SKIP W=%d] base point failed to verify\n", W); continue
    end
    x0 = copy(lfd0.dual_x)
    @printf("W=%d Delta0=%.6e nStatus=%d base_solve=%.2fs\n", W, lfd0.Delta, lfd0.nStatus, solve_t)
    flush(stdout)

    qpiv = build_q_gravity_pivot(ctx)
    leverage = abs.(qpiv.c[qpiv.other] ./ qpiv.c[qpiv.pivot])
    perm = sortperm(leverage)
    m_lowest = perm[1]; m_highest = perm[end]

    # focal-origin export cell: an (j, d) cell for d != j among the free coordinates.
    f_free_lin = ctx.f_free_lin
    j = ctx.target_country
    m_focal_origin = findfirst(k -> lin2od(f_free_lin[qpiv.other[k]], D)[1] == j &&
                                     lin2od(f_free_lin[qpiv.other[k]], D)[2] != j, 1:nq)
    m_focal_dest = findfirst(k -> lin2od(f_free_lin[qpiv.other[k]], D)[2] == j &&
                                    lin2od(f_free_lin[qpiv.other[k]], D)[1] != j, 1:nq)
    m_ordinary = perm[nq ÷ 2]

    coord_labels = [("lowest_leverage", m_lowest), ("highest_leverage", m_highest),
                    ("ordinary", m_ordinary),
                    ("focal_origin_export", m_focal_origin === nothing ? m_ordinary : m_focal_origin),
                    ("focal_dest_import", m_focal_dest === nothing ? m_ordinary : m_focal_dest)]

    policies = [("B_alpha_half", nothing), ("C_target25", FixedCrossingQBandwidth(25))]
    for (clabel, m) in coord_labels
        for (plabel, pol_maybe) in policies
            pol = pol_maybe === nothing ? PowerScaledQBandwidth(
                (_melitz_bisect_h_two_sided(25, theta0, m, ctx, sorted_ctx))[1], W, 0.5) : pol_maybe
            r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
            push!(results_coord, (W=W, coord_label=clabel, m=m, policy=plabel, h=r.h,
                                   cplus=r.crossings_plus_total, cminus=r.crossings_minus_total,
                                   secant=r.secant, leverage=leverage[m]))
            @printf("  [%s / %s] m=%d h=%.3e crossings=(+%d,-%d) secant=%.6e\n",
                    clabel, plabel, m, r.h, r.crossings_plus_total, r.crossings_minus_total, r.secant)
            flush(stdout)
        end
    end

    # dense directions
    rng = MersenneTwister(20260729)
    dirs = Dict{String,Vector{Float64}}()
    dirs["dense_random_1"] = normalize(randn(rng, nq))
    dirs["dense_random_2"] = normalize(randn(rng, nq))
    half = nq ÷ 2
    ob = zeros(nq); ob[1:half] .= randn(rng, half); dirs["origin_block"] = normalize(ob)
    dirs["leverage_weighted"] = normalize(leverage .* randn(rng, nq))

    for (dname, d) in dirs
        h_ref, _, _ = _melitz_bisect_h_two_sided(25, theta0, argmax(abs.(d)), ctx, sorted_ctx)
        t = h_ref * sqrt(nq)   # crude amplitude scaling for a dense direction (disclosed, not LP-derived)
        theta_p = copy(theta0); theta_p[1+nA+1:end] .+= t .* d
        theta_m = copy(theta0); theta_m[1+nA+1:end] .-= t .* d
        melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Bp = -obj(x0)
        melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Bm = -obj(x0)
        melitz_update_operator_at_theta!(obj.op, theta0, ctx)
        secant_fd = (Bp - Bm) / (2t)

        obj.use_cached_x = false; obj.x .= NaN; lp = melitz_recover_lfd(obj, theta_p)
        obj.use_cached_x = false; obj.x .= NaN; lm = melitz_recover_lfd(obj, theta_m)
        secant_reopt = (lp.lfd_ok && lm.lfd_ok) ? (lp.Delta - lm.Delta) / (2t) : NaN

        @printf("  [dense-dir %s] t=%.3e secant_fd=%.6e secant_reopt=%.6e (ok_p=%s ok_m=%s)\n",
                dname, t, secant_fd, secant_reopt, lp.lfd_ok, lm.lfd_ok)
        flush(stdout)
        push!(results_dir, (W=W, direction=dname, t=t, secant_fd=secant_fd, secant_reopt=secant_reopt,
                             lfd_ok_p=lp.lfd_ok, lfd_ok_m=lm.lfd_ok))
    end
end

open(joinpath(OUTDIR, "melitz_qbw_phase10_realD20_coord_2026-07-29.csv"), "w") do io
    println(io, "W,coord_label,m,policy,h,cplus,cminus,secant,leverage")
    for r in results_coord
        println(io, join([r.W, r.coord_label, r.m, r.policy, r.h, r.cplus, r.cminus, r.secant, r.leverage], ","))
    end
end
open(joinpath(OUTDIR, "melitz_qbw_phase10_realD20_directions_2026-07-29.csv"), "w") do io
    println(io, "W,direction,t,secant_fd,secant_reopt,lfd_ok_p,lfd_ok_m")
    for r in results_dir
        println(io, join([r.W, r.direction, r.t, r.secant_fd, r.secant_reopt, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end
println("\nPhase 10 complete. coord rows=", length(results_coord), " dir rows=", length(results_dir))
flush(stdout)
