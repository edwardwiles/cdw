# Governing prompt (outer-search session), Phase 7: select block scales at D=4.
#
# D=4, W=20,000, seed=29 (this repo's own standing FIXTURE), production default
# :logA/:logf, strict production-fast, evaluation cap=10, dimensionless divergence
# constraint. Base point: the D=4 FIXTURE's own calibrated point -- a genuinely separate
# D=4 "near-boundary" point (distinct from calibration) was not independently established in
# any prior D=20-focused local-geometry session, so this script uses the calibration point
# directly (disclosed, not silently substituted) -- Section 8 of the prior closure doc's own
# finding (the calibration point is close to a critical point AND close to several
# participation-switch thresholds simultaneously) makes it an economically meaningful,
# already-characterized reference point for this purpose, not an arbitrary stand-in.
#
# For each grid point: fully reoptimized finite DeltaStar (`evaluate_melitz_delta(...;
# cold=true)`, a genuine fresh inner KNITRO solve at the displaced theta -- NOT a fixed-dual
# probe), realized DeltaStar change, exact draw-level participation switches (reusing the
# already-validated sorted-tail `melitz_active_tail_start` exact-switch method from
# scripts/melitz_gradient_switch_diagnostics_2026-07-27.jl), cutoff slack, inner nStatus.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, Statistics, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true)
ctx = obj.γ
n = length(theta0)
D = ctx.D
nA = D^2 - 1
sorted_ctx = ctx.sorted_tail_ctx
@assert sorted_ctx !== nothing

r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
@assert r0.verified
println("theta0 (calibration/base point): Delta0=", r0.Delta, "  nStatus=", r0.nStatus)

function cutoff_of(A, f, o, d, ctx)
    C_od = melitz_C(ctx.w[o], ctx.tau[o, d], A[o, d], ctx.sigma, ctx.expenditure[d])
    return melitz_cutoff(ctx.w[o], f[o, d], ctx.sigma, C_od)
end
function autarky_cutoff(A, f_jj, gamma_prime_j, ctx)
    j = ctx.target_country
    C_jj = melitz_C(ctx.w[j] * gamma_prime_j, 1.0, A[j, j], ctx.sigma, ctx.expenditure[j])
    return melitz_cutoff(ctx.w[j] * gamma_prime_j, f_jj, ctx.sigma, C_jj)
end

"Exact draw-level switch count between two theta vectors, direct cells + focal link, via the sorted-tail exact method."
function exact_switches(theta_a, theta_b, ctx, sorted_ctx)
    A_a, f_a, gp_a, fjj_a = melitz_expand_theta(theta_a, ctx)
    A_b, f_b, gp_b, fjj_b = melitz_expand_theta(theta_b, ctx)
    D_ = ctx.D
    total = 0
    for o in 1:D_, d in 1:D_
        d == o && continue
        ca = cutoff_of(A_a, f_a, o, d, ctx)
        cb = cutoff_of(A_b, f_b, o, d, ctx)
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        ka = melitz_active_tail_start(sorted_z_o, ca)
        kb = melitz_active_tail_start(sorted_z_o, cb)
        total += abs(ka - kb)
    end
    zj_sorted = @view sorted_ctx.sorted_z[:, ctx.target_country]
    auk_a = autarky_cutoff(A_a, fjj_a, gp_a, ctx)
    auk_b = autarky_cutoff(A_b, fjj_b, gp_b, ctx)
    ka = melitz_active_tail_start(zj_sorted, auk_a)
    kb = melitz_active_tail_start(zj_sorted, auk_b)
    total += abs(ka - kb)
    return total
end

function min_cutoff_slack(theta, ctx)
    A, f, gp, fjj = melitz_expand_theta(theta, ctx)
    D_ = ctx.D
    mn = Inf
    for o in 1:D_, d in 1:D_
        d == o && continue
        c = cutoff_of(A, f, o, d, ctx)
        mn = min(mn, c - 1.0)   # export cutoff must exceed domestic-normalized threshold; use c-1 as a slack proxy
    end
    return mn
end

compact = melitz_compact_columns_map(ctx)
g_idx = 1
A_idx_all = collect(2:1+nA)
f_idx_all = collect(2+nA:n)

rows = NamedTuple[]

function probe!(rows, label, theta_disp, base_theta, r0, ctx, obj, sorted_ctx)
    r = evaluate_melitz_delta(theta_disp, ctx, obj; cold=true, store_G=false)
    sw = exact_switches(base_theta, theta_disp, ctx, sorted_ctx)
    slack = r.nStatus == 0 ? min_cutoff_slack(theta_disp, ctx) : NaN
    push!(rows, (block=label[1], raw_step=label[2], Delta=r.nStatus == 0 ? r.Delta : NaN,
        realized_Delta_change=r.nStatus == 0 ? (r.Delta - r0.Delta) : NaN,
        exact_switches=sw, cutoff_slack=slack, nStatus=r.nStatus, verified=r.verified))
    println(@sprintf("%-14s step=%.2e  nStatus=%-4d Delta=%s  switches=%d  slack=%s",
        label[1], label[2], r.nStatus, r.nStatus == 0 ? @sprintf("%.6e", r.Delta) : "NA", sw,
        isnan(slack) ? "NA" : @sprintf("%.4e", slack)))
end

println("\n== g block (raw |dg|) ==")
for dg in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
    for sgn in (1.0, -1.0)
        theta_disp = copy(theta0); theta_disp[g_idx] += sgn * dg
        probe!(rows, ("g", sgn * dg), theta_disp, theta0, r0, ctx, obj, sorted_ctx)
    end
end

println("\n== technology block (normalized direction, aggregate raw norm grid) ==")
rng = MersenneTwister(2027)
tech_dir = zeros(n); tech_dir[A_idx_all] .= randn(rng, length(A_idx_all)); tech_dir ./= sqrt(sum(abs2, tech_dir))
for nrm in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    for sgn in (1.0, -1.0)
        theta_disp = theta0 .+ (sgn * nrm) .* tech_dir
        probe!(rows, ("technology", sgn * nrm), theta_disp, theta0, r0, ctx, obj, sorted_ctx)
    end
end

println("\n== participation block (normalized direction, aggregate raw norm grid) ==")
part_dir = zeros(n); part_dir[f_idx_all] .= randn(rng, length(f_idx_all)); part_dir ./= sqrt(sum(abs2, part_dir))
for nrm in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    for sgn in (1.0, -1.0)
        theta_disp = theta0 .+ (sgn * nrm) .* part_dir
        probe!(rows, ("participation", sgn * nrm), theta_disp, theta0, r0, ctx, obj, sorted_ctx)
    end
end

outfile = joinpath(OUTDIR, "melitz_phase7_d4_scale_selection_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("\nWrote ", outfile)
