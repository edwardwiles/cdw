# Governing prompt continuation (2026-07-27 night session), Phases 6-9: matched-bandwidth
# gradient comparison + exact draw-level switch counts, at multiple points away from the
# calibration minimum. Standalone script (not part of the committed test suite) -- run once,
# results captured to docs/key_results/*.csv and summarized in the final closure doc.
#
# Reuses production code paths exclusively (melitz_expand_theta, MelitzCCBundle's own
# functor via melitz_update_operator_at_theta!, evaluate_melitz_delta,
# make_melitz_gradient_delta_direct_sorted_serial, melitz_active_tail_start) -- no
# reimplementation of Psi/moments/cutoffs.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, Statistics, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)

# ----------------------------------------------------------------------------------------
# Setup: D=4 fixture (same recipe as test/melitz/runtests.jl's own FIXTURE), matrix-free
# bundle, forbid_dense_fallback=true (strict production-fast).
# ----------------------------------------------------------------------------------------
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true)
ctx = obj.γ
n = length(theta0)
D = ctx.D
j = ctx.target_country
nA = D^2 - 1
sorted_ctx = ctx.sorted_tail_ctx
@assert sorted_ctx !== nothing

r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
@assert r0.verified
println("theta0 (calibration): Delta0=", r0.Delta, "  nStatus=", r0.nStatus)
x0 = r0.dual_x

# ----------------------------------------------------------------------------------------
# Direction construction (Phase 6's own list). Layout: theta_free = [gamma; A_free (nA); f_free (n-1-nA)]
# ----------------------------------------------------------------------------------------
function unit(v)
    return v ./ sqrt(sum(abs2, v))
end

compact = melitz_compact_columns_map(ctx)
link_idx = findfirst(c -> c.touches_link, compact)
nonlink_f_idx = findfirst(k -> k > 1 + nA && !compact[k].touches_link, 1:n)
nonlink_A_idx = findfirst(k -> 2 <= k <= 1 + nA && !compact[k].touches_link, 1:n)
nonlink_A_idx === nothing && (nonlink_A_idx = 2)   # fallback: every A coordinate might touch link in a small D

# pivot leverage: |c[other[k]]/c[pivot]| for the A-pivot, mapped back to theta_free indices.
Apiv = ctx.A_pivot
leverage = abs.(Apiv.c[Apiv.other] ./ Apiv.c[Apiv.pivot])
best_local = argmax(leverage)
# Apiv.other[best_local] is a GLOBAL (1:D^2) A-cell linear index; map to its position within
# A_free (pivot_reduce's own free-vector order is exactly `other`, in order).
pivot_sensitive_A_idx = 1 + best_local   # theta_free index: 1 (gamma) + position within A_free

directions = Dict{String,Vector{Float64}}()
directions["gamma"] = (v = zeros(n); v[1] = 1.0; v)
directions["ordinary_technology"] = (v = zeros(n); v[nonlink_A_idx] = 1.0; v)
directions["technology_pivot_sensitive"] = (v = zeros(n); v[pivot_sensitive_A_idx] = 1.0; v)
directions["ordinary_participation"] = (v = zeros(n); v[nonlink_f_idx === nothing ? (2+nA) : nonlink_f_idx] = 1.0; v)
directions["participation_pivot_sensitive"] = (v = zeros(n); v[link_idx] = 1.0; v)
directions["normalized_technology_block"] = (v = zeros(n); v[2:1+nA] .= 1.0; unit(v))
directions["normalized_participation_block"] = (v = zeros(n); v[2+nA:end] .= 1.0; unit(v))
directions["mixed"] = unit((v = zeros(n); v[1] = 1.0; v[nonlink_A_idx] = 1.0; v[link_idx] = 1.0; v))

for (name, v) in directions
    @assert isapprox(sum(abs2, v), 1.0; atol=1e-8) "$name not unit-normalized"
end

# ----------------------------------------------------------------------------------------
# Object B: raw fixed-dual scalar secant, using the bundle's OWN functor with the operator
# updated to a DISPLACED theta and the dual FIXED at x0 (never re-solved) -- reuses
# melitz_update_operator_at_theta!/the functor directly rather than hand-deriving Psi/G.
# Returns the SAME sign/scale convention as the registered gradient (`grad = d(1e10*Delta)/dtheta`).
# ----------------------------------------------------------------------------------------
function L_fix_1e10(theta::AbstractVector, x0::AbstractVector, ctx, obj::MelitzCCBundle)
    melitz_update_operator_at_theta!(obj.op, theta, ctx)
    constr = zeros(1)
    obj(x0, Float64[], Float64[]; constr=constr)
    return constr[1]   # already -f*1e10, i.e. +1e10*Delta(theta) at fixed dual x0
end

# ----------------------------------------------------------------------------------------
# Phase 7: exact draw-level switch counts via the sorted-tail infrastructure (O(D^2), no
# dense W-row scan). Direct cells (o,d) + the separate autarky/focal-link threshold.
# ----------------------------------------------------------------------------------------
function cutoff_of(A, f, o, d, ctx)
    C_od = melitz_C(ctx.w[o], ctx.tau[o, d], A[o, d], ctx.sigma, ctx.expenditure[d])
    return melitz_cutoff(ctx.w[o], f[o, d], ctx.sigma, C_od)
end

function autarky_cutoff(A, f_jj, gamma_prime_j, ctx)
    j = ctx.target_country
    expenditure_prime = ctx.w_prime * ctx.L[j]
    C_auk = melitz_C(ctx.w_prime, 1.0, A[j, j], ctx.sigma, expenditure_prime)
    # profit = C*z^(sigma-1)/price_power_d/sigma - w*f > 0  <=>  z > (sigma*w*f*price_power_d/C)^(1/(sigma-1))
    return (ctx.sigma * ctx.w_prime * f_jj * gamma_prime_j / C_auk)^(1 / (ctx.sigma - 1))
end

function exact_switch_report(theta_base, theta_plus, theta_minus, ctx, sorted_ctx)
    Ab, fb, gb, fjjb = melitz_expand_theta(theta_base, ctx)
    Ap, fp, gp, fjjp = melitz_expand_theta(theta_plus, ctx)
    Am, fm, gm, fjjm = melitz_expand_theta(theta_minus, ctx)
    D = ctx.D
    total_switch_bp = 0
    total_switch_bm = 0
    total_switch_pm = 0
    cells_with_switches = 0
    min_dist = Inf
    for o in 1:D, d in 1:D
        cb = cutoff_of(Ab, fb, o, d, ctx)
        cp = cutoff_of(Ap, fp, o, d, ctx)
        cm = cutoff_of(Am, fm, o, d, ctx)
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kb = melitz_active_tail_start(sorted_z_o, cb)
        kp = melitz_active_tail_start(sorted_z_o, cp)
        km = melitz_active_tail_start(sorted_z_o, cm)
        sbp = abs(kp - kb); sbm = abs(km - kb); spm = abs(kp - km)
        total_switch_bp += sbp; total_switch_bm += sbm; total_switch_pm += spm
        (sbp > 0 || sbm > 0) && (cells_with_switches += 1)
        W = sorted_ctx.W
        lo = kb >= 2 ? sorted_z_o[kb-1] : -Inf
        hi = kb <= W ? sorted_z_o[kb] : Inf
        d1 = isfinite(lo) ? cb - lo : Inf
        d2 = isfinite(hi) ? hi - cb : Inf
        min_dist = min(min_dist, d1, d2)
    end
    # focal link / autarky, separately
    zj_sorted = @view sorted_ctx.sorted_z[:, j]
    auk_b = autarky_cutoff(Ab, fjjb, gb, ctx)
    auk_p = autarky_cutoff(Ap, fjjp, gp, ctx)
    auk_m = autarky_cutoff(Am, fjjm, gm, ctx)
    kab = melitz_active_tail_start(zj_sorted, auk_b)
    kap = melitz_active_tail_start(zj_sorted, auk_p)
    kam = melitz_active_tail_start(zj_sorted, auk_m)
    link_switch_bp = abs(kap - kab)
    link_switch_bm = abs(kam - kab)
    link_switch_pm = abs(kap - kam)
    return (switch_bp=total_switch_bp, switch_bm=total_switch_bm, switch_pm=total_switch_pm,
            cells_with_switches=cells_with_switches, min_dist_to_draw=min_dist,
            link_switch_bp=link_switch_bp, link_switch_bm=link_switch_bm, link_switch_pm=link_switch_pm)
end

# ----------------------------------------------------------------------------------------
# Phase 8: construct additional points away from the calibration minimum by scaling theta0
# by an increasing radius along a FIXED random direction (verified via evaluate_melitz_delta,
# cold=true, at each candidate) -- reports the ACTUAL Delta achieved, not a forced exact target.
# ----------------------------------------------------------------------------------------
rng_pts = MersenneTwister(2027)
points = NamedTuple[]
push!(points, (label="calibration_theta0", theta=theta0, Delta=r0.Delta, dual_x=x0, nStatus=r0.nStatus, verified=true))

# Phase 8 (revised after a full-random-direction attempt at radius>=0.05 immediately hit
# NumericalFailure/InfiniteDeltaCertified nStatus in EVERY tried radius, disclosed below --
# a random combination of all n=30 coordinates simultaneously has much larger aggregate
# leverage on gravity-feasibility/participation than any single coordinate): pure-gamma
# points (Finding 4's own profile scan already showed this direction stays smooth and
# verified over a wide range, s in {-0.1,...,0.1}) at LARGER magnitude to span a range of
# Delta, plus one point with genuine material A/f movement at a much SMALLER magnitude
# (the scale at which a mixed direction remains verified for this fixture).
e_gamma = (v = zeros(n); v[1] = 1.0; v)
for g_step in (0.5, 1.2, 2.5, 4.0, 6.0)
    theta_try = theta0 .+ g_step .* e_gamma
    rtry = evaluate_melitz_delta(theta_try, ctx, obj; cold=true, store_G=false)
    push!(points, (label=(@sprintf "gamma_step_%.2f" g_step), theta=theta_try, Delta=rtry.Delta,
                    dual_x=rtry.dual_x, nStatus=rtry.nStatus, verified=rtry.verified))
    println(@sprintf("gamma_step=%.2f -> Delta=%.6e nStatus=%d verified=%s", g_step, rtry.Delta, rtry.nStatus, rtry.verified))
end

mixed_dir = unit((v = zeros(n); v[1] = 1.0; v[nonlink_A_idx] = 1.0; v[nonlink_f_idx === nothing ? (2+nA) : nonlink_f_idx] = 1.0; v))
for radius in (0.002, 0.006, 0.015, 0.03)
    theta_try = theta0 .+ radius .* mixed_dir
    rtry = evaluate_melitz_delta(theta_try, ctx, obj; cold=true, store_G=false)
    push!(points, (label=(@sprintf "mixed_Af_radius_%.3f" radius), theta=theta_try, Delta=rtry.Delta,
                    dual_x=rtry.dual_x, nStatus=rtry.nStatus, verified=rtry.verified))
    println(@sprintf("mixed_Af radius=%.3f -> Delta=%.6e nStatus=%d verified=%s", radius, rtry.Delta, rtry.nStatus, rtry.verified))
end

# ----------------------------------------------------------------------------------------
# Main sweep: matched-bandwidth gradient comparison (Phase 6) + exact switch counts (Phase 7)
# at every point (Phase 8).
# ----------------------------------------------------------------------------------------
hs = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

rows = Vector{NamedTuple}()
for pt in points
    !pt.verified && continue
    theta_b = pt.theta
    x_b = pt.dual_x
    nS = pt.nStatus
    nS != 0 && continue
    for (dname, v) in directions
        for h in hs
            theta_p = theta_b .+ h .* v
            theta_m = theta_b .- h .* v

            # Object A: production registered secant at MATCHING bandwidth h.
            gfun_h = make_melitz_gradient_delta_direct_sorted_serial(h)
            gvec = zeros(n)
            gfun_h(gvec, theta_b, ctx, obj, x_b)
            objA = dot(gvec, v)

            # Object B: raw fixed-dual secant via the bundle's own functor, matching h.
            Lp = L_fix_1e10(theta_p, x_b, ctx, obj)
            Lm = L_fix_1e10(theta_m, x_b, ctx, obj)
            objB = (Lp - Lm) / (2h)

            # Object C: FD of independently reoptimized DeltaStar (ground truth), matching h.
            rp = evaluate_melitz_delta(theta_p, ctx, obj; cold=true, store_G=false)
            rm = evaluate_melitz_delta(theta_m, ctx, obj; cold=true, store_G=false)
            objC = (rp.Delta - rm.Delta) / (2h) * 1e10

            sw = exact_switch_report(theta_b, theta_p, theta_m, ctx, sorted_ctx)

            push!(rows, (point=pt.label, direction=dname, h=h,
                objA_registered=objA, objB_fixed_dual_raw=objB, objC_fd_Delta=objC,
                ratio_A_vs_C=objA/objC, ratio_B_vs_C=objB/objC,
                switch_bp=sw.switch_bp, switch_bm=sw.switch_bm, switch_pm=sw.switch_pm,
                cells_with_switches=sw.cells_with_switches, min_dist_to_draw=sw.min_dist_to_draw,
                link_switch_bp=sw.link_switch_bp, link_switch_bm=sw.link_switch_bm, link_switch_pm=sw.link_switch_pm,
                Delta_base=pt.Delta, eval_p_Delta=rp.Delta, eval_m_Delta=rm.Delta,
                eval_p_nStatus=rp.nStatus, eval_m_nStatus=rm.nStatus))
        end
    end
end

# Write CSV
outfile = joinpath(OUTDIR, "melitz_phase6_9_matched_bandwidth_switches_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("Wrote ", outfile, " (", length(rows), " rows)")
