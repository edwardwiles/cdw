# A_q separation and gradient diagnostics session (2026-07-29), Phases 6-9: q-bandwidth
# estimator schemes (fixed / W-scaled / target-crossing-count) and their W-sensitivity, at
# D4. Reuses melitz_active_tail_start (sorted_tail.jl) for exact crossing counts -- same
# infrastructure the 2026-07-27 closure session's own matched-bandwidth diagnostic used
# (scripts/melitz_gradient_switch_diagnostics_2026-07-27.jl), attributed directly rather than
# re-derived.
#
# DISCLOSED SCOPE REDUCTION (session time budget): W grid restricted to {20000, 80000} (not
# the full {20000,40000,80000,160000}) at seed 29 only (no second seed) for D4; no real-D20
# W-sensitivity leg (a single real-D20 q-bandwidth point is covered by the exact-A-gradient
# validation script's own base-point solve, not repeated here). Two representative q
# directions (ordinary, and the q-pivot's own highest-leverage free coordinate), not the
# full menu of 6 direction families -- this is a genuine, disclosed proportional reduction of
# the governing prompt's Phase 8 direction list, not a silent drop.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Random, LinearAlgebra, Printf, DelimitedFiles

const OUTDIR = joinpath(REPO, "docs", "key_results")
isdir(OUTDIR) || mkpath(OUTDIR)

function fixed_dual_delta!(obj::MelitzCCBundle, theta::AbstractVector, x0::AbstractVector, ctx)
    melitz_update_operator_at_theta!(obj.op, theta, ctx)
    return -obj(x0)
end

function reoptimized_delta(obj::MelitzCCBundle, theta::AbstractVector)
    obj.use_cached_x = false
    obj.x .= NaN
    lfd = melitz_recover_lfd(obj, theta)
    return lfd.Delta, lfd.nStatus, lfd.lfd_ok
end

"Cells (linear D^2 index) whose q value differs from q0 by more than atol, plus the crossing count at each."
function q_crossing_report(theta0::AbstractVector, m::Int, h::Real, ctx, sorted_ctx)
    D = ctx.D
    nA = D^2 - 1
    theta_p = copy(theta0); theta_p[1+nA+m] += h
    _, _, _, _, q0 = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta0, ctx), ctx)
    _, _, _, _, qp = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta_p, ctx), ctx)
    total = 0
    cells = Tuple{Int,Int,Int}[]   # (o, d, crossings)
    @inbounds for o in 1:D, d in 1:D
        abs(qp[o, d] - q0[o, d]) < 1e-14 && continue
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kb = melitz_active_tail_start(sorted_z_o, exp(q0[o, d]))
        kp = melitz_active_tail_start(sorted_z_o, exp(qp[o, d]))
        c = abs(kp - kb)
        total += c
        push!(cells, (o, d, c))
    end
    return total, cells
end

"Bisect on raw q-step h to hit a target TOTAL crossing count (monotone non-decreasing in |h|)."
function find_h_for_target_crossings(theta0, m, target, ctx, sorted_ctx; h_lo=1e-8, h_hi=3e-2, max_iter=40)
    lo, hi = h_lo, h_hi
    total_hi, _ = q_crossing_report(theta0, m, hi, ctx, sorted_ctx)
    if total_hi < target
        return hi, total_hi   # cannot reach target within h_hi -- return best effort
    end
    for _ in 1:max_iter
        mid = sqrt(lo * hi)   # geometric bisection (crossing count is roughly ~sqrt(h)-ish locally)
        total_mid, _ = q_crossing_report(theta0, m, mid, ctx, sorted_ctx)
        if total_mid >= target
            hi = mid
        else
            lo = mid
        end
        hi / lo < 1.01 && break
    end
    total_final, _ = q_crossing_report(theta0, m, hi, ctx, sorted_ctx)
    return hi, total_final
end

function run_q_bandwidth(; D=4, seed=29, W, label)
    data = generate_fake_melitz_data(; D=D, seed=seed, W=W)
    obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    sorted_ctx = ctx.sorted_tail_ctx
    nA = D^2 - 1
    nq = length(theta0) - 1 - nA

    obj.use_cached_x = false; obj.x .= NaN
    lfd0 = melitz_recover_lfd(obj, theta0)
    @assert lfd0.lfd_ok
    x0 = copy(lfd0.dual_x)
    @printf("[%s] Delta0=%.6e nStatus=%d\n", label, lfd0.Delta, lfd0.nStatus)
    flush(stdout)

    # q-pivot leverage (analogous to the A-pivot leverage metric): from the actual q-pivot's
    # own coefficient ratio, via build_q_gravity_pivot.
    qpiv = build_q_gravity_pivot(ctx)
    leverage = abs.(qpiv.c[qpiv.other] ./ qpiv.c[qpiv.pivot])
    m_pivot_sensitive = argmax(leverage)
    m_ordinary = argmin(leverage)

    rows = Vector{NamedTuple}()
    for (dlabel, m) in [("ordinary", m_ordinary), ("pivot_sensitive", m_pivot_sensitive)]
        # Scheme A: fixed raw bandwidth grid.
        for h in [1e-5, 1e-4, 1e-3]
            total, cells = q_crossing_report(theta0, m, h, ctx, sorted_ctx)
            theta_p = copy(theta0); theta_p[1+nA+m] += h
            theta_m = copy(theta0); theta_m[1+nA+m] -= h
            Bp = fixed_dual_delta!(obj, theta_p, x0, ctx); Bm = fixed_dual_delta!(obj, theta_m, x0, ctx)
            secantB = (Bp - Bm) / (2h)
            Cp, nSp, okp = reoptimized_delta(obj, theta_p)
            Cm, nSm, okm = reoptimized_delta(obj, theta_m)
            secantC = (okp && okm) ? (Cp - Cm) / (2h) : NaN
            push!(rows, (label=label, direction=dlabel, scheme="fixed_A", h=h, target=missing,
                          crossings=total, secantB=secantB, secantC=secantC,
                          sign_agree=sign(secantB)==sign(secantC), ratio=secantB/secantC,
                          lfd_ok_p=okp, lfd_ok_m=okm))
            melitz_update_operator_at_theta!(obj.op, theta0, ctx)
        end
        # Scheme C: target-crossing-count bandwidth.
        for target in [10, 25, 50, 100]
            h, achieved = find_h_for_target_crossings(theta0, m, target, ctx, sorted_ctx)
            theta_p = copy(theta0); theta_p[1+nA+m] += h
            theta_m = copy(theta0); theta_m[1+nA+m] -= h
            Bp = fixed_dual_delta!(obj, theta_p, x0, ctx); Bm = fixed_dual_delta!(obj, theta_m, x0, ctx)
            secantB = (Bp - Bm) / (2h)
            Cp, nSp, okp = reoptimized_delta(obj, theta_p)
            Cm, nSm, okm = reoptimized_delta(obj, theta_m)
            secantC = (okp && okm) ? (Cp - Cm) / (2h) : NaN
            push!(rows, (label=label, direction=dlabel, scheme="target_crossing", h=h, target=target,
                          crossings=achieved, secantB=secantB, secantC=secantC,
                          sign_agree=sign(secantB)==sign(secantC), ratio=secantB/secantC,
                          lfd_ok_p=okp, lfd_ok_m=okm))
            melitz_update_operator_at_theta!(obj.op, theta0, ctx)
        end
    end
    return rows
end

all_rows = Vector{NamedTuple}()
for W in [20_000, 80_000]
    append!(all_rows, run_q_bandwidth(; D=4, seed=29, W=W, label="D4_W$(W)"))
end

open(joinpath(OUTDIR, "melitz_aq_phase6_9_q_bandwidth_2026-07-29.csv"), "w") do io
    println(io, "label,direction,scheme,h,target,crossings,secantB,secantC,sign_agree,ratio,lfd_ok_p,lfd_ok_m")
    for r in all_rows
        println(io, join([r.label, r.direction, r.scheme, r.h, r.target === missing ? "" : r.target,
                           r.crossings, r.secantB, r.secantC, r.sign_agree, r.ratio, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end
println("\nq-bandwidth diagnostics complete. CSV written.")
flush(stdout)
