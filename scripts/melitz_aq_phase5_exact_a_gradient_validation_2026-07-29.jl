# A_q separation and gradient diagnostics session (2026-07-29), Phase 5: validate the exact
# envelope-theorem A-block gradient (src/melitz/exact_a_gradient.jl) against
#   (B) a direct fixed-dual directional secant (same dual x0, displaced theta, no re-solve)
#   (C) a fully reoptimized DeltaStar secant (re-solved inner problem at displaced theta)
# at D=4 (every free A coordinate) and a representative set at real D=20, under
# outer_parameterization=:logcutoff (the (A,q) coordinate system this session investigates --
# the exact gradient is a fixed-q object by construction, Phase 1-2).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Random, LinearAlgebra, Printf, DelimitedFiles

const OUTDIR = joinpath(REPO, "docs", "key_results")
isdir(OUTDIR) || mkpath(OUTDIR)

function fixed_dual_delta!(obj::MelitzCCBundle, theta::AbstractVector, x0::AbstractVector, ctx)
    melitz_update_operator_at_theta!(obj.op, theta, ctx)
    f = obj(x0)
    return -f
end

function reoptimized_delta(obj::MelitzCCBundle, theta::AbstractVector)
    obj.use_cached_x = false
    obj.x .= NaN
    lfd = melitz_recover_lfd(obj, theta)
    return lfd.Delta, lfd.nStatus, lfd.lfd_ok
end

function bin_rank_unchanged(op::MelitzMomentOperator, bin0::Matrix{UInt8}, rank0::Matrix{Int})
    return op.bin == bin0 && op.rank == rank0
end

function run_validation(; D::Int, seed::Int, W::Int, label::String, test_coords::Vector{Int}=Int[],
                          hs::Vector{Float64}=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    println("="^100)
    println("A-gradient validation: $label (D=$D, seed=$seed, W=$W)")
    println("="^100)
    flush(stdout)

    data = generate_fake_melitz_data(; D=D, seed=seed, W=W)
    obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    nA = D^2 - 1

    obj.use_cached_x = false
    obj.x .= NaN
    lfd0 = melitz_recover_lfd(obj, theta0)
    @assert lfd0.lfd_ok "base point LFD failed to verify -- fixture not usable for this diagnostic"
    x0 = copy(lfd0.dual_x)
    Delta0 = lfd0.Delta
    @printf("Base point: Delta0=%.6e, nStatus=%d, lfd_ok=%s\n", Delta0, lfd0.nStatus, lfd0.lfd_ok)
    flush(stdout)

    A0, f0, gpj0, fjj0 = melitz_expand_theta(theta0, ctx)
    state0 = MelitzExpandedState(D)
    state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)   # ensure obj.op reflects theta0
    exact_free, exact_full = melitz_exact_a_gradient(obj, x0, state0, ctx)

    bin0 = copy(obj.op.bin)
    rank0 = copy(obj.op.rank)

    coords = isempty(test_coords) ? collect(1:nA) : test_coords
    rows = Vector{NamedTuple}()
    for k in coords
        (o, d) = lin2od(ctx.A_pivot.other[k], D)
        row_exact = exact_free[k]
        for h in hs
            theta_p = copy(theta0); theta_p[1+k] += h
            theta_m = copy(theta0); theta_m[1+k] -= h

            # Zero-switch check: q (hence bin/rank) must be EXACTLY unchanged.
            Bp = fixed_dual_delta!(obj, theta_p, x0, ctx)
            switches_p = bin_rank_unchanged(obj.op, bin0, rank0)
            Bm = fixed_dual_delta!(obj, theta_m, x0, ctx)
            switches_m = bin_rank_unchanged(obj.op, bin0, rank0)
            secantB = (Bp - Bm) / (2h)

            Cp, nSp, okp = reoptimized_delta(obj, theta_p)
            Cm, nSm, okm = reoptimized_delta(obj, theta_m)
            secantC = (okp && okm) ? (Cp - Cm) / (2h) : NaN

            push!(rows, (label=label, D=D, o=o, d=d, k=k, h=h, exact=row_exact,
                          secantB=secantB, secantC=secantC,
                          zero_switch_p=switches_p, zero_switch_m=switches_m,
                          nStatus_p=nSp, nStatus_m=nSm, lfd_ok_p=okp, lfd_ok_m=okm))
        end
        # restore operator to theta0 before the next coordinate
        melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    end

    # Summary print: smallest-h row per coordinate.
    println("\n-- summary (h=$(minimum(hs))) --")
    @printf("%-4s %-4s %-4s %14s %14s %14s %8s\n", "o", "d", "k", "exact", "secantB", "secantC", "zeroSw")
    for r in rows
        r.h == minimum(hs) || continue
        zs = r.zero_switch_p && r.zero_switch_m
        @printf("%-4d %-4d %-4d %14.6e %14.6e %14.6e %8s\n", r.o, r.d, r.k, r.exact, r.secantB, r.secantC, zs)
    end
    flush(stdout)

    return rows
end

# D=4: every free A coordinate.
rows_d4 = run_validation(D=4, seed=29, W=20_000, label="D4_calibration")

# Write CSV
open(joinpath(OUTDIR, "melitz_aq_phase5_exact_a_gradient_d4_2026-07-29.csv"), "w") do io
    println(io, "label,D,o,d,k,h,exact,secantB,secantC,zero_switch_p,zero_switch_m,nStatus_p,nStatus_m,lfd_ok_p,lfd_ok_m")
    for r in rows_d4
        println(io, join([r.label, r.D, r.o, r.d, r.k, r.h, r.exact, r.secantB, r.secantC,
                           r.zero_switch_p, r.zero_switch_m, r.nStatus_p, r.nStatus_m, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end

println("\nD=4 validation complete. CSV written.")
flush(stdout)
