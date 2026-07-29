# A_q separation and gradient diagnostics session (2026-07-29), Phase 5 continuation: exact
# A-gradient validation at real D=20, representative coordinates only (ordinary, focal-origin,
# domestic, export, normalized random A-block direction) -- per the governing prompt's own
# compute bounds (one real-D20 seed, listed directions only).

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

println("="^100)
println("A-gradient validation: real D=20, seed=1, W=80000, outer_parameterization=:logcutoff")
println("="^100)
flush(stdout)

real_dir = joinpath(REPO, "real_data", "noah_D20")
@assert isdir(real_dir)
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
    p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    outer_parameterization=:logcutoff,
    inner_loop_opt=joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=CappedEvaluation(10.0))
ctx = obj.γ
D = ctx.D
j = ctx.target_country
nA = D^2 - 1
println("D=$D, target_country=$j, nA=$nA, n_theta=$(length(theta0))")
flush(stdout)

obj.use_cached_x = false
obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("Base point solved in %.1fs: Delta0=%.6e nStatus=%d lfd_ok=%s\n", time()-t0, lfd0.Delta, lfd0.nStatus, lfd0.lfd_ok)
@assert lfd0.lfd_ok "real-D20 base point failed to verify"
x0 = copy(lfd0.dual_x)
flush(stdout)

A0, f0, gpj0, fjj0 = melitz_expand_theta(theta0, ctx)
state0 = MelitzExpandedState(D)
state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
melitz_update_operator_at_theta!(obj.op, theta0, ctx)
exact_free, exact_full = melitz_exact_a_gradient(obj, x0, state0, ctx)
bin0 = copy(obj.op.bin); rank0 = copy(obj.op.rank)

Apiv = ctx.A_pivot
rng = MersenneTwister(2026)

# Representative coordinates: ordinary (some non-focal off-diag cell), focal-origin (j,d),
# domestic (j,j), export (o,d, o!=j,d!=o,d!=j), pivot-sensitive (max |c[other]/c[pivot]|),
# and one normalized random A-block direction (tested separately as a directional secant).
leverage = abs.(Apiv.c[Apiv.other] ./ Apiv.c[Apiv.pivot])
piv_k = argmax(leverage)
jj_k = findfirst(i -> i == ctx.jj_lin, Apiv.other)
focal_k = findfirst(i -> begin (o,d) = lin2od(i, D); o == j && d != j end, Apiv.other)
ordinary_k = findfirst(i -> begin (o,d) = lin2od(i, D); o != j && d == o end, Apiv.other)
export_k = findfirst(i -> begin (o,d) = lin2od(i, D); o != j && d != o && d != j end, Apiv.other)
coord_list = [("pivot_sensitive", piv_k), ("domestic_jj", jj_k), ("focal_origin", focal_k),
              ("ordinary", ordinary_k), ("export", export_k)]
coord_list = [(lbl, k) for (lbl, k) in coord_list if k !== nothing]

hs = [1e-6, 1e-4]
rows = Vector{NamedTuple}()
for (lbl, k) in coord_list
    (o, d) = lin2od(Apiv.other[k], D)
    row_exact = exact_free[k]
    for h in hs
        theta_p = copy(theta0); theta_p[1+k] += h
        theta_m = copy(theta0); theta_m[1+k] -= h
        t1 = time()
        Bp = fixed_dual_delta!(obj, theta_p, x0, ctx)
        sw_p = obj.op.bin == bin0 && obj.op.rank == rank0
        Bm = fixed_dual_delta!(obj, theta_m, x0, ctx)
        sw_m = obj.op.bin == bin0 && obj.op.rank == rank0
        secantB = (Bp - Bm) / (2h)

        Cp, nSp, okp = reoptimized_delta(obj, theta_p)
        Cm, nSm, okm = reoptimized_delta(obj, theta_m)
        secantC = (okp && okm) ? (Cp - Cm) / (2h) : NaN
        wall = time() - t1
        @printf("[%s] o=%d d=%d h=%.0e  exact=%.6e  B=%.6e  C=%s  zeroSw=%s  wall=%.1fs\n",
            lbl, o, d, h, row_exact, secantB, okp&&okm ? @sprintf("%.6e", secantC) : "FAIL", sw_p&&sw_m, wall)
        flush(stdout)
        push!(rows, (label=lbl, D=D, o=o, d=d, k=k, h=h, exact=row_exact, secantB=secantB, secantC=secantC,
                      zero_switch_p=sw_p, zero_switch_m=sw_m, nStatus_p=nSp, nStatus_m=nSm, lfd_ok_p=okp, lfd_ok_m=okm))
    end
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
end

open(joinpath(OUTDIR, "melitz_aq_phase5_exact_a_gradient_realD20_2026-07-29.csv"), "w") do io
    println(io, "label,D,o,d,k,h,exact,secantB,secantC,zero_switch_p,zero_switch_m,nStatus_p,nStatus_m,lfd_ok_p,lfd_ok_m")
    for r in rows
        println(io, join([r.label, r.D, r.o, r.d, r.k, r.h, r.exact, r.secantB, r.secantC,
                           r.zero_switch_p, r.zero_switch_m, r.nStatus_p, r.nStatus_m, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end
println("\nreal-D20 validation complete. CSV written.")
flush(stdout)
