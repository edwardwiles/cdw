# Governing prompt continuation (2026-07-27 night session), Phases 2-5: dynamic allocation
# audit of the production hot call graph, at D=4 (W=20,000) and real D=20 (W=80,000,
# n_theta~798). Standalone script -- reuses production entry points exclusively
# (MelitzCCBundle's own functor, melitz_update_operator_at_theta!, melitz_range_screen,
# the production gradient-backend factories, melitz_recover_lfd_from_solution, cb_F!/cb_G!
# via a mock KNITRO eval request/result). Measures @allocated post-JIT-warmup (2 consecutive
# calls each) -- not Profile.Allocs stack traces (disclosed scope reduction, matching this
# repo's own established convention for a single-session allocation pass).

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)

mutable struct MockEvalRequest
    x::Vector{Float64}
end
mutable struct MockEvalResult
    obj::Vector{Float64}
    c::Vector{Float64}
    objGrad::Vector{Float64}
    jac::Vector{Float64}
end

rows = Vector{NamedTuple}()

function audit_scale!(rows, label::String, obj, ctx, theta0::Vector{Float64}, W::Int)
    n = length(theta0)
    D = ctx.D
    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
    @assert r0.verified "$label: calibration point failed to verify (nStatus=$(r0.nStatus))"
    x0 = r0.dual_x
    println(label, ": Delta0=", r0.Delta, " nStatus=", r0.nStatus, " n_theta=", n, " D=", D, " W=", W)

    # 1. fixed-outer-point operator update
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)   # warmup
    b1 = @allocated melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    b2 = @allocated melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    push!(rows, (scale=label, callback="operator_update_at_theta", bytes1=b1, bytes2=b2))

    # 2. matrix-free objective callback (fixed operator state, at x0)
    obj(x0, Float64[])
    b1 = @allocated obj(x0, Float64[])
    b2 = @allocated obj(x0, Float64[])
    push!(rows, (scale=label, callback="objective_callback", bytes1=b1, bytes2=b2))

    # 3. matrix-free (inner-dual) gradient callback (length = d+1: zeta + one per moment,
    # NOT n_theta+1 -- the inner dual's own dimension, distinct from the outer theta gradient)
    gbuf = zeros(obj.d + 1)
    obj(x0, gbuf)
    b1 = @allocated obj(x0, gbuf)
    b2 = @allocated obj(x0, gbuf)
    push!(rows, (scale=label, callback="inner_dual_gradient_callback", bytes1=b1, bytes2=b2))

    # 4. serial structured Hessian callback (+ packing) -- packed size n_dual*(n_dual+1)/2
    # where n_dual = d+1 (the inner dual's own dimension), not n_theta.
    n_dual = obj.d + 1
    nh = n_dual * (n_dual + 1) ÷ 2
    hbuf = zeros(nh)
    obj(x0, Float64[]; h=hbuf)
    b1 = @allocated obj(x0, Float64[]; h=hbuf)
    b2 = @allocated obj(x0, Float64[]; h=hbuf)
    push!(rows, (scale=label, callback="hessian_callback_$(obj.hessian_backend)", bytes1=b1, bytes2=b2))

    # 5. matrix-free range screen
    melitz_range_screen(obj.op)
    b1 = @allocated melitz_range_screen(obj.op)
    b2 = @allocated melitz_range_screen(obj.op)
    push!(rows, (scale=label, callback="matrix_free_range_screen", bytes1=b1, bytes2=b2))

    # 6/7. complete outer gradient (serial), h=1e-4 (production default bandwidth)
    gfun_serial = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
    gvec = zeros(n)
    gfun_serial(gvec, theta0, ctx, obj, x0)
    b1 = @allocated gfun_serial(gvec, theta0, ctx, obj, x0)
    b2 = @allocated gfun_serial(gvec, theta0, ctx, obj, x0)
    push!(rows, (scale=label, callback="complete_outer_gradient_serial (n=$n coords => 0 bytes implies 0/coord)", bytes1=b1, bytes2=b2))

    if Threads.nthreads() > 1
        gfun_par = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
        gvec2 = zeros(n)
        gfun_par(gvec2, theta0, ctx, obj, x0)
        b1 = @allocated gfun_par(gvec2, theta0, ctx, obj, x0)
        b2 = @allocated gfun_par(gvec2, theta0, ctx, obj, x0)
        push!(rows, (scale=label, callback="complete_outer_gradient_parallel", bytes1=b1, bytes2=b2))
        @assert isapprox(gvec, gvec2; rtol=1e-8) "serial/parallel gradient mismatch at $label"
    end

    # LFD recovery + moment/KKT verification (one function in this codebase's architecture)
    melitz_recover_lfd_from_solution(r0.Delta, x0, r0.nStatus, theta0, obj)   # warmup
    b1 = @allocated melitz_recover_lfd_from_solution(r0.Delta, x0, r0.nStatus, theta0, obj)
    b2 = @allocated melitz_recover_lfd_from_solution(r0.Delta, x0, r0.nStatus, theta0, obj)
    push!(rows, (scale=label, callback="lfd_recovery_and_kkt_verification", bytes1=b1, bytes2=b2))

    # candidate registration path: cb_F!/cb_G! via melitz_build_finite_delta_callbacks, at
    # the SAME theta (exact-point cache hit) and at a FRESH theta (full recompute).
    m = 1 + D + D * (D - 1)
    delta_loose = max(r0.Delta * 5, 1e-3)
    cbset = melitz_build_finite_delta_callbacks(obj, ctx, delta_loose, true;
        gradient_backend=:B_direct_argument_sorted_serial, h=1e-4, cutoff_constraint_backend=:linear)
    evalRes = MockEvalResult(zeros(1), zeros(m), zeros(n), zeros(n * m))
    cbset.cb_F!(nothing, nothing, MockEvalRequest(copy(theta0)), evalRes, nothing)
    cbset.cb_G!(nothing, nothing, MockEvalRequest(copy(theta0)), evalRes, nothing)   # warmup
    b1f = @allocated cbset.cb_F!(nothing, nothing, MockEvalRequest(copy(theta0)), evalRes, nothing)
    b1g = @allocated cbset.cb_G!(nothing, nothing, MockEvalRequest(copy(theta0)), evalRes, nothing)
    push!(rows, (scale=label, callback="cb_F_candidate_registration_cache_hit_theta", bytes1=b1f, bytes2=b1g))

    return nothing
end

# ----------------------------------------------------------------------------------------
# D=4 fixture
# ----------------------------------------------------------------------------------------
data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true)
audit_scale!(rows, "D4_W20000", obj4, obj4.γ, theta0_4, 20_000)

# ----------------------------------------------------------------------------------------
# Real D=20 fixture (noah_D20), matching test/melitz/runtests.jl's own real-data recipe.
# ----------------------------------------------------------------------------------------
real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
if isdir(real_dir)
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData,
        countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
    p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
    theta0_20 = melitz_reduce_theta(p20, ctx20)
    op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
    hessian_backend20 = melitz_resolve_hessian_backend(MelitzBackendConfig(inner_backend=:matrix_free), ctx20.D)
    obj20 = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
        outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
        inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt,
        hessian_backend=hessian_backend20)
    audit_scale!(rows, "realD20_W80000", obj20, ctx20, theta0_20, 80_000)
else
    println("real_data/noah_D20 not found -- real D=20 audit SKIPPED (disclosed)")
end

outfile = joinpath(OUTDIR, "melitz_phase2_5_allocation_audit_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("Wrote ", outfile, " (", length(rows), " rows)")
for r in rows
    println(rpad(r.scale, 16), " ", rpad(r.callback, 60), " bytes1=", r.bytes1, " bytes2=", r.bytes2)
end
