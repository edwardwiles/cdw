# ============================================================================
# jac_h runtime audit, D=6/8 microbenchmark (kept in its OWN process,
# deliberately separate from audit_jach.jl -- see that script's tail comment:
# context.jl/context_scaled.jl are not designed to be include()'d twice in
# one Julia process, since re-including redefines the CounterfactualSensitivity
# module and breaks type identity for already-constructed objects).
#
# Per docs/fullA_block_local_performance.md sec 7, a prior session found the
# calibration-anchored base point cold-infeasible at D=6 -- this script tries
# the calibration point first, then a handful of small random perturbations
# (bounded attempts, per the task's explicit "don't spend excessive time
# chasing a feasible D=6 point" instruction) before reporting D unreachable.
#
# Run: julia --project=. full_aod_diag/d4_exact/audit_jach_d6d8.jl
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # -> ctx-building via d_exact_setup_scaled, CS, etc.
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
using Random, Statistics, Printf

function cmp_scalar(name, a, b; tol = 1e-8)
    ok = (isnan(a) && isnan(b)) ? true : abs(a - b) <= tol
    ok || println("    MISMATCH: ", name, "  a=", a, "  b=", b)
    return ok
end

function try_scaled_feasible(D, W; tries = 6)
    Random.seed!(4200 + D)
    local ctxS, x_try
    for attempt in 1:tries
        ctxS = d_exact_setup_scaled(D = D, W = W, needs_outer_moment_jacobian = true)
        x_try = attempt == 1 ? CS.pack_free(ctxS.θ0_up, ctxS.m) :
                CS.pack_free(ctxS.θ0_up, ctxS.m) .* (1 .+ 0.02 .* (2 .* rand(CS.n_free(ctxS.m)) .- 1))
        r = evaluate_fullA(x_try, ctxS; cache = nothing, warm = true)
        println("  D=$D attempt $attempt/$tries: inner_status=", r.inner_status,
                r.inner_status in (0, -100, -101, -103) ? "  FEASIBLE" : "  infeasible")
        if r.inner_status in (0, -100, -101, -103)
            return ctxS, x_try, attempt
        end
    end
    return nothing, nothing, tries
end

println("="^78); println("jac_h AUDIT: D=6/8 microbenchmark"); println("="^78)

for D_try in (6, 8)
    println("\n---- D=$D_try ----")
    ctxS, xS, attempts = try_scaled_feasible(D_try, 8000)
    if ctxS === nothing
        println("D=$D_try: NO feasible point found in $attempts random attempts (calibration + perturbations)")
        println("  -- SKIPPED, per the task's explicit allowance (\"D=4-only validation is acceptable")
        println("     if D=6 proves impractical, just say so clearly\").")
        continue
    end
    println("D=$D_try: feasible point found on attempt $attempts/6.")

    NS = ctxS.obj.N; dS = ctxS.obj.d; lS = ctxS.obj.l
    bytesS = 8 * NS * (dS + 2) * lS
    println("  N(=W)=", NS, " d(=nTotalMoments)=", dS, " l(=l_full)=", lS,
            "  theoretical jac_h bytes=", bytesS, " (", round(bytesS / 1024^2, digits = 2), " MB)")
    println("  measured size(ctxS.obj.jac_h)=", size(ctxS.obj.jac_h))

    # ---- second context, jac_h disabled, SAME economy/draws (fresh process so no module-identity issue) ----
    ctxS2 = d_exact_setup_scaled(D = D_try, W = 8000, needs_outer_moment_jacobian = false)
    println("  measured size(ctxS2.obj.jac_h)=", size(ctxS2.obj.jac_h))
    @assert ctxS.θ0_up == ctxS2.θ0_up && ctxS.U == ctxS2.U "ctxS/ctxS2 must be the identical economy/draws"

    # ---- correctness: Delta_dual/zeta/lambda agreement at the feasible point ----
    r1S = evaluate_fullA(xS, ctxS; cache = nothing, warm = true)
    r2S = evaluate_fullA(xS, ctxS2; cache = nothing, warm = true)
    ok1 = cmp_scalar("Delta_dual", r1S.Delta_dual, r2S.Delta_dual)
    ok2 = cmp_scalar("zeta", r1S.zeta, r2S.zeta)
    ok3 = length(r1S.lambda) == length(r2S.lambda) && maximum(abs.(r1S.lambda .- r2S.lambda)) <= 1e-8
    ok3 || println("    MISMATCH: lambda vectors differ")
    println("  ctx/ctx2 agreement at D=$D_try: Delta_dual=", ok1, " zeta=", ok2, " lambda=", ok3,
            "  ", (ok1 && ok2 && ok3) ? "PASS" : "FAIL")

    # ---- performance: constructor time WITH vs WITHOUT jac_h, n=5 reps each, alternating order ----
    tw = Float64[]; two = Float64[]
    for _ in 1:5
        t0 = time(); d_exact_setup_scaled(D = D_try, W = 8000, needs_outer_moment_jacobian = true); push!(tw, time() - t0)
        t0 = time(); d_exact_setup_scaled(D = D_try, W = 8000, needs_outer_moment_jacobian = false); push!(two, time() - t0)
    end
    println("  constructor wall time WITH jac_h    : median=", round(median(tw) * 1000, digits = 2),
            "ms  all=", round.(tw .* 1000, digits = 2))
    println("  constructor wall time WITHOUT jac_h : median=", round(median(two) * 1000, digits = 2),
            "ms  all=", round.(two .* 1000, digits = 2))
    println("  attributed jac_h alloc+zero cost    : ", round((median(tw) - median(two)) * 1000, digits = 2), " ms")

    # ---- warm evaluate_fullA timing, alternating order, n=10 each ----
    evaluate_fullA(xS, ctxS; cache = nothing, warm = true)   # pre-warm compile
    evaluate_fullA(xS, ctxS2; cache = nothing, warm = true)
    e1 = Float64[]; e2 = Float64[]
    for _ in 1:10
        s1 = @timed evaluate_fullA(xS, ctxS; cache = nothing, warm = true); push!(e1, s1.time)
        s2 = @timed evaluate_fullA(xS, ctxS2; cache = nothing, warm = true); push!(e2, s2.time)
    end
    println("  warm evaluate_fullA median wall: WITH jac_h=", round(median(e1) * 1000, digits = 3),
            "ms  WITHOUT=", round(median(e2) * 1000, digits = 3), "ms  ratio=",
            round(median(e1) / median(e2), digits = 3))
end

println("\nD=6/8 MICROBENCHMARK COMPLETE.")
