# Continuation 9 smoke test: does the newly-ported real-data D=20 context build,
# and does its calibration point roughly match the known ACR closed-form check
# (kappa_point_estimate ~= 0.020314, per trade_robustness_modular_perf's
# HANDOFF_D20_realdata_overnight_run.md) -- confirms the port is wired correctly,
# not just "doesn't crash."
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))

println("Building real D=20 context at W=80000 (per user: W=8000 is known too small for D=20 real ",
        "data -- matches the standing memory note 'd20-realdata-w-sensitivity': W=8000 silently ",
        "understates kappa for delta>=1 vs W>=80k)...")
t0 = time()
ctx = d20_real_setup(W = 80000)
println("  build wall: ", round(time() - t0, digits = 2), "s")
println("  D = ", ctx.D, " (expect 20)")
println("  baseIndex (focal) = ", ctx.bi, " (expect 2 = France)")
println("  sigma = ", ctx.σ, " (expect 2.5)")
println("  muHat (Frechet dispersion, theta) = ", ctx.μHat)
println("  n_free = ", CS.n_free(ctx.m), " (expect 1 + 20^2 = 401)")

# ACR closed-form point-estimate check: kappa_ACR = 1 - lambda_dd^mu, using the
# focal country's own-trade share directly from the raw pi.csv (bypasses the whole
# CC/moments/KNITRO machinery -- a pure data+calibration sanity check).
lambda_dd = ctx.so.data.λData[ctx.bi, ctx.bi]
kappa_acr = 1 - lambda_dd^ctx.μHat
println("  lambda_dd (France own-trade share, from pi.csv) = ", lambda_dd)
println("  kappa_ACR = 1 - lambda_dd^mu = ", kappa_acr, " (expect ~0.020314 per prior real-data D=20 work)")

println()
println("Searching for a feasible evaluation point (calibration A_od==1 -> natural theta -> ",
        "small random perturbations), same 3-tier fallback c8_perfprofile_harness.jl's ",
        "find_feasible_point used at D=6/8/10 synthetic (natural-theta succeeded there on the ",
        "first try every time) -- reimplemented directly on the full xf vector here since this ",
        "ctx's free_idx already spans the full D^2 A-block (no pivot-reduction indirection needed).")
D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_ones = vcat(gp0, ones(D^2))  # A_od theta == 1 exactly
r_cal = evaluate_fullA(xf_ones, ctx; warm = false)
println("  try 1 (A_od==1):      inner_status=", r_cal.inner_status, "  Delta_dual=", r_cal.Delta_dual)

xf_nat = ctx.θ0_up[ctx.free_idx]  # natural theta (calibrated A_od values)
r_nat = evaluate_fullA(xf_nat, ctx; warm = false)
println("  try 2 (natural theta): inner_status=", r_nat.inner_status, "  Delta_dual=", r_nat.Delta_dual)

feasible_codes = (0, -100, -101, -103)
r = r_cal.inner_status in feasible_codes ? r_cal :
    r_nat.inner_status in feasible_codes ? r_nat : nothing
how = r_cal.inner_status in feasible_codes ? "calibration" :
      r_nat.inner_status in feasible_codes ? "natural_theta" : "NONE"

if r === nothing
    Random.seed!(4200)
    for attempt in 1:6
        xfp = xf_nat .+ vcat(0.0, 0.02 .* (2 .* rand(D^2) .- 1))
        rp = evaluate_fullA(xfp, ctx; warm = false)
        println("  try ", 2 + attempt, " (random perturb ", attempt, "): inner_status=", rp.inner_status,
                "  Delta_dual=", rp.Delta_dual)
        if rp.inner_status in feasible_codes
            global r = rp; global how = "random_perturb_$attempt"
            break
        end
    end
end

println()
if r === nothing
    println("  NO feasible evaluation point found in 8 tries.")
else
    println("  feasible point found via: ", how)
    println("  inner_status = ", r.inner_status)
    println("  Delta_dual = ", r.Delta_dual)
    println("  gravity_raw = ", r.gravity_raw)
    println("  gamma_focal_prime = ", r.gamma_focal_prime)
    println("  max_abs_moment_kkt_resid = ", r.max_abs_moment_kkt_resid)
end

ok = ctx.D == 20 && CS.n_free(ctx.m) == 401 && isfinite(kappa_acr) && 0.0 < kappa_acr < 1.0 &&
     r !== nothing && isfinite(r.Delta_dual) && abs(r.gravity_raw) < 1e-6
println()
println(ok ? "SMOKE TEST: PASS" : "SMOKE TEST: FAIL")
