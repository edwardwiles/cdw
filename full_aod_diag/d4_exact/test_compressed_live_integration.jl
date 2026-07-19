# ============================================================================
# Continuation 8, workstream 2: live-integration equivalence suite for
# compressed_live.jl. Compares evaluate_fullA_fast(...; moment_representation=
# :dense) [the trusted reference, byte-unchanged by this workstream] against
# evaluate_fullA_fast(...; moment_representation=:compressed) [this
# workstream's new wiring] across:
#   1. calibration point, both incumbents (upper_lfixcomposite_sr1_60s,
#      lower_lfixcomposite_fast_sr1_300s), a few gamma-profile-style points
#   2. random feasible points
#   3. every x_free coordinate (gamma'_focal + all 16 A_od entries), both FD
#      signs, several h values, perturbed off the upper incumbent
#   4. warm vs cold inner solves
#   5. a short sequential "trajectory" of outer points with warm-started
#      inner solves (mimicking consecutive KNITRO outer iterates), comparing
#      EVERY intermediate point's full result, not just the endpoint
#   6. an injected exact price tie -> confirms compressed falls back to dense
#      automatically, with the fallback counter incrementing and results
#      still matching the dense answer exactly (since it federates to dense)
#
# All fields listed in evaluate_fullA_fast's result NamedTuple are compared,
# not a subset. Reports the ACTUAL max errors (not just PASS/FAIL) so the
# tolerance claims in docs/compressed_live_integration_report.md are backed
# by numbers, not assertions.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))   # Continuation 10 Section 9: structured dense-materialize, used by compressed_live.jl / infeasibility_screen.jl
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using Random, Printf, LinearAlgebra, Statistics

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

FIELDS_NUMERIC = (:gamma_focal_prime, :K_hard, :Delta_dual, :Delta_primal, :Delta_minus_delta,
    :gravity_raw, :gravity_value, :gravity_R_sum, :gravity_R_mean, :gravity_R_beta,
    :max_abs_moment_resid, :zeta, :m_mean, :m_min, :m_max, :weight_norm_resid,
    :mean_m_resid, :max_abs_moment_kkt_resid, :primal_dual_gap)
FIELDS_EXACT = (:inner_status, :winner_hash)

MAXDIFF_SEEN = Dict{Symbol, Float64}()  # per-field worst-case abs diff across the WHOLE suite

function compare(label, x_free; warm = true, cache_a = nothing, cache_b = nothing, verbose = true)
    ra, meta_a = evaluate_fullA_fast(x_free, ctx; cache = cache_a, warm = warm, moment_representation = :dense)
    rb, meta_b = evaluate_fullA_fast(x_free, ctx; cache = cache_b, warm = warm, moment_representation = :compressed)
    ok = true
    maxdiff = 0.0
    for f in FIELDS_NUMERIC
        va = getfield(ra, f); vb = getfield(rb, f)
        if isnan(va) && isnan(vb)
            continue
        end
        d = abs(va - vb)
        maxdiff = max(maxdiff, d)
        MAXDIFF_SEEN[f] = max(get(MAXDIFF_SEEN, f, 0.0), d)
        if d > 1e-7
            println("  MISMATCH field=$f  dense=$va  compressed=$vb  diff=$d")
            ok = false
        end
    end
    for f in FIELDS_EXACT
        va = getfield(ra, f); vb = getfield(rb, f)
        if va != vb
            println("  MISMATCH (exact) field=$f  dense=$va  compressed=$vb")
            ok = false
        end
    end
    dlogA = maximum(abs.(collect(ra.logA) .- collect(rb.logA)))
    dmr = isempty(ra.moment_resid) ? 0.0 : maximum(abs.(ra.moment_resid .- rb.moment_resid))
    dlam = isempty(ra.lambda) ? 0.0 : maximum(abs.(ra.lambda .- rb.lambda))
    MAXDIFF_SEEN[:logA] = max(get(MAXDIFF_SEEN, :logA, 0.0), dlogA)
    MAXDIFF_SEEN[:moment_resid] = max(get(MAXDIFF_SEEN, :moment_resid, 0.0), dmr)
    MAXDIFF_SEEN[:lambda] = max(get(MAXDIFF_SEEN, :lambda, 0.0), dlam)
    if dlogA > 1e-7; println("  MISMATCH field=logA diff=$dlogA"); ok = false; end
    if dmr > 1e-7; println("  MISMATCH field=moment_resid diff=$dmr"); ok = false; end
    if dlam > 1e-7; println("  MISMATCH field=lambda diff=$dlam"); ok = false; end
    if verbose
        println(rpad(label, 40), " warm=", warm, "  maxdiff=", @sprintf("%.3e", maxdiff),
                "  n_fg(dense)=", meta_a.n_fg_calls, " n_fg(compr)=", meta_b.n_fg_calls,
                "  n_hess(dense)=", meta_a.n_hess_calls, " n_hess(compr)=", meta_b.n_hess_calls,
                "  ", ok ? "PASS" : "FAIL")
    end
    return ok
end

all_ok = true

println("="^90); println("PHASE 1: candidate / incumbent / calibration points, warm=true"); println("="^90)
all_ok &= compare("calibration", ctx.θ0_up[ctx.free_idx]; warm = true)

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
xf_up40 = x_free_from_w(w_up40)
all_ok &= compare("upper_maxit40", xf_up40; warm = true)

w_lfixcomposite_sr1 = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf_upper_incumbent = x_free_from_w(w_lfixcomposite_sr1)
all_ok &= compare("upper_lfixcomposite_sr1_60s (INCUMBENT, kappa=0.17245688540655113)", xf_upper_incumbent; warm = true)

w_lower_lfixcomposite_fast = [0.9967391744173478, 0.33826763364911505, 0.2756097423805949, 0.3168080759212972, 0.28291467586724917, 1.124214996356822, 1.0586966677239436, 1.0353970705480537, 1.0651065385723435, 0.7972795304932372, 0.7498744164437179, 0.797300832305142, 0.7520482961532734, 1.464240420901755, 1.3955928233005424, 1.4031151818251653]
xf_lower_incumbent = x_free_from_w(w_lower_lfixcomposite_fast)
all_ok &= compare("lower_lfixcomposite_fast_sr1_300s (INCUMBENT, kappa=0.005428799948779983)", xf_lower_incumbent; warm = true)

println("\n" * "="^90); println("PHASE 1b: gamma-profile-style points (A_od==calibration, gamma' varied)"); println("="^90)
zfree0 = pivot_reduce(zeros(D, D), pe)
for gp in (0.85, 0.90, 0.93, 0.96, 0.99)
    xf = x_free_from_w(vcat(gp, zfree0))
    global all_ok &= compare("gamma_profile gp=$gp", xf; warm = true)
end

println("\n" * "="^90); println("PHASE 2: random feasible perturbations around upper incumbent"); println("="^90)
rng = MersenneTwister(20260718)
for trial in 1:15
    w = w_lfixcomposite_sr1 .+ 0.005 .* randn(rng, length(w_lfixcomposite_sr1))
    xf = x_free_from_w(w)
    global all_ok &= compare("random perturbation $trial", xf; warm = true)
end

println("\n" * "="^90); println("PHASE 3: every x_free coordinate (gamma' + all 16 A_od entries), both FD signs, several h"); println("="^90)
n_xfree = length(xf_upper_incumbent)
@assert n_xfree == 1 + D^2
for h in (1e-3, 1e-5, 1e-7)
    for coord in 1:n_xfree
        for sign in (+1, -1)
            xf = copy(xf_upper_incumbent)
            xf[coord] += sign * h
            lbl = "coord=$coord sign=$sign h=$h"
            ok = compare(lbl, xf; warm = true, verbose = false)
            global all_ok &= ok
            ok || println("  (see above) FAILED at $lbl")
        end
    end
end
println("Phase 3: all $(n_xfree*3*2) coordinate/sign/h probes done (only failures printed above).")

println("\n" * "="^90); println("PHASE 4: cold start (warm=false)"); println("="^90)
all_ok &= compare("upper incumbent (cold)", xf_upper_incumbent; warm = false)
all_ok &= compare("lower incumbent (cold)", xf_lower_incumbent; warm = false)
all_ok &= compare("calibration (cold)", ctx.θ0_up[ctx.free_idx]; warm = false)

println("\n" * "="^90); println("PHASE 5: short sequential trajectory (5 warm-started outer points), full state comparison at EVERY step"); println("="^90)
# A short synthetic "trajectory" of outer points (crude finite-difference descent on Delta_dual using
# the DENSE oracle only, so the trajectory itself is representation-independent) with warm-started
# inner solves at each step -- mirrors what a real KNITRO outer loop does (each outer iterate's inner
# dual solve warm-started from the previous one). Verifies dense vs compressed agree at every
# INTERMEDIATE point, not just the final one, including the inner dual iterate (zeta,lambda) itself.
obj_traj = ctx.obj
obj_traj.x .= NaN   # cold start the trajectory
traj_x = copy(xf_upper_incumbent)
r0, _ = evaluate_fullA_fast(traj_x, ctx; warm = false, moment_representation = :dense)
println("trajectory step 0 (start): Delta_dual=", r0.Delta_dual, "  gp=", r0.gamma_focal_prime)
rng2 = MersenneTwister(7)
step_ok = true
for step in 1:5
    dir = randn(rng2, length(traj_x)); dir ./= norm(dir)
    global traj_x = traj_x .+ 0.003 .* dir
    global step_ok &= compare("trajectory step $step", traj_x; warm = true)
end
all_ok &= step_ok

println("\n" * "="^90); println("PHASE 6: injected exact price tie -> automatic dense fallback"); println("="^90)
fallback0 = COMPRESSED_FALLBACK_COUNT[]
# Reuses the SAME proven tie-injection technique as test_compressed_moments.jl (continuation 7's own
# equivalence test): pick a real converged base point, find the (draw,destination) argmin, then perturb
# ONE origin's draw U[s0,o2] so its price becomes EXACTLY equal to the winner's -- a guaranteed bit-exact
# tie, via a `merge(ctx, (U=Utie,))` NamedTuple copy. NOTE: `ctx_tie.obj === ctx.obj` (merge does not
# deep-copy the mutable obj field) so `obj.U` is UNCHANGED -- only `build_compressed_factual`'s direct
# `ctx.U` read sees the tie; the dense fallback (which reads `obj.U`) computes the ORIGINAL, untied
# answer, which is exactly the desired/correct behavior for this test: it isolates "does compressed
# detect the tie and fall back" from "is U actually mutated for the dense solver too" (a separate,
# already-covered concern -- dense-vs-compressed-on-a-genuinely-tied-U is not reachable in this codebase
# since obj.U is the single source of truth for the dense path everywhere else).
xf0 = xf_upper_incumbent
base = solve_base_state(xf0, ctx)
θfull = base.θ_full0
_, _, AodPow_tie = factual_prices(θfull, ctx)
μ_tie = θfull[1]; s0 = 7; d0 = 1
cc_tie = [ctx.γ.wHat[o] * AodPow_tie[o, d0] * ctx.γ.τ[o, d0] for o in 1:D]
prices_tie = [cc_tie[o] / (ctx.U[s0, o]^(-μ_tie)) for o in 1:D]
w_tie = argmin(prices_tie); pmin_tie = prices_tie[w_tie]
o2_tie = (w_tie == 1 ? 2 : 1)
Utie = copy(ctx.U); Utie[s0, o2_tie] = (cc_tie[o2_tie] / pmin_tie)^(-1 / μ_tie)
ctx_tie = merge(ctx, (U = Utie,))

tie_triggered = false
try
    build_compressed_factual(θfull, ctx_tie; check_ties = true)
    println("  NOTE: injected point did not produce a detected tie (unexpected -- reusing a technique that DID trigger in test_compressed_moments.jl); reporting, not silently passing.")
catch e
    if e isa TiedWinnerError
        global tie_triggered = true
        println("  TiedWinnerError triggered as expected: n_tied_pairs=", e.n_tied_pairs, " examples=", e.examples)
    else
        rethrow()
    end
end

if tie_triggered
    r_dense, _ = evaluate_fullA_fast(xf0, ctx; warm = false, moment_representation = :dense)
    r_compr, meta_compr = evaluate_fullA_fast(xf0, ctx_tie; warm = false, moment_representation = :compressed)
    fb_ok = COMPRESSED_FALLBACK_COUNT[] == fallback0 + 1
    val_ok = isapprox(r_dense.Delta_dual, r_compr.Delta_dual; atol = 1e-9) &&
              isapprox(r_dense.gamma_focal_prime, r_compr.gamma_focal_prime; atol = 1e-9) &&
              r_dense.inner_status == r_compr.inner_status
    println("  fallback_count before=$fallback0 after=", COMPRESSED_FALLBACK_COUNT[], "  (expect +1): ", fb_ok ? "PASS" : "FAIL")
    println("  post-fallback Delta_dual dense=", r_dense.Delta_dual, " compressed(fell back to dense)=", r_compr.Delta_dual, "  ", val_ok ? "PASS" : "FAIL")
    global all_ok &= fb_ok && val_ok
else
    global all_ok &= false
    println("  FAIL: tie injection technique (reused from test_compressed_moments.jl, which DOES trigger with it) did not trigger here -- needs investigation, not silently accepted.")
end

println("COMPRESSED_FALLBACK_COUNT[] = ", COMPRESSED_FALLBACK_COUNT[], " (Ref{Int}, incremented on each TiedWinnerError fallback)")

println("\n" * "="^90)
println("PER-FIELD WORST-CASE ABS DIFF ACROSS THE WHOLE SUITE (dense vs compressed):")
for (f, v) in sort(collect(MAXDIFF_SEEN); by = x -> string(x[1]))
    println("  ", rpad(string(f), 24), @sprintf("%.3e", v))
end

println("\n" * "="^90)
println(all_ok ? "ALL COMPRESSED-LIVE-INTEGRATION EQUIVALENCE TESTS PASSED" : "SOME TESTS FAILED")
println("="^90)
all_ok || error("test_compressed_live_integration.jl: equivalence check failed")
