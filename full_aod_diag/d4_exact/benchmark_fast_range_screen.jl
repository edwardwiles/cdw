# Section 7 benchmark: warmed steady-state cost of the new screens, plus a
# W=800,000 build-cost check for the envelope precompute's genuine O(W) part
# (query cost is O(D^2), W-independent by construction).
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using DelimitedFiles, Random, Printf

const PD = joinpath(@__DIR__, "..", "..", "diagnostics", "infeasibility_points")
function jf(text, key)
    m = match(Regex("\"" * key * "\"\\s*:\\s*\"([^\"]*)\""), text); m !== nothing && return m.captures[1]
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+]+)"), text); m !== nothing && return m.captures[1]
    return nothing
end
function load_theta(ctx, jsonpath)
    text = read(jsonpath, String)
    D = parse(Int, jf(text, "D"))
    gp_str = jf(text, "gp_focal"); gp_str === nothing && (gp_str = jf(text, "gamma_prime_focal_at_checkpoint"))
    gp_focal = parse(Float64, gp_str)
    csvfield = jf(text, "Aod_theta_full_csv")
    csvpath = isabspath(csvfield) ? csvfield : joinpath(PD, csvfield)
    isfile(csvpath) || (csvpath = joinpath(PD, basename(csvfield)))
    Aod = readdlm(csvpath, ',', Float64)
    θ_full = copy(ctx.θ0_up); θ_full[3+D] = gp_focal
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(Aod)
    return θ_full
end

function main()
    println("=== W=80,000 warmed steady-state timing ===")
    Random.seed!(20260719)
    ctx = d20_real_setup(W = 80000, δ = 5.0, find_smallest = true)
    rsc = build_ranged_screen_context(ctx)

    θ_feasible = load_theta(ctx, joinpath(PD, "feasible_delta5_firstpass.json"))
    θ_infeas = load_theta(ctx, joinpath(PD, "nonzero_winner_infeasible_delta5_candidate1.json"))

    # warm JIT
    envelope_prewinner_screen(θ_feasible, ctx, rsc.envelope)
    envelope_prewinner_screen(θ_infeas, ctx, rsc.envelope)

    # precise timing: envelope screen alone, feasible point (INCONCLUSIVE path, full D^2 scan)
    n = 200
    t0 = time_ns()
    for _ in 1:n
        envelope_prewinner_screen(θ_feasible, ctx, rsc.envelope)
    end
    t_env_feasible = (time_ns() - t0) / 1e9 / n
    @printf("envelope_prewinner_screen (feasible pt, full scan, no hit): %.3f us/call\n", t_env_feasible * 1e6)

    t0 = time_ns()
    for _ in 1:n
        envelope_prewinner_screen(θ_infeas, ctx, rsc.envelope)
    end
    t_env_infeas = (time_ns() - t0) / 1e9 / n
    @printf("envelope_prewinner_screen (infeasible pt, early hit):       %.3f us/call\n", t_env_infeas * 1e6)

    # warm inner-solve baseline for comparison (single real solve, not looped -- CC solve is not idempotent-cheap to loop meaningfully without warm start reuse)
    xfree_feasible = θ_feasible[ctx.free_idx]
    evaluate_fullA_screened(xfree_feasible, ctx; moment_representation = :compressed, warm = false, use_cache = false)  # warm JIT
    t0 = time()
    res, meta = evaluate_fullA_screened(xfree_feasible, ctx; moment_representation = :compressed, warm = true, use_cache = false)
    t_full_call_base = time() - t0
    @printf("baseline evaluate_fullA_screened, feasible pt, warm-started full call: %.4f s\n", t_full_call_base)

    evaluate_fullA_screened_ranged(xfree_feasible, ctx, rsc; moment_representation = :compressed, warm = false, use_cache = false)  # warm JIT
    t0 = time()
    res2, meta2 = evaluate_fullA_screened_ranged(xfree_feasible, ctx, rsc; moment_representation = :compressed, warm = true, use_cache = false)
    t_full_call_int = time() - t0
    @printf("integrated evaluate_fullA_screened_ranged, feasible pt, warm-started full call: %.4f s\n", t_full_call_int)
    @printf("overhead of integrated vs baseline on a warm feasible call: %.2f%%\n", 100 * (t_full_call_int - t_full_call_base) / t_full_call_base)

    println("\n=== W=800,000 envelope precompute build-cost check (O(W) one-time part only) ===")
    Random.seed!(20260719)
    t0 = time()
    ctx2 = d20_real_setup(W = 800000, δ = 5.0, find_smallest = true, build_screen = false)
    t_ctx2 = time() - t0
    @printf("ctx build (W=800000, build_screen=false, no pairwise/witness): %.1f s\n", t_ctx2)

    t0 = time()
    ep2 = precompute_envelope(ctx2)
    t_env_build_800k = time() - t0
    @printf("precompute_envelope build @ W=800000: %.3f s\n", t_env_build_800k)

    θ_feasible2 = copy(ctx2.θ0_up)  # calibration point (theta0_up itself) is enough to time a query, doesn't need catalogue reload at this W
    envelope_prewinner_screen(θ_feasible2, ctx2, ep2)  # warm
    t0 = time_ns()
    for _ in 1:n
        envelope_prewinner_screen(θ_feasible2, ctx2, ep2)
    end
    t_env_800k = (time_ns() - t0) / 1e9 / n
    @printf("envelope_prewinner_screen query @ W=800000: %.3f us/call (compare to %.3f us/call @ W=80000 -- should be ~flat, O(D^2) only)\n",
            t_env_800k * 1e6, t_env_feasible * 1e6)
end
main()
