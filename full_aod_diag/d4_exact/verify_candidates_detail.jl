# Section 6 follow-up: for the 5 nonzero-winner-infeasible candidates, confirm
# (a) baseline's real KNITRO call genuinely reports numerical infeasibility
#     (inner_status outside the FEASIBLE_CODES set, not a silent success), and
# (b) the new envelope certificate's reported bound/target/margin, cross-checked
#     against dense_recheck_certificate-style independent recomputation via the
#     general range_screen_standalone safety net operating on the SAME cf.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using DelimitedFiles, Random, Printf

const POINTS_DIR2 = joinpath(@__DIR__, "..", "..", "diagnostics", "infeasibility_points")

function json_field2(text::AbstractString, key::AbstractString)
    m = match(Regex("\"" * key * "\"\\s*:\\s*\"([^\"]*)\""), text)
    m !== nothing && return m.captures[1]
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+]+)"), text)
    m !== nothing && return m.captures[1]
    return nothing
end

function load_point_theta2(ctx, jsonpath::AbstractString)
    text = read(jsonpath, String)
    D = parse(Int, json_field2(text, "D"))
    gp_str = json_field2(text, "gp_focal")
    gp_str === nothing && (gp_str = json_field2(text, "gamma_prime_focal_at_checkpoint"))
    gp_focal = parse(Float64, gp_str)
    csvfield = json_field2(text, "Aod_theta_full_csv")
    csvpath = isabspath(csvfield) ? csvfield : joinpath(POINTS_DIR2, csvfield)
    isfile(csvpath) || (csvpath = joinpath(POINTS_DIR2, basename(csvfield)))
    Aod = readdlm(csvpath, ',', Float64)
    θ_full = copy(ctx.θ0_up)
    θ_full[3+D] = gp_focal
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(Aod)
    return θ_full
end

function main()
    Random.seed!(20260719)
    ctx = d20_real_setup(W = 80000, δ = 5.0, find_smallest = true)
    rsc = build_ranged_screen_context(ctx)
    println("envelope supported = ", rsc.envelope !== nothing)

    for i in 1:5
        f = "nonzero_winner_infeasible_delta5_candidate$(i).json"
        jsonpath = joinpath(POINTS_DIR2, f)
        θ_full = load_point_theta2(ctx, jsonpath)
        xfree = θ_full[ctx.free_idx]

        res_base, meta_base = evaluate_fullA_screened(xfree, ctx; moment_representation = :compressed, warm = false, use_cache = false)
        eres = envelope_prewinner_screen(θ_full, ctx, rsc.envelope)

        # independent cross-check: general range screen on the SAME winner/wval the fused screen would compute
        Pmat = target_shares(ctx)
        a = compute_a_od(θ_full, ctx)
        pc = ctx.pairwise
        pres = pairwise_certificate(a, pc, Pmat)
        order = order_destinations(pres, ctx.D)
        wres_ranged = screen_hard_winners_ranged(θ_full, ctx, Pmat, rsc.envelope; order = order, full_scan = true)
        cf = compressed_factual_from_screen(θ_full, ctx, WinnerScreenResult(true, ctx.D, 0, 0, collect(order), wres_ranged.winner, wres_ranged.wval, wres_ranged.win_counts))
        rres = range_screen_standalone(cf)

        @printf("%-45s base.inner_status=%-6d base.error=%-70s\n", f, res_base.inner_status, string(res_base.error_reason)[1:min(70,end)])
        @printf("   envelope: status=%s col=%d o=%d d=%d upper_bound=%.6f target=%.6f margin_norm=%.3e\n",
                eres.status, eres.column, eres.origin, eres.destination, eres.h_upper_bound, eres.target, eres.margin_normalized)
        if rres.certificate !== nothing
            c = rres.certificate
            @printf("   general range_screen_standalone (full-scan winner/wval): status=%s col=%d o=%d d=%d sign=%s min=%.6f max=%.6f margin_norm=%.3e\n",
                    rres.status, c.column, c.origin, c.destination, c.sign, c.min_val, c.max_val, c.margin_normalized)
        else
            println("   general range_screen_standalone: INCONCLUSIVE (unexpected -- investigate)")
        end
        println()
        flush(stdout)
    end
end
main()
