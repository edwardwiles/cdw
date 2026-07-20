# ============================================================================
# Integration-branch validation suite for fast_range_screen.jl (Section 6 of
# the production integration brief). Real D=20/W=80,000 data.
#
# Loads the catalogue of previously-recovered points under
# diagnostics/infeasibility_points/ (ported verbatim, pure data, from
# diag/fullA-d20-range-screen-review -- attributed, not new investigation
# work) and re-validates every one of them against THIS branch's fused
# screens: evaluate_fullA_screened (existing, unmodified baseline) vs
# evaluate_fullA_screened_ranged (this branch's new integration).
#
# Usage: julia --project=. full_aod_diag/d4_exact/validate_fast_range_screen_d20.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using DelimitedFiles, Random, Printf

const POINTS_DIR = joinpath(@__DIR__, "..", "..", "diagnostics", "infeasibility_points")

"""
Minimal, dependency-free extractor for the handful of scalar/string fields this
script needs from the catalogue's flat (non-nested) JSON files -- avoids adding
a JSON.jl dependency to the production Project.toml for a validation-only script.
"""
function json_field(text::AbstractString, key::AbstractString)
    m = match(Regex("\"" * key * "\"\\s*:\\s*\"([^\"]*)\""), text)   # string value
    m !== nothing && return m.captures[1]
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+]+)"), text)  # numeric value
    m !== nothing && return m.captures[1]
    return nothing
end

"Build a full theta_full vector for a catalogue point from its JSON metadata + Aod CSV, given a live ctx."
function load_point_theta(ctx, jsonpath::AbstractString)
    text = read(jsonpath, String)
    D = parse(Int, json_field(text, "D"))
    @assert D == ctx.D "point D=$D != ctx.D=$(ctx.D)"
    gp_str = json_field(text, "gp_focal")
    gp_str === nothing && (gp_str = json_field(text, "gamma_prime_focal_at_checkpoint"))
    gp_str === nothing && error("no gp_focal / gamma_prime_focal_at_checkpoint field found")
    gp_focal = parse(Float64, gp_str)
    csvfield = json_field(text, "Aod_theta_full_csv")
    csvpath = isabspath(csvfield) ? csvfield : joinpath(POINTS_DIR, csvfield)
    isfile(csvpath) || (csvpath = joinpath(POINTS_DIR, basename(csvfield)))
    Aod = readdlm(csvpath, ',', Float64)
    @assert size(Aod) == (D, D) "Aod CSV shape mismatch: $(size(Aod)) vs ($D,$D)"

    θ_full = copy(ctx.θ0_up)
    θ_full[3+D] = gp_focal
    # CSV rows/cols: written by writedlm(path, Aod, ',') from a D x D Julia matrix (row-major text,
    # column-major memory) -- reshape convention in this file's own theta_full layout is
    # reshape(theta_full[Aod_offset+1:Aod_offset+D^2], (D,D)) i.e. COLUMN-major fill. readdlm gives
    # us the D x D matrix directly (row i, col j = Aod[i,j] as originally written), so we just need
    # to flatten it the SAME way reshape(...,(D,D)) would expect: vec(Aod) is column-major, matching.
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(Aod)
    return θ_full, nothing
end

function x_free_from_theta(θ_full, ctx)
    return θ_full[ctx.free_idx]
end

function main()
    W = 80000
    println("Building ctx at W=$W ..."); flush(stdout)
    t0 = time()
    Random.seed!(20260719)
    ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true)
    println("ctx built in ", round(time() - t0, digits = 1), "s"); flush(stdout)

    t0 = time()
    rsc = build_ranged_screen_context(ctx)
    println("build_ranged_screen_context: ", round(time() - t0, digits = 3), "s, envelope supported = ",
            rsc.envelope !== nothing, rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason))" : ""); flush(stdout)

    files = filter(f -> endswith(f, ".json"), readdir(POINTS_DIR))
    results = NamedTuple[]

    for f in sort(files)
        jsonpath = joinpath(POINTS_DIR, f)
        text = read(jsonpath, String)
        json_field(text, "Aod_theta_full_csv") === nothing && (println("skip (no Aod csv): ", f); continue)
        local θ_full
        try
            θ_full, _ = load_point_theta(ctx, jsonpath)
        catch e
            println("skip (load error): ", f, " -- ", sprint(showerror, e))
            continue
        end
        xfree = x_free_from_theta(θ_full, ctx)

        t_base0 = time()
        res_base, meta_base = evaluate_fullA_screened(xfree, ctx; moment_representation = :compressed, warm = false, use_cache = false)
        t_base = time() - t_base0

        t_int0 = time()
        res_int, meta_int = evaluate_fullA_screened_ranged(xfree, ctx, rsc; moment_representation = :compressed, warm = false, use_cache = false)
        t_int = time() - t_int0

        agree = (res_base.inner_status == res_int.inner_status) ||
                (res_base.inner_status < -8999 && res_int.inner_status < -8999)  # both "some exact-infeasible sentinel"
        same_value = (res_base.inner_status in (0, -100, -101, -103) && res_int.inner_status in (0, -100, -101, -103)) ?
                      isapprox(res_base.Delta_dual, res_int.Delta_dual; atol = 1e-8, rtol = 1e-8) : missing

        push!(results, (file = f, base_status = res_base.inner_status, base_screen = meta_base.screen_status,
                         int_status = res_int.inner_status, int_screen = meta_int.screen_status,
                         t_base = t_base, t_int = t_int, agree = agree, same_value = same_value,
                         base_Delta = res_base.Delta_dual, int_Delta = res_int.Delta_dual))
        @printf("%-55s base=%-30s (%.3fs)  integrated=%-32s (%.3fs)  agree=%s  Δmatch=%s\n",
                f, string(meta_base.screen_status), t_base, string(meta_int.screen_status), t_int, agree, same_value)
        flush(stdout)
    end

    n_disagree = count(r -> !r.agree, results)
    n_value_mismatch = count(r -> r.same_value === false, results)
    println("\n=== SUMMARY ===")
    println("n_points = ", length(results))
    println("n_disagree (base vs integrated feasibility verdict) = ", n_disagree)
    println("n_value_mismatch (both feasible but Delta_dual differs) = ", n_value_mismatch)
    if !isempty(results)
        println("mean t_base  = ", round(sum(r.t_base for r in results) / length(results), digits = 4), "s")
        println("mean t_int   = ", round(sum(r.t_int for r in results) / length(results), digits = 4), "s")
    end

    return results
end

results = main()
