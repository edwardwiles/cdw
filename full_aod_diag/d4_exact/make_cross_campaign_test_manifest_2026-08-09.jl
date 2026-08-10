# Builds a minimal-but-VALID start manifest for `campaign_cm_family_runner.jl`, so the CROSS family
# arms can be exercised through the REAL campaign runner rather than only through the driver
# directly (2026-08-09). Matches the schema `three_starts_search.jl`/`common_five_starts_search.jl`
# emit and the runner actually parses: a `config` block recording the K the manifest was built at,
# `shared_extra_coordinates` sized to that K, and exactly 5 `starts` with `w_transformed_a` +
# `checksum_w_hash` (the runner re-verifies every checksum before running anything).
#
# The 5 starts are the calibration point plus 4 small deterministic perturbations of the ECONOMIC
# block only -- enough for a wiring exercise. This is NOT a substitute for a real
# `*_starts_search.jl` manifest (whose starts are qualified by actual feasibility screening); it
# exists so the campaign integration can be tested without a multi-hour search.
#
# nu seeding: theoretical population mean Gamma(1-mu*k) (E[z^k], z=U^(-mu), U~Exp(1)) -- NOT
# mean(U^k) and NOT a sample average of the same draws the restriction is imposed on.
#
# Usage: julia --project=. -t 4 .../make_cross_campaign_test_manifest_2026-08-09.jl <K> <W> <out.json>
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "nested_quantile_grids.jl",
          "cm_aspace_coordinate.jl", "country_resolve.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates
using SpecialFunctions: gamma

const KK   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const W    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000
const OUTF = length(ARGS) >= 3 ? ARGS[3] : joinpath(_D4E, "cross_campaign_test_manifest.json")

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D; Ddest = ctx.D_dest
pe = build_pivot_elimination(ctx)
theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
w_a_calib = cm_w0_from_calibration(ctx, pe, :powered_aspace)

# nu vectors, sized at THIS manifest's K -- the runner asserts these lengths against n_eta.
NU_MEANZC   = [gamma(1 - ctx.μHat * k) for k in 1:KK]                       # shared: length K_mean
NU_ORIGINZC = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:KK]...)     # per-origin: length K_mean*D

jvec(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"
jesc(s) = "\"" * replace(string(s), "\"" => "\\\"") * "\""

starts = NamedTuple[]
for i in 1:5
    w = copy(w_a_calib)
    if i > 1
        # deterministic, tiny, economic-block-only perturbation (no RNG -- reproducible by construction)
        for j in 2:length(w)
            w[j] += 1e-4 * sin(0.7 * i + 0.13 * j)
        end
    end
    push!(starts, (index = i, label = "cross_test_start_$i", w = w, h = string(hash(w), base = 16)))
end

open(OUTF, "w") do io
    println(io, "{")
    println(io, "  \"generated\": ", jesc(string(now())), ",")
    println(io, "  \"script\": \"full_aod_diag/d4_exact/make_cross_campaign_test_manifest_2026-08-09.jl\",")
    println(io, "  \"purpose\": \"wiring-exercise manifest for the CROSS campaign arms; starts are NOT feasibility-qualified\",")
    println(io, "  \"config\": {")
    println(io, "    \"W\": ", W, ", \"delta\": 1.0, \"find_smallest\": true,")
    println(io, "    \"draw_design\": \"sobol_randomized\", \"draw_seed\": 20260719,")
    println(io, "    \"destination_sample\": \"exclude_row\", \"D\": ", D, ", \"D_dest\": ", Ddest, ",")
    println(io, "    \"A_coordinate_mode\": \"powered_aspace\", \"cm_L\": 50, \"cm_contrasts\": \"orthonormal\",")
    println(io, "    \"meanzc_K_mean\": ", KK, ", \"meanzc_K_pair\": ", KK,
                ", \"originzc_K_mean\": ", KK, ", \"originzc_K_pair\": ", KK, ",")
    println(io, "    \"draw_checksum_uniform\": ", jesc(ctx.draw_meta.checksum_uniform), ",")
    println(io, "    \"draw_checksum_transformed\": ", jesc(ctx.draw_meta.checksum_transformed))
    println(io, "  },")
    println(io, "  \"shared_extra_coordinates\": {")
    println(io, "    \"cm_meanzc_nu\": ", jvec(NU_MEANZC), ",")
    println(io, "    \"origin_zc_nu\": ", jvec(NU_ORIGINZC))
    println(io, "  },")
    println(io, "  \"starts\": [")
    for (i, s) in enumerate(starts)
        println(io, "    {")
        println(io, "      \"index\": ", s.index, ", \"label\": ", jesc(s.label), ",")
        println(io, "      \"checksum_w_hash\": ", jesc(s.h), ",")
        println(io, "      \"gp\": ", @sprintf("%.17g", s.w[1]), ",")
        println(io, "      \"w_transformed_a\": ", jvec(s.w))
        print(io, "    }", i == length(starts) ? "\n" : ",\n")
    end
    println(io, "  ]")
    println(io, "}")
end
println("wrote ", OUTF)
println("  K=", KK, " W=", W, " D=", D, " |w_a|=", length(w_a_calib),
        " |nu_meanzc|=", length(NU_MEANZC), " |nu_originzc|=", length(NU_ORIGINZC))
println("  nu_meanzc = ", NU_MEANZC)
