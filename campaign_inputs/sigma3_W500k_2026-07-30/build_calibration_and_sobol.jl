# Phase B: rebuild calibration (A*/gp*/theta/pivot) at sigma=3 on the new immutable data
# snapshot, and generate the shared W=500,000 Sobol draw artifact for the sigma3/W500k
# five-family production campaign (2026-07-30).
#
# Usage: julia --project=. campaign_inputs/sigma3_W500k_2026-07-30/build_calibration_and_sobol.jl
# Run from the repo root. Uses REAL_DATA_DIR to point at the frozen snapshot (not the mutable
# real_data/noah_D20/ path), so this script reads from the immutable campaign inputs.

using Dates, SHA
ENV["REAL_DATA_DIR"] = joinpath(@__DIR__, "data_snapshot")

include(joinpath(pwd(), "full_aod_diag", "d4_exact", "context_real_d20.jl"))
include(joinpath(pwd(), "full_aod_diag", "d4_exact", "qmc_context_real_d20.jl"))
include(joinpath(pwd(), "full_aod_diag", "d4_exact", "qmc_draws.jl"))
include(joinpath(pwd(), "full_aod_diag", "d4_exact", "draw_design.jl"))
include(joinpath(pwd(), "full_aod_diag", "d4_exact", "gravity_elimination.jl"))
include(joinpath(pwd(), "full_aod_diag", "d4_exact", "campaign_cell_io.jl"))  # write_json_file

const W_CAMPAIGN = 500_000
const SOBOL_SEED = 20260719
const SIGMA = 3.0

println("="^78); println("STEP 1: build the base sigma=3 context at W=", W_CAMPAIGN, " via the REAL production entry point (d20_real_setup_design)"); println("="^78)
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_CAMPAIGN, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = SOBOL_SEED,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true, σHat = SIGMA,
    build_screen = true)
println("context build wall time: ", round(t_ctx, digits=1), "s")
println("ctx.D = ", ctx.D, "  ctx.D_dest = ", ctx.D_dest, "  ctx.σ = ", ctx.σ, "  ctx.W = ", ctx.W)
println("ctx.exclude_diagonal_gravity = ", ctx.exclude_diagonal_gravity)

theta_star = 1 / ctx.θ0_up[1]
println("theta* (via mu=theta0_up[1]) = ", theta_star)
@assert abs(theta_star - 4.7292535486122365) < 1e-9 "theta* drifted from the Phase B0-verified value -- investigate before freezing"

println("\n" * "="^78); println("STEP 2: machine-precision gravity equality at the calibration point"); println("="^78)
pe = build_pivot_elimination(ctx)
piv_o = ((pe.pivot_lin - 1) % ctx.D) + 1
piv_d = ((pe.pivot_lin - 1) ÷ ctx.D) + 1
println("pivot -> (o=$piv_o, d=$piv_d)  |c[pivot]|=", abs(pe.c[pe.pivot_lin]))
@assert piv_o != piv_d "pivot landed on a diagonal (own-trade) cell -- WRONG under exclude_diagonal_gravity"
z_free_zero = zeros(length(pe.other_idx))
z_full = pivot_expand(z_free_zero, pe)
g_at_calibration = gravity_from_logz(z_full, ctx)
println("g_gravity at calibration (z=0, pivot-solved) = ", g_at_calibration)
@assert abs(g_at_calibration) < 1e-9 "gravity NOT machine-precision zero at the calibration point"

println("\n" * "="^78); println("STEP 3: draw artifact checksums"); println("="^78)
println("draw_design = ", ctx.draw_design, "  draw_seed = ", ctx.draw_seed)
println("checksum_uniform     = ", ctx.draw_meta.checksum_uniform)
println("checksum_transformed = ", ctx.draw_meta.checksum_transformed)
println("sobol_jl_version = ", ctx.draw_meta.sobol_jl_version, "  julia_version = ", ctx.draw_meta.julia_version)
println("n_at_boundary = ", ctx.draw_meta.n_at_boundary, "  n_inf_transformed = ", ctx.draw_meta.n_inf_transformed)

println("\n" * "="^78); println("STEP 4: freeze calibration + Sobol manifest"); println("="^78)
production_sha = strip(read(`git rev-parse HEAD`, String))
data_manifest_path = joinpath(@__DIR__, "data_manifest.json")

manifest = Dict(
    "campaign" => "sigma3_W500k_five_family_2026-07-30",
    "generated_at" => string(now()),
    "production_sha" => production_sha,
    "sigma" => SIGMA,
    "W" => W_CAMPAIGN,
    "draw_design" => string(ctx.draw_design),
    "draw_seed" => ctx.draw_seed,
    "exclude_diagonal_gravity" => true,
    "destination_sample" => "exclude_row",
    "focal_country_index" => ctx.bi,
    "D" => ctx.D,
    "D_dest" => ctx.D_dest,
    "theta_star" => theta_star,
    "mu_star" => ctx.θ0_up[1],
    "pivot_linear_index" => pe.pivot_lin,
    "pivot_origin" => piv_o,
    "pivot_destination" => piv_d,
    "gravity_residual_at_calibration" => g_at_calibration,
    "gamma_prime_bounds" => Dict("kappa_min" => ctx.bounds.κ_min, "kappa_max" => ctx.bounds.κ_max,
                                  "gp_lo" => ctx.bounds.γp_lo, "gp_hi" => ctx.bounds.γp_hi),
    "theta0_up_checksum" => bytes2hex(sha256(join(string.(ctx.θ0_up), ","))),
    "draw_checksum_uniform" => ctx.draw_meta.checksum_uniform,
    "draw_checksum_transformed" => ctx.draw_meta.checksum_transformed,
    "sobol_jl_version" => string(ctx.draw_meta.sobol_jl_version),
    "julia_version" => string(ctx.draw_meta.julia_version),
    "context_build_wall_s" => t_ctx,
    "context_fingerprint" => context_fingerprint(ctx),
)
write_json_file(joinpath(@__DIR__, "calibration_manifest.json"), manifest)
println("Wrote ", joinpath(@__DIR__, "calibration_manifest.json"))
println("\nALL PHASE B (base context) CHECKS PASS")
