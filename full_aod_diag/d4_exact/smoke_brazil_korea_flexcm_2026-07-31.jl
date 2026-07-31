# Real-KNITRO smoke: flexible-CM family, real D=20 economy, Brazil->Korea excluded, through the
# REAL public checkpointed driver (run_cm_upper_checkpointed) -- not a test-script bypass.
# W=80,000 (documented production-safe scale; W=2000 has a known unrelated small-W conditioning
# issue per smoke_frechet_checkpointed_driver.jl's own comment). Short maxtime_real budget
# (matching this campaign's own preflight philosophy).
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_lookup_production.jl", "cm_frechet_lookup_kernels.jl",
          "cm_meanzc_lookup_production.jl", "cm_originzc_lookup_production.jl",
          "production_bundle_api.jl", "cm_checkpoint.jl", "country_resolve.jl"]
    include(joinpath(_D4E, f))
end
using DelimitedFiles
println("=== includes OK ==="); flush(stdout)

countries = vec(readdlm(joinpath(_D4E, "..", "..", "real_data", "noah_D20", "countries.csv"), ',', String))
bra_idx = resolve_country_index(countries, "Brazil")
kor_idx = resolve_country_index(countries, "Korea")
row_idx_g = findfirst(==("row"), countries)
named_dest = filter(!=(row_idx_g), 1:length(countries))
kor_slot = global_to_dest_slot(kor_idx, named_dest)
gravity_exclude_cells = [(bra_idx, kor_slot)]
println("gravity_exclude_cells = ", gravity_exclude_cells); flush(stdout)

W = 80000
ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
                       exclude_diagonal_gravity = true, gravity_exclude_cells = gravity_exclude_cells, σHat = 3.0)
pe0 = build_pivot_elimination(ctx0)
D0 = ctx0.D; Ddest0 = ctx0.D_dest
x_free_calib0 = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib0[1], pivot_reduce(log.(reshape(x_free_calib0[2:end], D0, Ddest0)), pe0))
println("ctx0 built, W=", W, " D=", D0, " D_dest=", Ddest0, " length(w0)=", length(w0)); flush(stdout)

ckpt_dir = mktempdir()
println("ckpt_dir = ", ckpt_dir); flush(stdout)

L = 10
probs = cm_equal_grid_probs(L)
result = run_cm_upper_checkpointed(w0; find_smallest = true, W = W, delta = 1.0,
    draw_design = :pseudorandom, draw_seed = 20260719, L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured,
    ckpt_dir = ckpt_dir, run_id = "brazil_korea_smoke", label = "flexcm_smoke",
    maxtime_real = 120.0, checkpoint_interval_s = 60.0, verbose = true,
    exclude_diagonal_gravity = true, gravity_exclude_cells = gravity_exclude_cells, σHat = 3.0)

println("=== RESULT ===")
println(result)
flush(stdout)
