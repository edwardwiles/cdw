# ================================================================================================
# DIAGNOSTIC: how does Delta*(calibration point) depend on the restriction's SIZE, and on the draw
# design?  (CM + pairwise-quantile family #7, 2026-08-12)
#
# QUESTION BEING ANSWERED. The production configuration (L=5 PQ bins, G=50 CM grid, two CM moment
# families) gives Delta* ~ 0.029 at the calibration point, against an UNRESTRICTED delta* of order
# 1e-3 at the same data. Is that 0.029 the restriction genuinely biting, or an artefact? The direct
# way to find out is to SHRINK the restriction and watch: Delta* must fall monotonically toward the
# unrestricted value as rows are removed, because every row is an additional moment the LFD has to
# match. If it does not, something is wrong with the family rather than with the configuration.
#
# TWO AXES, both varied here:
#   * L  -- PQ bins on the shared reference marginal. Row count is (L-1) + (L-1)^2 * C(D,2), so it
#           grows QUADRATICALLY: 191 rows at L=2, 3044 at L=5, 15,390 at L=10.
#   * G  -- CM grid size. CM contributes n_families * (D-1) * (G-1) rows: 19 at G=2/1 family,
#           1862 at G=50/2 families. (`L` must divide `G`.)
#
# AND THE DRAW DESIGN. Every earlier D=20 number for this family was taken at
# `draw_design = :pseudorandom`, which is NOT the production design (`:sobol_randomized`) -- and
# under `:pseudorandom` the `draw_seed` is INERT (memory `feedback-draw-seed-inert-under-pseudorandom`:
# an inner seedU=888 overrides it), so those runs were not even seed-controlled. The production cell
# is therefore run under BOTH designs here so the difference is measured rather than assumed.
#
# ONE context is built per draw design and reused across every (L, G, families) cell -- the context
# depends on W/draws/gravity/sigma only, never on the restriction -- so the ~90 s real-data build is
# paid twice, not once per cell.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/diag_cmpq_delta_star_vs_L.jl <W>
# ================================================================================================

const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "incumbent_logic.jl", "cm_checkpoint.jl", "cm_originzc_target_layout.jl",
          "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl",
          "country_resolve.jl", "cross_delta_cache.jl", "compressed_moments.jl",
          "canonical_price_precompute_workspace.jl", "hard_score_b_cache.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "lfix_buffer_reuse.jl",
          "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl", "bandwidth_cache_policy.jl",
          "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl", "dual_bank_ab_harness.jl",
          "reusable_context.jl", "organic_failure_capture.jl", "knitro_status.jl",
          "knitro_version_check.jl", "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl",
          "cm_pairwise_quantile_cplus.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

const W_ARG = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const GRAV = default_gravity_exclude_cells_brazil_korea()
const OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")
lp(xs...) = (println(xs...); flush(stdout))

@printf("=== Delta* vs restriction SIZE and DRAW DESIGN, real D=20, W=%d, julia threads=%d ===\n",
        W_ARG, Threads.nthreads())
lp("production gravity mask (exclude_diagonal + Brazil-Korea cells), sigma=3.0, exclude_row,")
lp("calibration point, uniform masses mu = 1/L.")

"One (L, G, n_families) cell against an already-built context."
function cell(ctx, xf, L::Int, G::Int, nfam::Int)
    cfg = CMPairwiseQuantileConfig(L = L, cm_grid_size = G, cm_moment_families = nfam,
                                   contrasts = :orthonormal, min_bin_count = 1, mass_start = :uniform)
    tb = @elapsed cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = OPT)
    th = @elapsed ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx = true)
    ctx_cm = merge(ctx_cm, (cmpq_econ_ctx = ctx,))
    pcx = (ctx_cm = ctx_cm, cmpq = cmpq)
    mass0 = cmpq_uniform_mass_raw(L)
    ts = @elapsed b, v = archCMPQ_verified_state(xf, mass0, ctx_cm)
    n_x = cmpq.obj_cmpq.outer_constr_index
    return (L = L, G = G, nfam = nfam, n_restr = cmpq.n_restr, ncm = cmpq.ncm, n_x = n_x,
            D = v.Delta_dual, n_fg = v.n_fg, n_hess = v.n_hess, status = v.inner_status,
            cls = classify_inner_result(v), t_solve = ts, t_build = tb + th,
            kkt = v.max_abs_moment_kkt_resid)
end

# `L | G` throughout. Ordered smallest restriction first, so a monotone fall is visible by eye.
const CELLS = [(2, 2, 1), (2, 2, 2), (2, 10, 1), (2, 10, 2), (2, 50, 2),
               (5, 10, 1), (5, 10, 2), (5, 50, 1), (5, 50, 2)]

for design in (:sobol_randomized, :pseudorandom)
    lp("\n", "="^104)
    lp("DRAW DESIGN = :", design, design === :sobol_randomized ?
       "   (PRODUCTION)" : "   (what every earlier D=20 number for this family used; seed is INERT here)")
    lp("="^104)
    t0 = time()
    ctx_raw = d20_real_setup_design(W = W_ARG, δ = 1.0, find_smallest = true, draw_design = design,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0, inner_loop_opt = OPT)
    ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
    xf = ctx.θ0_up[ctx.free_idx]
    @printf("context: %.1fs   D=%d  obj.d=%d\n", time() - t0, ctx.D, ctx.obj.d); flush(stdout)

    @printf("\n%3s %3s %4s | %8s %6s %6s | %-16s | %4s %5s %7s | %s\n",
            "L", "G", "fam", "n_restr", "ncm", "n_x", "Delta*", "nfg", "nhess", "solve s", "class")
    lp("-"^104)
    for (L, G, nf) in CELLS
        r = try
            cell(ctx, xf, L, G, nf)
        catch e
            @printf("%3d %3d %4d | %s\n", L, G, nf, "FAILED: " * first(sprint(showerror, e), 70))
            flush(stdout); continue
        end
        @printf("%3d %3d %4d | %8d %6d %6d | %16.10g | %4d %5d %7.1f | %s\n",
                r.L, r.G, r.nfam, r.n_restr, r.ncm, r.n_x, r.D, r.n_fg, r.n_hess, r.t_solve,
                string(r.cls))
        flush(stdout)
    end
end

lp("\nHOW TO READ THIS. Delta* should fall MONOTONICALLY as rows are removed -- every restriction row")
lp("is one more moment the least-favourable distribution must match, so a smaller restriction cannot")
lp("give a larger divergence. The bottom-right cell (L=5, G=50, 2 families) is the production one.")
lp("Compare the two draw-design blocks cell by cell: any difference there is the Monte Carlo design,")
lp("not the restriction.")
