# ================================================================================================
# BEFORE/AFTER Delta* at the real D=20 calibration point, OLD dyadic CM grid vs NEW equal-mass grid
# (2026-08-12 equal-mass-grid task, gate 3).
#
# WHAT THIS MEASURES AND WHY IT IS THE RIGHT COMPARISON.
# The change is to WHICH restriction is imposed, not to any solver/kernel/coordinate machinery. So
# the honest A/B is: hold the economic point, the draws, sigma, W, delta, the family definition and
# every solver option EXACTLY fixed, vary ONLY the probability grid, and report the inner dual
# optimum Delta*(theta_calib) each grid produces. That is a fixed-state value-only evaluation
# (`build_family` + `evaluate_family`, multistart_seed_generator.jl) -- the SAME production
# evaluator the seed qualifier and the outer driver's own verification path call -- with no outer
# KNITRO solve to introduce path dependence between the arms.
#
# Delta* IS EXPECTED TO MOVE. Both grids impose (D-1)*n_levels genuine CDF-contrast restrictions at
# the same point, but at different cutoffs and (49 vs 50) slightly different counts, so the dual
# optimum is a different number. What would be a RED FLAG is a solve that stops converging, or a
# Delta* that moves by orders of magnitude / into the unbounded regime (CLAUDE.md: Delta* is
# essentially bimodal -- small-and-finite, or diverging to the inner lower_limit floor).
#
# Arms are interleaved by grid within each family so that neither arm systematically eats JIT
# warm-up (CLAUDE.md: wall-clock comparisons on this box are worthless unless arms are interleaved).
# Wall-clock is reported for information only; the scientific output is Delta*.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. -t 10 \
#       full_aod_diag/d4_exact/diag_cm_equal_mass_grid_ab_2026-08-12.jl [W]
# ================================================================================================

const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "knitro_status.jl", "knitro_version_check.jl",
          "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(_D4E, f))
end

using Printf, Dates, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

# ---- paper_upper_v1 [scientific] block, verbatim (protocols/paper_upper_v1.toml) ----------------
const W        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const DELTA    = 1.0
const SIGMA    = 3.0
const SEED     = 20260719
const DESIGN   = :sobol_randomized
const LL       = -10.0
const DESTSAMP = :exclude_row

# ---- the two grids under comparison ------------------------------------------------------------
# OLD: exactly what `resolve_cm_probs(50)` returned before 2026-08-12 -- the dyadic largest-gap
#      bisection snapshot. 50 levels -> 51 buckets, masses only 0.015625 or 0.03125.
# NEW: `resolve_cm_probs(50)` today -- 50 equal-mass buckets, 49 cutpoints k/50, mass exactly 0.02.
const L_BUCKETS  = 50
const PROBS_OLD  = nested_grid_sequence([10, 20, 50])[L_BUCKETS]
const PROBS_NEW  = resolve_cm_probs(L_BUCKETS)

bucket_masses(p) = [p[1]; diff(p); 1 - p[end]]

lp("="^100)
lp("CM equal-mass grid A/B -- Delta* at the real D=20 calibration point")
lp("started ", Dates.now(), "  W=", W, "  delta=", DELTA, "  sigma=", SIGMA,
   "  draws=", DESIGN, "/", SEED, "  threads=", Threads.nthreads())
lp("="^100)
for (tag, p) in (("OLD dyadic ", PROBS_OLD), ("NEW equalmass", PROBS_NEW))
    m = bucket_masses(p)
    lp(@sprintf("  %s : %2d levels -> %2d buckets, masses %s", tag, length(p), length(m),
                string(sort(unique(round.(m, digits = 10))))))
end
lp()

const GRAV = default_gravity_exclude_cells_brazil_korea()
_ctx_raw = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = DESIGN,
    draw_seed = SEED, destination_sample = DESTSAMP, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = SIGMA, inner_lower_limit = LL)
const CTX = attach_compressed_factual_workspace(_ctx_raw, _ctx_raw.D, _ctx_raw.D_dest, _ctx_raw.W)
lp("ctx: D=", CTX.D, " D_dest=", CTX.D_dest, " W=", CTX.W, " bi=", CTX.bi, " sigma=", CTX.σ, " mu=", CTX.μHat)

# The calibration point itself, in the economic free-coordinate vector the evaluators take.
const X_FREE_CALIB = CTX.θ0_up[CTX.free_idx]
lp("x_free (calibration): n=", length(X_FREE_CALIB), " gp=", X_FREE_CALIB[1])
lp()

# ---- the three CM-grid-carrying paper families, each under both grids --------------------------
spec_for(family::Symbol, probs::Vector{Float64}) =
    family === :COMMON_MARGINALS ? cm_only_family_spec(:COMMON_MARGINALS; L = L_BUCKETS,
                                                       contrasts = :orthonormal, probs = probs) :
    family === :COMMON_FRECHET   ? common_frechet_family_spec(:COMMON_FRECHET; L = L_BUCKETS,
                                                       contrasts = :orthonormal, probs = probs) :
    family === :CM_PLUS_ZC       ? cm_zc_family_spec(:CM_PLUS_ZC; K_mean = 3, K_pair = 3, L = L_BUCKETS,
                                                       contrasts = :orthonormal, probs = probs) :
    error("spec_for: unknown family :$family")

const FAMILIES = (:COMMON_MARGINALS, :COMMON_FRECHET, :CM_PLUS_ZC)
results = Dict{Tuple{Symbol,Symbol},Any}()

eval_id = 0
for fam in FAMILIES
    lp("-"^100)
    lp("FAMILY ", fam)
    lp("-"^100)
    for (arm, probs) in ((:OLD, PROBS_OLD), (:NEW, PROBS_NEW))   # interleaved by grid within family
        global eval_id += 1
        spec = spec_for(fam, probs)
        lp(@sprintf("  [%s] grid=%s  n_levels=%d (spec.L=%d buckets)  cm_moments=(D-1)*n_levels=%d",
                    string(fam), string(arm), cm_n_levels(spec), spec.L, (CTX.D - 1) * cm_n_levels(spec)))
        t0 = time()
        res = try
            fb = build_family(CTX, spec)
            evaluate_family(CTX, fb, X_FREE_CALIB; eval_id = eval_id)
        catch e
            lp("    THREW: ", sprint(showerror, e))
            e
        end
        wall = time() - t0
        results[(fam, arm)] = (res = res, wall = wall)
        if res isa Exception
            lp(@sprintf("    -> EXCEPTION after %.1fs", wall))
        else
            lp(@sprintf("    -> Delta*=%.10g  verified=%s  inner_status=%d  class=%s  wall=%.1fs",
                        res.Delta_star, string(res.verified), res.inner_status, string(res.verification_class), wall))
        end
        flush(stdout)
    end
end

lp()
lp("="^100)
lp("SUMMARY -- Delta*(theta_calib), OLD dyadic grid vs NEW equal-mass grid")
lp("="^100)
lp(@sprintf("%-20s %-14s %-14s %-14s %-10s %-10s", "family", "Delta*_OLD", "Delta*_NEW", "ratio NEW/OLD", "ok_OLD", "ok_NEW"))
for fam in FAMILIES
    o = results[(fam, :OLD)].res
    n = results[(fam, :NEW)].res
    do_ = o isa Exception ? NaN : o.Delta_star
    dn  = n isa Exception ? NaN : n.Delta_star
    ok_o = o isa Exception ? "THREW" : string(o.verified)
    ok_n = n isa Exception ? "THREW" : string(n.verified)
    lp(@sprintf("%-20s %-14.8g %-14.8g %-14.6g %-10s %-10s", string(fam), do_, dn, dn / do_, ok_o, ok_n))
end
lp()
lp("finished ", Dates.now())
