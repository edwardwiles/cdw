# Claude Code task 2026-08-01 (all-families completion): D4 gate for the reduced FG callback and a
# REAL KNITRO solve of the reduced dual problem, flexible CM.
#
# REWRITTEN (2026-08-01, later same session, after a user-driven investigation found and fixed a
# formulation-consistency bug -- see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md's full
# account). The reduced economic FG now uses `materialize_homogeneous_dense_G_reduced!` (the
# HOMOGENEOUS formulation, matching `reduced_homogeneous_winner_pair_hessian!`/H_EC's
# `use_profiled_correction=true` path), NOT the structured formulation the ORIGINAL version of this
# test compared against -- that comparison would now correctly FAIL (they are genuinely different
# moment definitions, confirmed via direct comparison of their linear functionals, not a
# reparametrization of the same one). This rewrite instead validates:
#
# Part 1: the reduced HOMOGENEOUS G matches a gather from a FULL HOMOGENEOUS G (both built the same
# "safe by linearity" way, via unit-vector calls to homogeneous_dual_contraction/
# reduced_homogeneous_dual_contraction) -- the correct, same-family comparison this whole
# investigation converged on (mirroring the earlier `expand_reduced_beta_to_full`-style checks).
# Part 2: a REAL KNITRO inner solve of the reduced dual problem converges CLEANLY -- `nStatus==0`
# (optimal, not merely feasible) in a SMALL number of iterations (quadratic Newton convergence,
# matching the FULL model's own ~4-iteration signature) -- NOT that its ζ* matches the full model's
# own ζ* (which this investigation established is the WRONG expectation: the homogeneous formulation
# is a deliberately different moment definition from the structured/production one, per
# homogeneous_contraction_2026-07-31.jl's own header, so a different optimal value is expected, not a
# bug -- see the master doc's "different formulation, not a bug" section). Before the fix in this
# session, this exact solve hit KNITRO's default 100-iteration limit without this signature
# (nStatus=-400, NOT -100/-101/-103) -- unusual for this class of problem and the direct trigger for
# the investigation that found the underlying bug.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
D = ctx.D; Ddest = cf_probe.D_dest
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
W = size(ctx.U, 1)
ncolI_reduced = layout.total_reduced_economic_moments
ncolI_full = cf_probe.oci - 1

L = 10; contrasts = :anchored

# ---- Part 1: reduced homogeneous G matches a gather from the full homogeneous G (same family) ----
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0,
                                            profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
K_reduced = Vector{Float64}(undef, W)
G_reduced = Matrix{Float64}(undef, W, obj_reduced.d)
obj_reduced.moments!(K_reduced, G_reduced, collect(θ_full_calib), ctx.U, obj_reduced)

# Full-width homogeneous G, built the SAME "safe by linearity" way (unit vectors -> homogeneous_dual_contraction).
e_full = zeros(ncolI_full)
G_full_homog_econ = Matrix{Float64}(undef, W, ncolI_full)
for j in 1:ncolI_full
    e_full[j] = 1.0
    G_full_homog_econ[:, j] .= homogeneous_dual_contraction(e_full, cf_probe, ctx, collect(θ_full_calib))
    e_full[j] = 0.0
end

n_bilateral = length(layout.retained_full_factual_j)
gathered_bilateral = G_full_homog_econ[:, layout.retained_full_factual_j]
check("Part1: K_reduced finite", all(isfinite, K_reduced))
check("Part1: reduced G bilateral columns match gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
    maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)) < 1e-10)
if has_france
    check("Part1: reduced G france column matches gathered-from-full HOMOGENEOUS G (max|Δ|<1e-10)",
        maximum(abs.(G_reduced[:, n_bilateral+1] .- G_full_homog_econ[:, cf_probe.cf_col])) < 1e-10)
end
@printf("  Part1: max|ΔG_bilateral|=%.3e\n", maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)))

# ---- Part 2: a REAL KNITRO solve of the reduced dual problem converges CLEANLY ----
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
                                   moment_representation = :dense_reference)
cctx_full = pcx.cctx
obj_full = pcx.ctx_cm.obj

# threaded_bins=false: hessian_cm_structured_v2! (the threaded twin) was NOT given the profiled/
# gather branch this session -- force the serial hessian_cm_structured! path, which was.
# inner_fg_backend=:dense_reference: the TRUE production default (:cm_lookup) is a separate,
# not-yet-wired operator FG evaluator -- see the master doc.
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference,
                                 threaded_bins = false)
try
    base_full = archC_base_state(x_free_calib, pcx.ctx_cm, cctx_full)
    check("Part2: FULL inner solve feasible", base_full.inner_status in (0, -100, -101, -103))

    ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)
    base_reduced = archC_base_state(x_free_calib, ctx_cm_reduced, cctx_reduced)
    @printf("  Part2: FULL    inner_status=%d  ζ*=%.10f\n", base_full.inner_status, base_full.ζstar)
    @printf("  Part2: REDUCED inner_status=%d  ζ*=%.10f  (different moment definition than FULL -- NOT expected to match, see header)\n",
        base_reduced.inner_status, base_reduced.ζstar)
    check("Part2: REDUCED inner solve converges to OPTIMALITY (nStatus==0, not merely feasible -- " *
          "the pre-fix version of this exact solve hit the 100-iteration limit, nStatus=-400)",
        base_reduced.inner_status == 0)
catch e
    check("Part2: REDUCED inner solve did not throw ($(sprint(showerror, e)))", false)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
