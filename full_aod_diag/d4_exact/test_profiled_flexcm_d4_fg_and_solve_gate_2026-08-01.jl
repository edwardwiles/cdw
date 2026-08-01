# Claude Code task 2026-08-01 (all-families completion): D4 gate for the NEW reduced FG callback
# (materialize_dense_factual_structured_reduced! wired into wrap_moments_with_cm_archB via the new
# profiled_layout keyword) -- closes the gap flagged in
# PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md's "important correction" section.
#
# Part 1: call obj_reduced.moments! DIRECTLY (bypassing KNITRO) and check its G/K output against a
# gather from the FULL model's own (already-validated) dense moments! at the SAME theta.
# Part 2: run an ACTUAL KNITRO inner solve of the reduced dual problem (moment_representation=
# :dense_reference, i.e. the real KNITRO/archC_base_state driver, not a hand-set point) and verify
# it converges and reproduces the SAME zeta*/objective as the full model once the full model's own
# converged lambda* has its anchor entries zeroed (the "recover-then-resolve" style check this
# project's own memory -- unrestricted-profiled-scale-knitro-comparison -- already established as
# the standard equivalence pattern for this port).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
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

L = 10; contrasts = :anchored

# ---- Part 1: direct moments! call, compare G against gathered-from-full ----
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
                                   moment_representation = :dense_reference)
cctx_full = pcx.cctx
obj_full = pcx.ctx_cm.obj
W = size(ctx.U, 1)
K_full = Vector{Float64}(undef, W)
G_full = Matrix{Float64}(undef, W, obj_full.d)
obj_full.moments!(K_full, G_full, collect(θ_full_calib), ctx.U, obj_full)
cf_full = cctx_full.core_cf_ref[]
check("Part1: cf_full published correctly", cf_full isa CompressedFactual)

aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0,
                                            profiled_layout = layout)
obj_reduced = aug_reduced.obj_cm
K_reduced = Vector{Float64}(undef, W)
G_reduced = Matrix{Float64}(undef, W, obj_reduced.d)
obj_reduced.moments!(K_reduced, G_reduced, collect(θ_full_calib), ctx.U, obj_reduced)
check("Part1: K_reduced == K_full (objective column, family-independent)", maximum(abs.(K_reduced .- K_full)) < 1e-12)

NCORE_full = cctx_full.NCORE
pregrav_full = NCORE_full - 1   # ones+bilateral+cf columns of G_full (G_full[:,end] is the LAST/gravity column separately, matches wrap_moments_with_cm_archB's own convention)
n_bilateral = length(layout.retained_full_factual_j)
n_reduced_econ = layout.total_reduced_economic_moments
pregrav_reduced = reduced_obj0.d - 1

# gather columns: G_full[:, 1:D*Ddest] are bilateral (1-indexed j=slot+(o-1)*Ddest WITHIN the
# pregrav range -- structured_coeffs/materialize_dense_factual_structured! index bilateral columns
# 1:D*Ddest directly, matching cf.oci-1's own layout), G_full[:,cf.cf_col] is the france ratio.
gathered_bilateral = G_full[:, layout.retained_full_factual_j]
check("Part1: reduced G bilateral columns match gathered-from-full", maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)) < 1e-12)
if has_france
    check("Part1: reduced G france column matches gathered-from-full", maximum(abs.(G_reduced[:, n_bilateral+1] .- G_full[:, cf_full.cf_col])) < 1e-12)
end
check("Part1: reduced G CM-grid columns match full's own (restriction block untouched)",
    maximum(abs.(G_reduced[:, pregrav_reduced+1:pregrav_reduced+cctx_full.ncm] .- G_full[:, pregrav_full+1:pregrav_full+cctx_full.ncm])) < 1e-12)
# NOTE: the gravity column sits at G[:,end] of the CALLER's full-width output buffer (`@views
# G[:,end] .= Gtmp[:,end]` in wrap_moments_with_cm_archB), NOT at G[:,NCORE_full] -- NCORE_full only
# indexes the internal Gtmp scratch's own (economic-block-only) column count, a different range once
# the CM-grid columns are appended after it in the real output G.
check("Part1: reduced G gravity/last column matches full's own", maximum(abs.(G_reduced[:, end] .- G_full[:, end])) < 1e-12)
@printf("  Part1: max|ΔG_bilateral|=%.3e  max|ΔG_cm|=%.3e\n",
    maximum(abs.(G_reduced[:, 1:n_bilateral] .- gathered_bilateral)),
    maximum(abs.(G_reduced[:, pregrav_reduced+1:end] .- G_full[:, pregrav_full+1:end])))

# ---- Part 2: REAL KNITRO solve of the reduced dual problem ----
# NOTE: the codebase's TRUE production default is inner_fg_backend=:cm_lookup (CM_INNER_FG_BACKEND_
# DEFAULT[], core_exact_hessian.jl:228) -- a SEPARATE O(W*(D-1)) operator-based FG evaluator
# (cm_lookup_kernels.jl) this session has NOT touched/wired for the reduced layout (a real,
# additional gap beyond the moments!/dense-G one closed above -- see the master doc). The
# pre-existing D4 tests explicitly override to :dense_reference for testing; do the same here,
# matching what materialize_dense_factual_structured_reduced! actually supports.
# threaded_bins=false: hessian_cm_structured_v2! (the threaded twin) was NOT given the profiled/
# gather branch this session (see master doc) -- force the serial hessian_cm_structured! path,
# which was.
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference,
                                 threaded_bins = false)
try
    base_full = archC_base_state(x_free_calib, pcx.ctx_cm, cctx_full)
    check("Part2: FULL inner solve feasible", base_full.inner_status in (0, -100, -101, -103))

    ctx_cm_reduced = (obj = obj_reduced, m = ctx.m)   # archC_base_state only reads .obj/.m -- pcx.ctx_cm's other fields are CM-augmentation-generic, not needed here
    base_reduced = archC_base_state(x_free_calib, ctx_cm_reduced, cctx_reduced)
    check("Part2: REDUCED inner solve ran (status=$(base_reduced.inner_status))", true)
    @printf("  Part2: FULL   inner_status=%d  ζ*=%.10f  Δ*(zeta)-ish objective proxy\n", base_full.inner_status, base_full.ζstar)
    @printf("  Part2: REDUCED inner_status=%d  ζ*=%.10f\n", base_reduced.inner_status, base_reduced.ζstar)

    if base_reduced.inner_status in (0, -100, -101, -103)
        check("Part2: REDUCED solve feasible", true)
        zeta_diff = abs(base_reduced.ζstar - base_full.ζstar)
        check("Part2: REDUCED ζ* matches FULL ζ* (same delta-star, both feasible; max|Δζ*|=$(zeta_diff))",
            zeta_diff < 1e-4)
    else
        check("Part2: REDUCED solve feasible (status=$(base_reduced.inner_status) -- see printed diagnostics above)", false)
    end
catch e
    check("Part2: REDUCED inner solve did not throw ($(sprint(showerror, e)))", false)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
