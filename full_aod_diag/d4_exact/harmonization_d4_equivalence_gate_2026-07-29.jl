# FG harmonization (2026-07-29) D=4 equivalence gate.
#
# Proves the refactored CMLookupState/CMFrechetLookupState FG callables (cm_lookup_kernels.jl,
# cm_frechet_lookup_kernels.jl -- now composed from shared economic_forward_into_arg0!/
# cm_forward_contribution!/economic_transpose_into_g1_and_gE!/cm_transpose_into_g! functions
# instead of two independently inlined copies) still reproduce the DENSE `obj(x,g)` reference
# callable's (f,g) to floating-point-accumulation-order tolerance, across:
#   - both families (flexible_cm via CMLookupState, common_frechet via CMFrechetLookupState)
#   - L in {10, 20, 50}
#   - both CM bases/contrasts (:anchored, :orthonormal)
#   - flexible_cm's own :interval AND :suffix/cumulative lookup methods (common_frechet always
#     uses the cumulative/:suffix basis -- see cm_frechet_lookup_kernels.jl's module docstring)
#   - a battery of deterministic-random dual vectors PLUS the real KNITRO-solver-derived point
#     (the converged inner dual at the calibration outer point)
#
# Writes FLEXCM_FRECHET_D4_EQUIVALENCE_GATE_2026-07-29.csv (one row per (family,L,contrasts,method,
# point) cell) to the repo root's docs/ output directory (passed as ARGS[1], default ".").
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

outdir = length(ARGS) >= 1 ? ARGS[1] : "."
csv_path = joinpath(outdir, "FLEXCM_FRECHET_D4_EQUIVALENCE_GATE_2026-07-29.csv")

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

rows = String[]
push!(rows, "family,L,contrasts,method,point,n_var,ferr_abs,frel,gerr_abs,grel,pass")

function random_x_battery(n_var::Int; scale = 1.0, n = 6, seed = 4242)
    rng = MersenneTwister(seed)
    return [scale .* randn(rng, n_var) for _ in 1:n]
end

function record!(rows, family, L, contrasts, method, label, fd, fl, gd, gl)
    ferr = abs(fd - fl)
    frel = ferr / max(1.0, abs(fd))
    gerr = maximum(abs.(gd .- gl))
    grel = gerr / max(1.0, maximum(abs.(gd)))
    ok = (ferr < 1e-8) && (grel < 1e-6)
    push!(rows, "$family,$L,$contrasts,$method,$label,$(length(gd)),$ferr,$frel,$gerr,$grel,$(ok ? "PASS" : "FAIL")")
    return ok
end

ALL_OK = Ref(true)

# ---- flexible_cm: CMLookupState vs dense, L in {10,20,50}, both contrasts, both methods ----
println("="^90); println("flexible_cm"); println("="^90); flush(stdout)
for L in (10, 20, 50), contrasts in (:anchored, :orthonormal)
    pcx_dense = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, moment_representation = :dense_reference)
    obj_dense = pcx_dense.ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free_calib, pcx_dense.ctx_cm.m)
    obj_dense.moments!(@view(obj_dense.H[:, 1]), CS.select_G_from_H(obj_dense, obj_dense.H), θ_full0, obj_dense.U, obj_dense)
    obj_dense.H[:, 2] .= 1.0

    cctx = pcx_dense.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    nvar = CS.inner_loop_number_variables(obj_dense)
    x0 = CS.inner_loop_initial_values(obj_dense)

    # solver-derived point: the real converged inner dual at the calibration outer point.
    Kd, xd_star, nsd, _, _ = inner_loop_internal_archgeneric(obj_dense, θ_full0; hess_cb_builder = _obj -> archC_hess_cb_builder(cctx))

    battery = vcat([("knitro_init", x0), ("solver_derived_calib", xd_star)],
                   [("random_$i", x0 .+ 0.02 .* v) for (i, v) in enumerate(random_x_battery(nvar))])

    # NOTE: only :suffix (cumulative basis) is exercised here, because `build_cm_production_context`
    # always builds the CUMULATIVE-basis dense reference (`use_archB_moments=true` default) -- the
    # SAME basis production's own `inner_loop_internal_cmlookup_production` uses (its own docstring:
    # "method defaults to :suffix ... NOT :interval ... comparing :interval lookup against a
    # cumulative-basis obj.H is exactly the category error"). :interval-vs-INTERVAL-basis-dense-
    # reference correctness is already covered by the pre-existing, still-passing
    # c12i_validate_lookup_fg.jl gate (which builds a separate interval-basis dense reference via
    # `build_cm_augmented_obj_interval` for that comparison) -- not duplicated here.
    method = :suffix
    st = CMLookupState(obj_dense, cctx.NCORE, cctx.ncm, L, cctx.origins, cctx.refIndex1, bins_u, cctx.R; method = method)
    for (label, xtest) in battery
        gd = zeros(nvar); gl = zeros(nvar)
        fd = obj_dense(xtest, gd)
        fl = st(xtest, gl)
        ok = record!(rows, "flexible_cm", L, contrasts, method, label, fd, fl, gd, gl)
        ALL_OK[] &= ok
    end
    @printf "  L=%d contrasts=%s: done\n" L contrasts; flush(stdout)
end

# ---- common_frechet: CMFrechetLookupState vs dense, L in {10,20,50}, both contrasts ----
println("="^90); println("common_frechet"); println("="^90); flush(stdout)
for L in (10, 20, 50), contrasts in (:anchored, :orthonormal)
    pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :dense_reference)
    obj_dense = pcx_dense.ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free_calib, pcx_dense.ctx_cm.m)
    obj_dense.moments!(@view(obj_dense.H[:, 1]), CS.select_G_from_H(obj_dense, obj_dense.H), θ_full0, obj_dense.U, obj_dense)
    obj_dense.H[:, 2] .= 1.0

    cctx = pcx_dense.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ncm_cm = cctx.ncm - cctx.L
    nvar = CS.inner_loop_number_variables(obj_dense)
    x0 = CS.inner_loop_initial_values(obj_dense)

    Kd, xd_star, nsd, _, _ = inner_loop_internal_archgeneric(obj_dense, θ_full0; hess_cb_builder = pcx_dense.hess_cb_builder)

    battery = vcat([("knitro_init", x0), ("solver_derived_calib", xd_star)],
                   [("random_$i", x0 .+ 0.02 .* v) for (i, v) in enumerate(random_x_battery(nvar))])

    st = CMFrechetLookupState(obj_dense, cctx.NCORE, ncm_cm, cctx.L, cctx.L, cctx.D,
        cctx.origins, cctx.refIndex1, bins_u, cctx.R, pcx_dense.aug.level_targets)
    for (label, xtest) in battery
        gd = zeros(nvar); gl = zeros(nvar)
        fd = obj_dense(xtest, gd)
        fl = st(xtest, gl)
        ok = record!(rows, "common_frechet", L, contrasts, "suffix", label, fd, fl, gd, gl)
        ALL_OK[] &= ok
    end
    @printf "  L=%d contrasts=%s: done\n" L contrasts; flush(stdout)
end

open(csv_path, "w") do io
    for r in rows
        println(io, r)
    end
end
println("Wrote ", csv_path, " (", length(rows) - 1, " data rows)")
println(ALL_OK[] ? "ALL D=4 EQUIVALENCE CHECKS PASS" : "SOME D=4 EQUIVALENCE CHECKS FAILED")
ALL_OK[] || exit(1)
