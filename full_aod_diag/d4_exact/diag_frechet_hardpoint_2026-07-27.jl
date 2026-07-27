# Ad-hoc diagnostic (2026-07-27): D=20 winner-bin H_ER wiring gate FAILED at the "hard point"
# (x_free0 .* 1.01) with complete-Hessian max|Delta|~5990 at Hessian scale ~1590, while the
# isolated H_E,level slice only differs by ~5.75 -- most of the error is NOT in the new level-block
# math, so this isolates whether it's H_EC (Part A reuse) or something structural (e.g. a
# tied-winner/dense-fallback state mismatch specific to this extreme point).
const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

println("Building real D=20 context..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 50
contrasts = :anchored

pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
    cm_hessian_backend = :structured, cm_cross_hessian_backend = :dense_reference)
pcx_wbin = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
    cm_hessian_backend = :structured, cm_cross_hessian_backend = :winner_bin)

x_free_hard = x_free_calib .* 1.01
println("Solving hard point at dense backend..."); flush(stdout)
base_hard = archC_frechet_base_state(x_free_hard, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
println("hard point status=", base_hard.inner_status)
x = vcat(base_hard.ζstar, base_hard.λstar)

obj_d = pcx_dense.ctx_cm.obj; obj_w = pcx_wbin.ctx_cm.obj
_archC_prep_for_hessian!(obj_d, x); _archC_prep_for_hessian!(obj_w, x)

cf_d = pcx_dense.cctx.core_cf_ref[]
cf_w = pcx_wbin.cctx.core_cf_ref[]
println("cf_d isa CompressedFactual: ", cf_d isa CompressedFactual, "  (else: ", cf_d isa CompressedFactual ? "n/a" : cf_d, ")")
println("cf_w isa CompressedFactual: ", cf_w isa CompressedFactual, "  (else: ", cf_w isa CompressedFactual ? "n/a" : cf_w, ")")

NCORE = pcx_dense.cctx.NCORE; ncm = pcx_dense.cctx.ncm; n = NCORE + ncm
ncm_cm = pcx_dense.cctx.nO * L
level_off = NCORE + ncm_cm

hd = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hw = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hessian_cm_frechet_structured!(hd, obj_d, pcx_dense.cctx, pcx_dense.aug.level_targets)
Hd_full = copy(pcx_dense.cctx.Hfull)
hessian_cm_frechet_structured!(hw, obj_w, pcx_wbin.cctx, pcx_wbin.aug.level_targets)
Hw_full = copy(pcx_wbin.cctx.Hfull)

println()
println("use_winner_bin check (pcx_wbin side): cm_cross_hessian_backend=", pcx_wbin.cctx.cm_cross_hessian_backend,
    "  ncore_core=", pcx_wbin.cctx.ncore_core, "  NCORE=", pcx_wbin.cctx.NCORE,
    "  core_ws!==nothing=", pcx_wbin.cctx.core_ws !== nothing,
    "  core_ws_for===cf_w=", pcx_wbin.cctx.core_ws_for === cf_w)

# block-by-block comparison
HEE_d = Hd_full[1:NCORE, 1:NCORE]; HEE_w = Hw_full[1:NCORE, 1:NCORE]
println("max|Delta H_EE| = ", maximum(abs.(HEE_d .- HEE_w)))

HEC_d = Hd_full[1:NCORE, NCORE+1:NCORE+ncm_cm]; HEC_w = Hw_full[1:NCORE, NCORE+1:NCORE+ncm_cm]
println("max|Delta H_EC| = ", maximum(abs.(HEC_d .- HEC_w)))

HCC_d = Hd_full[NCORE+1:NCORE+ncm_cm, NCORE+1:NCORE+ncm_cm]; HCC_w = Hw_full[NCORE+1:NCORE+ncm_cm, NCORE+1:NCORE+ncm_cm]
println("max|Delta H_CC| = ", maximum(abs.(HCC_d .- HCC_w)))

HElevel_d = Hd_full[1:NCORE, level_off+1:level_off+L]; HElevel_w = Hw_full[1:NCORE, level_off+1:level_off+L]
println("max|Delta H_E,level| = ", maximum(abs.(HElevel_d .- HElevel_w)))

HCMlevel_d = Hd_full[NCORE+1:NCORE+ncm_cm, level_off+1:level_off+L]; HCMlevel_w = Hw_full[NCORE+1:NCORE+ncm_cm, level_off+1:level_off+L]
println("max|Delta H_CM,level| = ", maximum(abs.(HCMlevel_d .- HCMlevel_w)))

Hlevellevel_d = Hd_full[level_off+1:level_off+L, level_off+1:level_off+L]; Hlevellevel_w = Hw_full[level_off+1:level_off+L, level_off+1:level_off+L]
println("max|Delta H_level,level| = ", maximum(abs.(Hlevellevel_d .- Hlevellevel_w)))

println()
println("scale (max|Hd_full|) = ", maximum(abs.(Hd_full)))
