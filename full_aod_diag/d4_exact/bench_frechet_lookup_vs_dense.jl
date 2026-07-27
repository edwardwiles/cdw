# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26, Phase A item 4:
# complete-inner-solve performance A/B for common-Frechet's :cm_frechet_lookup (now allocation-
# fixed, obj::O type param) vs :dense_reference, at D=4 and real D=20/W=80,000 -- the missing gate
# from docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md before the default can flip.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/bench_frechet_lookup_vs_dense.jl [d4|d20]
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, Statistics

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"
const NREPS = SCALE == "d4" ? 15 : 5

function bench(label, f, nreps)
    f()   # warm-up
    times = Float64[]; allocs = Int[]; gctimes = Float64[]
    for _ in 1:nreps
        s = @timed f()
        push!(times, s.time); push!(allocs, s.bytes); push!(gctimes, s.gctime)
    end
    med_t = sort(times)[cld(nreps,2)]; med_b = sort(allocs)[cld(nreps,2)]; med_g = sort(gctimes)[cld(nreps,2)]
    @printf "  %-24s median=%.4fs  alloc=%.3fMB  gc=%.4fs\n" label med_t med_b/1e6 med_g
    return (time=med_t, bytes=med_b)
end

function run(ctx, x_free, L, W_label)
    println("\n---- $W_label L=$L ----")
    pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
        cm_hessian_backend = :structured, inner_fg_backend = :dense_reference)
    pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = :anchored,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup)

    sd = bench("dense", () -> archC_frechet_base_state(x_free, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets), NREPS)
    sl = bench("lookup(operator)", () -> archC_frechet_base_state(x_free, pcx_lookup.ctx_cm, pcx_lookup.cctx, pcx_lookup.aug.level_targets), NREPS)
    @printf "  => speedup=%.3fx  alloc_ratio(lookup/dense)=%.4f\n" sd.time/sl.time sl.bytes/max(sd.bytes,1)

    bd = archC_frechet_base_state(x_free, pcx_dense.ctx_cm, pcx_dense.cctx, pcx_dense.aug.level_targets)
    bl = archC_frechet_base_state(x_free, pcx_lookup.ctx_cm, pcx_lookup.cctx, pcx_lookup.aug.level_targets)
    println("  correctness: zeta* diff=", abs(bd.ζstar - bl.ζstar), "  status dense=", bd.inner_status, " lookup=", bl.inner_status)
end

if SCALE == "d4"
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    x_free = ctx.θ0_up[ctx.free_idx]
    run(ctx, x_free, 10, "D4")
    run(ctx, x_free, 50, "D4")
elseif SCALE == "d20"
    W = 80000
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                                 draw_seed = 20260719, destination_sample = :exclude_row)
    pe_g = build_pivot_elimination(ctx)
    D = ctx.D; Ddest = ctx.D_dest
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0 * 1.01, zfree0)
    x_free = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe_g))))
    run(ctx, x_free, 50, "D20/W=$W")
else
    error("unknown SCALE=$SCALE")
end
