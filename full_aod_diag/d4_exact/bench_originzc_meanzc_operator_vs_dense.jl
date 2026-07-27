# port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26, Phase A item 1:
# complete-inner-solve performance A/B for origin-ZC and CM+ZC operator FG (:operator) vs the
# pre-existing dense-reference path (:dense_reference), at D=4 and real D=20/W=80,000. Fills the
# gap left open by docs/FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md ("Origin-ZC and CM+ZC's
# own complete-inner-solve wall-clock/allocation/GC/KNITRO-iteration A/B ... NOT MEASURED").
#
# Measures the SAME calls task §17/§18 ask about: complete inner solve (archOZ_verified_state /
# archCM_meanzc's own equivalent, invoked via *_production_value_verified) + full downstream
# gradient (*_production_gradient), repeated at a fixed outer point (fresh KNITRO solve each call --
# this is the real "complete inner solve" cost, not a cached no-op).
#
# Usage: julia --project=. -t <N> full_aod_diag/d4_exact/bench_originzc_meanzc_operator_vs_dense.jl [d4|d20]
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics

const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"
const NREPS = SCALE == "d4" ? 15 : 5

nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

"Warm up (JIT) then time NREPS fresh complete inner-solve+gradient calls at x_free. Returns a NamedTuple of stats."
function bench_complete_solve(label::String, solve_fn::Function, grad_fn::Function, x_free, extra_args...; nreps::Int = NREPS)
    # warm-up (JIT + first-call allocation noise)
    _, base, verify = solve_fn(x_free, extra_args...)
    grad_fn(x_free, extra_args...; base = base, verify = verify)

    times = Float64[]
    allocs = Int[]
    gctimes = Float64[]
    local base_last, verify_last
    for r in 1:nreps
        t0 = time_ns()
        stats = @timed begin
            _, b, v = solve_fn(x_free, extra_args...)
            g, _ = grad_fn(x_free, extra_args...; base = b, verify = v)
            (b, v, g)
        end
        push!(times, stats.time)
        push!(allocs, stats.bytes)
        push!(gctimes, stats.gctime)
        base_last, verify_last = stats.value[1], stats.value[2]
    end
    return (label = label, median_time_s = median(times), min_time_s = minimum(times),
            median_bytes = median(allocs), median_gctime_s = median(gctimes),
            inner_status = verify_last.inner_status, Delta_dual = verify_last.Delta_dual, nreps = nreps)
end

function print_row(s)
    @printf("  %-38s median=%.4fs  min=%.4fs  alloc=%.3fMB  gc=%.4fs  status=%d  nreps=%d\n",
            s.label, s.median_time_s, s.min_time_s, s.median_bytes / 1e6, s.median_gctime_s, s.inner_status, s.nreps)
end

results = Dict{String,Any}()

function bench_originzc(ctx, x_free_calib; K_configs, W_label)
    D = ctx.D
    for (K_mean, K_pair) in K_configs
        println("\n---- origin-ZC $W_label K_mean=$K_mean K_pair=$K_pair ----")
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        νfull0 = nu0_origin(K_mean, D)
        pcx_dense = build_originzc_production_context(ctx, CS, layout; fg_backend = :dense_reference)
        pcx_op = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)

        s_dense = bench_complete_solve("origin_zc/$W_label/K=$K_mean,$K_pair/dense",
            (xf, pcx) -> cm_originzc_production_value_verified(xf, νfull0, pcx),
            (xf, pcx; base, verify) -> cm_originzc_production_gradient(xf, νfull0, pcx, ctx, pe_g; base = base, verify = verify, threaded = false),
            x_free_calib, pcx_dense)
        s_op = bench_complete_solve("origin_zc/$W_label/K=$K_mean,$K_pair/operator",
            (xf, pcx) -> cm_originzc_production_value_verified(xf, νfull0, pcx),
            (xf, pcx; base, verify) -> cm_originzc_production_gradient(xf, νfull0, pcx, ctx, pe_g; base = base, verify = verify, threaded = false),
            x_free_calib, pcx_op)
        print_row(s_dense); print_row(s_op)
        speedup = s_dense.median_time_s / s_op.median_time_s
        alloc_ratio = s_op.median_bytes / max(s_dense.median_bytes, 1)
        @printf("  => speedup=%.3fx  alloc_ratio(op/dense)=%.3f  status_match=%s  Delta_dual_diff=%.3e\n",
                speedup, alloc_ratio, s_dense.inner_status == s_op.inner_status,
                abs(s_dense.Delta_dual - s_op.Delta_dual))
        results["origin_zc/$W_label/K=$K_mean,$K_pair"] = Dict(
            "dense" => Dict("median_time_s" => s_dense.median_time_s, "median_bytes" => s_dense.median_bytes, "median_gctime_s" => s_dense.median_gctime_s),
            "operator" => Dict("median_time_s" => s_op.median_time_s, "median_bytes" => s_op.median_bytes, "median_gctime_s" => s_op.median_gctime_s),
            "speedup" => speedup, "alloc_ratio" => alloc_ratio,
            "status_match" => s_dense.inner_status == s_op.inner_status,
            "delta_dual_diff" => abs(s_dense.Delta_dual - s_op.Delta_dual))
    end
end

function bench_meanzc(ctx, x_free_calib; L::Int, K_configs, W_label)
    for (K_mean, K_pair) in K_configs
        println("\n---- CM+ZC $W_label L=$L K_mean=$K_mean K_pair=$K_pair ----")
        νvec0 = [Float64(factorial(k)) for k in 1:K_mean]
        pcx_dense = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
                                                          contrasts = :anchored, meanzc_basis = :direct, inner_fg_backend = :dense_reference)
        pcx_op = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
                                                       contrasts = :anchored, meanzc_basis = :direct, inner_fg_backend = :operator)

        s_dense = bench_complete_solve("cm_meanzc/$W_label/L=$L,K=$K_mean,$K_pair/dense",
            (xf, pcx) -> cm_meanzc_production_value_verified(xf, νvec0, pcx),
            (xf, pcx; base, verify) -> cm_meanzc_production_gradient(xf, νvec0, pcx, ctx, pe_g; base = base, verify = verify, threaded = false),
            x_free_calib, pcx_dense)
        s_op = bench_complete_solve("cm_meanzc/$W_label/L=$L,K=$K_mean,$K_pair/operator",
            (xf, pcx) -> cm_meanzc_production_value_verified(xf, νvec0, pcx),
            (xf, pcx; base, verify) -> cm_meanzc_production_gradient(xf, νvec0, pcx, ctx, pe_g; base = base, verify = verify, threaded = false),
            x_free_calib, pcx_op)
        print_row(s_dense); print_row(s_op)
        speedup = s_dense.median_time_s / s_op.median_time_s
        alloc_ratio = s_op.median_bytes / max(s_dense.median_bytes, 1)
        @printf("  => speedup=%.3fx  alloc_ratio(op/dense)=%.3f  status_match=%s  Delta_dual_diff=%.3e\n",
                speedup, alloc_ratio, s_dense.inner_status == s_op.inner_status,
                abs(s_dense.Delta_dual - s_op.Delta_dual))
        results["cm_meanzc/$W_label/L=$L,K=$K_mean,$K_pair"] = Dict(
            "dense" => Dict("median_time_s" => s_dense.median_time_s, "median_bytes" => s_dense.median_bytes, "median_gctime_s" => s_dense.median_gctime_s),
            "operator" => Dict("median_time_s" => s_op.median_time_s, "median_bytes" => s_op.median_bytes, "median_gctime_s" => s_op.median_gctime_s),
            "speedup" => speedup, "alloc_ratio" => alloc_ratio,
            "status_match" => s_dense.inner_status == s_op.inner_status,
            "delta_dual_diff" => abs(s_dense.Delta_dual - s_op.Delta_dual))
    end
end

pe_g = nothing
if SCALE == "d4"
    global ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    global pe_g = build_pivot_elimination(ctx)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    bench_originzc(ctx, x_free_calib; K_configs = [(1, 0), (1, 1), (2, 2)], W_label = "D4")
    bench_meanzc(ctx, x_free_calib; L = 10, K_configs = [(1, 0), (1, 1), (2, 2)], W_label = "D4")
elseif SCALE == "d20"
    W = 80000
    global ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                                        draw_seed = 20260719, destination_sample = :exclude_row)
    global pe_g = build_pivot_elimination(ctx)
    D = ctx.D; Ddest = ctx.D_dest
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe_g)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0 * 1.01, zfree0)
    x_free_calib = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe_g))))
    bench_originzc(ctx, x_free_calib; K_configs = [(1, 1)], W_label = "D20/W=$W")
    bench_meanzc(ctx, x_free_calib; L = 50, K_configs = [(1, 1)], W_label = "D20/W=$W")
else
    error("unknown SCALE=$SCALE, expected d4|d20")
end

open(joinpath(D4X, "bench_originzc_meanzc_operator_vs_dense_$(SCALE)_results.csv"), "w") do io
    println(io, "case,backend,median_time_s,median_bytes,median_gctime_s,speedup,alloc_ratio,status_match,delta_dual_diff")
    for (case, r) in results
        for backend in ("dense", "operator")
            b = r[backend]
            println(io, "$case,$backend,$(b["median_time_s"]),$(b["median_bytes"]),$(b["median_gctime_s"]),$(r["speedup"]),$(r["alloc_ratio"]),$(r["status_match"]),$(r["delta_dual_diff"])")
        end
    end
end
println("\nWrote bench_originzc_meanzc_operator_vs_dense_$(SCALE)_results.csv")
