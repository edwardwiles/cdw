# Correctness gate for cm_hessian_architecture_threaded.jl: the threaded Hessian MUST reproduce
# the serial Architecture C Hessian exactly (this is a draw-partitioned SUM -- floating-point
# associativity differences are possible in principle since thread-chunk order differs from the
# serial 1:W order, so "exactly" is checked but a tiny FP-order tolerance would still be
# legitimate; report shows which it actually is).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_threaded.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics
using Base.Threads: nthreads

println(">>> nthreads() = ", nthreads(), " (need > 1 for this to be a real test, not just a no-op reduction)")

W = 80000
DELTA = 1.0
L = 50
println(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
@printf ">>> ctx built in %.1fs. D=%d W=%d\n" (time() - t0) D W

x_free_calib = ctx.θ0_up[ctx.free_idx]

cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = L,
               cm_basis = :cumulative, cm_hessian_backend = :structured)
pcx = build_cm_production_context_v2(ctx, CS, cfg)
ctx_cm = pcx.ctx_cm
cctx = build_cm_bin_ctx(ctx, pcx.aug)
cctxt = build_cm_bin_ctx_threaded(ctx, pcx.aug)

base = cm_base_state_v2(x_free_calib, pcx)
println(">>> inner_status=", base.inner_status, " ζstar=", base.ζstar)
_archC_prep_for_hessian!(ctx_cm.obj, vcat(base.ζstar, base.λstar))

n = pcx.aug.ncore + pcx.aug.ncm
h_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_threaded = Vector{Float64}(undef, n * (n + 1) ÷ 2)

hessian_cm_structured!(h_serial, ctx_cm.obj, cctx)
hessian_cm_structured_threaded!(h_threaded, ctx_cm.obj, cctxt)

d = maximum(abs.(h_serial .- h_threaded))
rel = d / max(1.0, maximum(abs.(h_serial)))
@printf ">>> max|serial-threaded|=%.3e  rel=%.3e  (n=%d, packed length %d)\n" d rel n length(h_serial)

# Repeat at a second (perturbed) point, and check the balanced_ranges partition itself.
Random.seed!(9911)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.02 .* randn(length(x_free_perturbed) - 1))
base2 = cm_base_state_v2(x_free_perturbed, pcx)
println(">>> [perturbed] inner_status=", base2.inner_status)
if base2.inner_status in (0, -100, -101, -103)
    _archC_prep_for_hessian!(ctx_cm.obj, vcat(base2.ζstar, base2.λstar))
    hessian_cm_structured!(h_serial, ctx_cm.obj, cctx)
    hessian_cm_structured_threaded!(h_threaded, ctx_cm.obj, cctxt)
    d2 = maximum(abs.(h_serial .- h_threaded))
    @printf ">>> [perturbed] max|serial-threaded|=%.3e\n" d2
end

ranges = balanced_ranges(80000, nthreads())
covered = vcat(collect.(ranges)...)
println(">>> balanced_ranges covers 1:W exactly once: ", sort(covered) == collect(1:80000))
println(">>> range lengths: ", length.(ranges), " (max-min=", maximum(length.(ranges)) - minimum(length.(ranges)), ")")

println("DONE")
