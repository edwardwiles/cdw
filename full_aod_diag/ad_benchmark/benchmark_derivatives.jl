# §8: performance benchmarking at production scale (N=Jac_W=8000).
using BenchmarkTools
include("setup_context.jl")
include("derivative_core.jl")
include("derivative_methods.jl")
using SparseArrays

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]
d = data["nTotalMoments"]; oci = data["outer_constr_index"]

rows = NamedTuple[]

for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = d, outer_constr_index = oci)

    println("=== benchmarking point $name (N=$(size(pp.U,1))) ===")

    # ---- Method A: dense Jacobian ----
    t_compile_A = @elapsed method_A_dense_jacobian(θ, ctx)
    bA = @benchmark method_A_dense_jacobian($θ, $ctx) samples=8 seconds=60
    tA_med = median(bA).time / 1e9; tA_min = minimum(bA).time / 1e9; allocA = median(bA).memory

    bA_contract = @benchmark contract_dense_to_div_grad(J, $ctx) setup=(J=method_A_dense_jacobian($θ,$ctx)) samples=5 seconds=30
    tAc_med = median(bA_contract).time / 1e9

    # ---- Method B: direct scalar ----
    t_compile_B = @elapsed method_B_forward_scalar(θ, ctx)
    bB = @benchmark method_B_forward_scalar($θ, $ctx) samples=20 seconds=30
    tB_med = median(bB).time / 1e9; tB_min = minimum(bB).time / 1e9; allocB = median(bB).memory

    # ---- Method D: colored sparse Jacobian ----
    t_prep_D = @elapsed (prep, backend, f!, Hvec0) = method_D_sparse_prep(θ, ctx)
    t_compile_D = @elapsed method_D_sparse_jacobian(θ, ctx, prep, backend, f!, Hvec0)
    bD = @benchmark method_D_sparse_jacobian($θ, $ctx, $prep, $backend, $f!, $Hvec0) samples=8 seconds=60
    tD_med = median(bD).time / 1e9; tD_min = minimum(bD).time / 1e9; allocD = median(bD).memory
    ncolors = length(unique(SparseMatrixColorings.column_colors(prep.coloring_result)))

    println(@sprintf("  A (dense jac):        median=%.4fs  min=%.4fs  alloc=%.1fMB  [+contract median=%.5fs]  compile-incl=%.3fs",
        tA_med, tA_min, allocA/1e6, tAc_med, t_compile_A))
    println(@sprintf("  B (scalar FD):         median=%.4fs  min=%.4fs  alloc=%.1fMB  compile-incl=%.3fs",
        tB_med, tB_min, allocB/1e6, t_compile_B))
    println(@sprintf("  D (sparse colored, %d colors): median=%.4fs  min=%.4fs  alloc=%.1fMB  prep=%.3fs  compile-incl=%.3fs",
        ncolors, tD_med, tD_min, allocD/1e6, t_prep_D, t_compile_D))

    push!(rows, (point=String(name), method="A_dense", median_s=tA_med, min_s=tA_min, alloc_MB=allocA/1e6, note="+contract=$(round(tAc_med,digits=5))s"))
    push!(rows, (point=String(name), method="B_scalar", median_s=tB_med, min_s=tB_min, alloc_MB=allocB/1e6, note=""))
    push!(rows, (point=String(name), method="D_sparse", median_s=tD_med, min_s=tD_min, alloc_MB=allocD/1e6, note="$(ncolors)colors,prep=$(round(t_prep_D,digits=3))s"))
end

open(joinpath(@__DIR__, "benchmark_results.csv"), "w") do f
    println(f, "point,method,median_s,min_s,alloc_MB,note")
    for r in rows
        println(f, "$(r.point),$(r.method),$(r.median_s),$(r.min_s),$(r.alloc_MB),$(r.note)")
    end
end
println("BENCHMARK DONE — full_aod_diag/ad_benchmark/benchmark_results.csv")
