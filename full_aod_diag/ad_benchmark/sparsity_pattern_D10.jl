# §6 at D=10 (per user request): does the coloring picture change with a
# less toy-sized economy, short of the full D=20 real-data build? Same
# γ_d≡1 + direct-γ' config, DFake=10 fake data (new random draw, seed fixed
# for reproducibility), W=Jac_W=8000 kept the same. Self-contained (does not
# reuse setup_context.jl's D=4 build) so the D=4 results/files are untouched.
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
using SparseConnectivityTracer, SparseDiffTools, DifferentiationInterface, SparseMatrixColorings
const DI = DifferentiationInterface
const ROOT = dirname(dirname(@__DIR__))
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(dirname(@__DIR__), "moments_gammanorm.jl"))
include(joinpath(@__DIR__, "derivative_core.jl"))

const D10 = 10
params10 = (server=1,user=2,fakeData=1,DFake=D10,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so10 = master_setup(params10)
up10 = (; params10..., D=so10.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up10)
ps10 = master_prestep(so10.data, so10.counters, up10)
pp10 = master_prepare_cc(so10.data, so10.counters, ps10, up10)
@unpack θ_initial, U, γ, outer_constr_index, nTotalMoments = pp10
D = so10.D; bi = params10.baseIndex; σ = params10.σHat; μHat = γ.μHat
@assert D == D10
l = length(θ_initial)
d = nTotalMoments
println(">>> D=$D  l=$l (theta length)  nTotalMoments=$d  outer_constr_index=$outer_constr_index")

θ_A = build_theta_gammanorm(θ_initial, D, bi, μHat, σ)
bounds = theoretical_gammaprime_bounds(γ, σ)
println(">>> theoretical γ'_focal bounds: [$(bounds.γp_lo), $(bounds.γp_hi)]")

# ---- numerical union sparsity, at init + 4 random perturbations (no KNITRO needed) ----
Random.seed!(20260712)
Aod_offset = 3 + D
test_thetas = [θ_A]
for _ in 1:4
    θr = copy(θ_A)
    θr[1] *= (0.7 + 0.6*rand())
    θr[3+D] = clamp(θr[3+D] * (0.9 + 0.2*rand()), bounds.γp_lo, bounds.γp_hi)
    for i in Aod_offset+1:Aod_offset+D^2
        θr[i] *= (0.7 + 0.6*rand())
    end
    push!(test_thetas, θr)
end

N_test = 200
Usub = U[1:N_test, :]
numeric_union = falses(d + 2, l)
tol = 1e-8
for θt in test_thetas
    f! = (Hvec, θ) -> begin
        H = reshape(Hvec, N_test, d + 2)
        moment_map!(H, θ, Usub, γ)
    end
    Hvec0 = zeros(N_test * (d + 2))
    J = ForwardDiff.jacobian(f!, Hvec0, θt)
    Jr = reshape(J, N_test, d + 2, l)
    for hrow in 1:d+2, j in 1:l
        if maximum(abs.(Jr[:, hrow, j])) > tol
            numeric_union[hrow, j] = true
        end
    end
end
nnz_total = count(numeric_union)
println(">>> numeric union nnz = $nnz_total / $(length(numeric_union))  density=$(round(nnz_total/length(numeric_union);digits=4))")

# per-row nnz count summary (trade-share rows should still show 3 [μ,σ,own-A]; gravity row dense)
tradeshare_nnz = [count(numeric_union[2+o+(dd-1)*D, :]) for dd in 1:D for o in 1:D]
gravity_row = 2 + D^2 + 2
cf_row = 2 + D^2 + 1
println(">>> trade-share row nnz: min=$(minimum(tradeshare_nnz)) max=$(maximum(tradeshare_nnz)) (expect 3 if D=4's structural finding holds)")
println(">>> counterfactual-price row nnz: $(count(numeric_union[cf_row,:]))")
println(">>> gravity row nnz: $(count(numeric_union[gravity_row,:])) / $l")

# ---- colored sparse Jacobian: does compression improve at D=10? ----
f! = (Hvec, θ) -> begin
    H = reshape(Hvec, N_test, d + 2)
    moment_map!(H, θ, Usub, γ)
end
Hvec0 = zeros(N_test * (d + 2))
backend = DI.AutoSparse(DI.AutoForwardDiff();
    sparsity_detector = DI.DenseSparsityDetector(DI.AutoForwardDiff(); atol=1e-12),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
t_prep = @elapsed prep = DI.prepare_jacobian(f!, Hvec0, backend, θ_A)
ncolors = length(unique(SparseMatrixColorings.column_colors(prep.coloring_result)))
println(">>> D=$D: colors used = $ncolors / $l params  (compression = $(round(100*(1-ncolors/l);digits=1))%)  prep=$(round(t_prep;digits=3))s")

# correctness: colored sparse vs dense, at N=N_test
JD = DI.jacobian(f!, copy(Hvec0), prep, backend, θ_A)
JA = ForwardDiff.jacobian(f!, Hvec0, θ_A)
using SparseArrays
err = maximum(abs.(Matrix(JD) .- JA))
println(">>> dense vs sparse max abs err = $err")

# ---- full N=8000 timing comparison: dense vs colored sparse, at D=10 ----
f_full! = (Hvec, θ) -> begin
    H = reshape(Hvec, size(U,1), d + 2)
    moment_map!(H, θ, U, γ)
end
Hvec0_full = zeros(size(U,1) * (d + 2))
t_dense_full = @elapsed ForwardDiff.jacobian(f_full!, Hvec0_full, θ_A)
t_dense_full2 = @elapsed ForwardDiff.jacobian(f_full!, Hvec0_full, θ_A)
backend_full = DI.AutoSparse(DI.AutoForwardDiff();
    sparsity_detector = DI.DenseSparsityDetector(DI.AutoForwardDiff(); atol=1e-12),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
t_prep_full = @elapsed prep_full = DI.prepare_jacobian(f_full!, Hvec0_full, backend_full, θ_A)
t_sparse_full = @elapsed DI.jacobian(f_full!, copy(Hvec0_full), prep_full, backend_full, θ_A)
t_sparse_full2 = @elapsed DI.jacobian(f_full!, copy(Hvec0_full), prep_full, backend_full, θ_A)
println(">>> FULL N=8000 timing: dense=$(round(t_dense_full2;digits=3))s  sparse(steady)=$(round(t_sparse_full2;digits=3))s  sparse_prep=$(round(t_prep_full;digits=3))s")

open(joinpath(@__DIR__, "sparsity_summary_D10.txt"), "w") do f
    println(f, "D=$D  l=$l  d=$d  nnz=$nnz_total/$(length(numeric_union))  density=$(round(nnz_total/length(numeric_union);digits=4))")
    println(f, "trade-share row nnz: min=$(minimum(tradeshare_nnz)) max=$(maximum(tradeshare_nnz))")
    println(f, "counterfactual-price row nnz: $(count(numeric_union[cf_row,:]))")
    println(f, "gravity row nnz: $(count(numeric_union[gravity_row,:])) / $l")
    println(f, "colors used: $ncolors / $l  (compression $(round(100*(1-ncolors/l);digits=1))%)")
    println(f, "dense vs sparse correctness: max abs err = $err")
    println(f, "FULL N=8000: dense=$(round(t_dense_full2;digits=3))s  sparse(steady)=$(round(t_sparse_full2;digits=3))s  sparse_prep=$(round(t_prep_full;digits=3))s")
end
@save joinpath(@__DIR__, "sparsity_pattern_D10.jld2") numeric_union D l d ncolors
println("D10 SPARSITY DONE")
