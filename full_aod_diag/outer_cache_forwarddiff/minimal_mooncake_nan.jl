# ============================================================================
# Mooncake counterpart to minimal_enzyme_nan.jl -- same bisection, same three
# steps (withinTransform alone / full grav_scalar / bare sum(;dims=k)), using
# Mooncake via DifferentiationInterface exactly as
# ../ad_benchmark/test_mooncake2.jl does. Independent AD system, same
# candidate root cause (sum(;dims=k) reductions), no project code loaded.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/minimal_mooncake_nan.jl
# ============================================================================
using ForwardDiff, LinearAlgebra, Random
using Mooncake, DifferentiationInterface
const DI = DifferentiationInterface

withinTransform(z) = begin
    lz = log.(z)
    D = size(z, 1)
    lz .- (sum(lz, dims = 2) ./ D) .- (sum(lz, dims = 1) ./ D) .+ (sum(lz) / D^2)
end

function grav_scalar(θ, D, τ)
    Aod = reshape(θ, D, D)
    Wτ = withinTransform(τ)
    WA = withinTransform(Aod)
    s = zero(eltype(θ))
    @inbounds for o in 1:D, d in 1:D
        s += Wτ[o, d] * WA[o, d]
    end
    return s
end

function within_sum_scalar(θ, D)
    A = reshape(θ, D, D)
    W = withinTransform(A)
    return sum(W)
end

function dims_sum_scalar(θ, D)
    A = reshape(θ, D, D)
    r = sum(A, dims = 2)
    c = sum(A, dims = 1)
    g = sum(A)
    return sum(r) + sum(c) + g
end

Random.seed!(1)
D = 4
τ = rand(D, D) .+ 0.5
θ0 = vec(rand(D, D) .+ 0.5)

backend = DI.AutoMooncake(; config = nothing)

function try_di(label, f, θ0)
    g_fd = ForwardDiff.gradient(f, θ0)
    println(">>> ", label)
    println("    ForwardDiff: ", g_fd, "  finite=", all(isfinite, g_fd))
    try
        prep = DI.prepare_gradient(f, backend, θ0)
        g_mc = DI.gradient(f, prep, backend, θ0)
        nnan = count(isnan, g_mc)
        relerr = norm(g_mc .- g_fd) / max(norm(g_fd), 1e-300)
        println("    Mooncake:    ", g_mc, "  finite=", all(isfinite, g_mc), "  nan_count=", nnan, "  relerr=", relerr)
    catch e
        println("    Mooncake FAILED: ", sprint(showerror, e)[1:min(400, end)])
    end
end

try_di("STEP 1: withinTransform alone, output=sum(W)", θ -> within_sum_scalar(θ, D), θ0)
try_di("STEP 2: grav_scalar (withinTransform(tau Const-like) .* withinTransform(Aod), summed)", θ -> grav_scalar(θ, D, τ), θ0)
try_di("STEP 3: bare sum(;dims=k) reductions only", θ -> dims_sum_scalar(θ, D), θ0)
println("\nDONE")
