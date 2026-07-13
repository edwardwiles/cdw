# ============================================================================
# Step 6: close the gap between steps 1-4 (clean, synthetic, no-NaN) and the
# prior session's claim that "the gravity moment alone" reproduces the NaN.
#
# Steps 1-4 tested `withinTransform` on a BARE reshape(θ,D,D) matrix -- but the
# REAL gravity gradient (`../PsiObjectiveBundleImplicitMethodB_fullA.jl::
# make_gravity_grad`, lines 206-225) builds `AodPow` through a materially
# richer chain before ever calling `withinTransform`:
#   Aod = Aod_θ .* cHat .* (((wHat.*τ)./(wHat[1,1].*τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
#   AodPow = (Aod ./ cHat) .^ (-μ)
# -- with μ appearing in TWO separate power operations (1/μ and -μ) across a
# chain of elementwise products/divisions of REAL data arrays (cHat, wHat, τ,
# lambda), not a simple reshape. This step differentiates that EXACT formula,
# copy-pasted from production, using REAL project data (via
# ../ad_benchmark/setup_context.jl::build_ad_context -- no KNITRO solve
# needed, this only needs U/γ/data, not an inner solve) and a REAL theta from
# the frozen benchmark points, with Enzyme.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/minimal_enzyme_nan_step6_real_gravity.jl
# ============================================================================
using ForwardDiff, LinearAlgebra, Random, JLD2
import Enzyme
const ADB = joinpath(dirname(@__DIR__), "ad_benchmark")
include(joinpath(ADB, "setup_context.jl"))   # brings Parameters/JLD2/CS/withinTransform(Main)/etc.

so, pp = build_ad_context()
D = so.D
γobj = pp.γ
pts = JLD2.load(joinpath(ADB, "benchmark_points.jld2"))["points"]
θ0 = pts[:A].θ
println(">>> D=", D, "  l=", length(θ0))

Aod_offset = 3 + D
τ = γobj.τ; cHat = γobj.cHat; wHat = γobj.wHat
lambda = reshape(γobj.P, (D, D))'

function sumGrav_scalar(θθ, τ, cHat, wHat, lambda, D, Aod_offset)
    T = eltype(θθ)
    μ = θθ[1]
    Aod_θ = reshape(vcat(θθ[Aod_offset+1:Aod_offset+D^2]), (D, D))
    Aod = Aod_θ .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ cHat) .^ (-μ)
    Wτ = withinTransform(τ)
    WA = withinTransform(AodPow)
    s = zero(T)
    for o in 1:D, d in 1:D
        s += Wτ[o, d] * WA[o, d]
    end
    return s
end

primal = sumGrav_scalar(θ0, τ, cHat, wHat, lambda, D, Aod_offset)
println(">>> primal sumGrav at real theta (point A) = ", primal, "  finite=", isfinite(primal))

g_fd = ForwardDiff.gradient(θ -> sumGrav_scalar(θ, τ, cHat, wHat, lambda, D, Aod_offset), θ0)
println(">>> ForwardDiff finite everywhere: ", all(isfinite, g_fd), "  nnan=", count(isnan, g_fd))

g_enz = zeros(length(θ0))
try
    Enzyme.autodiff(Enzyme.Reverse,
        θθ -> sumGrav_scalar(θθ, τ, cHat, wHat, lambda, D, Aod_offset),
        Enzyme.Active, Enzyme.Duplicated(θ0, g_enz))
    nnan = count(isnan, g_enz)
    relerr = norm(g_enz .- g_fd) / max(norm(g_fd), 1e-300)
    println(">>> Enzyme (plain Reverse): finite=", all(isfinite, g_enz), "  nnan=", nnan, "  relerr=", relerr)
    if nnan > 0
        println(">>> REPRODUCED with the REAL make_gravity_grad formula + real data. NaN indices: ", findall(isnan, g_enz))
        println(">>> First-NaN theta index maps to: ", (findall(isnan, g_enz) .<= Aod_offset) , " (true=scalar param, false=A_od entry)")
    end
catch e
    println(">>> Enzyme (plain Reverse) FAILED: ", sprint(showerror, e)[1:min(600,end)])
    println(">>> retrying with set_runtime_activity(Reverse)...")
    try
        fill!(g_enz, 0.0)
        Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
            θθ -> sumGrav_scalar(θθ, τ, cHat, wHat, lambda, D, Aod_offset),
            Enzyme.Active, Enzyme.Duplicated(θ0, g_enz))
        nnan = count(isnan, g_enz)
        relerr = norm(g_enz .- g_fd) / max(norm(g_fd), 1e-300)
        println(">>> Enzyme (runtime_activity): finite=", all(isfinite, g_enz), "  nnan=", nnan, "  relerr=", relerr)
    catch e2
        println(">>> Enzyme (runtime_activity) ALSO FAILED: ", sprint(showerror, e2)[1:min(400,end)])
    end
end
