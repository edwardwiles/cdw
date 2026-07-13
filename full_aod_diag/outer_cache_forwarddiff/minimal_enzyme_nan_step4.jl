# ============================================================================
# Step 4 of the reverse-mode NaN bisection: minimal_enzyme_nan.jl's steps 1-3
# (withinTransform alone, the full withinTransform-based grav_scalar, bare
# sum(;dims=k)) ALL matched ForwardDiff exactly, NO NaN -- ruling out
# sum(;dims=k)/broadcast-demeaning as the cause. This step adds back the ONE
# piece minimal_enzyme_nan.jl's clean reimplementation did NOT include: the
# real newGravityMoment!'s DEAD-CODE preamble (misc/doubleDiff.jl::doubleDiff,
# a MUTATING function -- `deltaZ = zeros(eltype(z),D,D)` then `@.deltaZ[o,:]
# = ...` in a loop -- whose entire output, `meanτ`, is provably unused
# whenever UoModel==1, per newGravityMoment!.jl's own control flow). The
# session's prior "type-stable" fix (`meanτ = 0` -> `zero(eltype(τ))`) fixed
# a COMPILE-time Enzyme crash on this dead code but did NOT fix the NaN
# gradients that appear once compilation succeeds -- meaning the dead
# `doubleDiff` call was never actually tested in ISOLATION for whether it is
# the thing making a mutating-array pattern's reverse-mode adjoint go wrong
# once combined with a live, differentiated `withinTransform` computation
# right after it in the same function.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/minimal_enzyme_nan_step4.jl
# ============================================================================
using ForwardDiff, LinearAlgebra, Random
import Enzyme

withinTransform(z) = begin
    lz = log.(z)
    D = size(z, 1)
    lz .- (sum(lz, dims = 2) ./ D) .- (sum(lz, dims = 1) ./ D) .+ (sum(lz) / D^2)
end

"Exact structural copy of misc/doubleDiff.jl::doubleDiff (the D=1 method), unmodified."
function doubleDiff_copy(z)
    D = size(z, 1)
    deltaZ = zeros(eltype(z), D, D)
    for o in 1:D
        @. deltaZ[o, :] = (log.(z[o, :]) .- log.(z[1, :])) .- (log.(z[o, 2]) .- log.(z[1, 2]))
    end
    return deltaZ
end

"Exact structural copy of newGravityMoment!.jl's UoModel==1 body, including the DEAD meanτ/doubleDiff preamble."
function grav_scalar_with_dead_code(θ, D, τ)
    Aod = reshape(θ, D, D)

    # --- dead code for UoModel==1 (present, executed, result unused) ---
    deltaτ = doubleDiff_copy(τ)
    meanτ = zero(eltype(τ))   # already-applied type-stability fix
    for o in 2:D
        meanτ += deltaτ[o, 1]
        for d in 3:D
            meanτ += deltaτ[o, d]
        end
    end
    meanτ /= (D - 1)^2
    # meanτ is now discarded -- never read again below.

    # --- live computation (UoModel==1 branch) ---
    Wτ = withinTransform(τ)
    WA = withinTransform(Aod)
    s = zero(eltype(θ))
    @inbounds for o in 1:D, d in 1:D
        s += Wτ[o, d] * WA[o, d]
    end
    return s
end

Random.seed!(1)
D = 4
τ = rand(D, D) .+ 0.5
θ0 = vec(rand(D, D) .+ 0.5)

g_fd = ForwardDiff.gradient(θ -> grav_scalar_with_dead_code(θ, D, τ), θ0)
println(">>> ForwardDiff: ", g_fd)
println(">>> ForwardDiff finite everywhere: ", all(isfinite, g_fd))

g_enz = zeros(length(θ0))
try
    Enzyme.autodiff(Enzyme.Reverse, θ -> grav_scalar_with_dead_code(θ, D, τ), Enzyme.Active,
                     Enzyme.Duplicated(θ0, g_enz))
    nnan = count(isnan, g_enz)
    println(">>> Enzyme:      ", g_enz)
    println(">>> Enzyme finite everywhere: ", all(isfinite, g_enz), "  nan_count=", nnan)
    if nnan > 0
        println(">>> REPRODUCED: dead doubleDiff_copy preamble + live withinTransform => NaN under Enzyme reverse mode.")
        println(">>> NaN at indices: ", findall(isnan, g_enz))
    else
        println(">>> NOT reproduced here -- the dead-code preamble alone (in this minimal form) is not the trigger;")
        println(">>> the real bug likely needs the broader EK_moments_gammanorm_directgp! context (shared/mutated")
        println(">>> preallocated scratch buffers in gamma_obj: UPow_scratch, UσPow_scratch, Ū, etc.) to reproduce.")
    end
catch e
    println(">>> Enzyme FAILED to compile/run: ", sprint(showerror, e))
end
