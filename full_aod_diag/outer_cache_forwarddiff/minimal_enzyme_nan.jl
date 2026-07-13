# ============================================================================
# Reverse-mode NaN bisection (mega-prompt §13), continuing from
# ../ad_benchmark/README.md §5 / SESSION_SUMMARY_2026-07-12.md §8, which
# established:
#   - NOT the custom `gamma` EnzymeRule (verified correct in total isolation
#     before being loaded; also: newGravityMoment!.jl calls NO gamma() at all,
#     yet the gravity moment ALONE already reproduces the identical NaN
#     pattern -- so the gamma rule is provably not the culprit for at least
#     that component).
#   - NOT hard-max/branching (gravity moment has zero branching).
#   - NOT the reshape(vcat(...)) A_od-construction pattern (tested directly,
#     rewriting as a preallocated-matrix loop gives identical NaN indices).
#   - IS reproduced identically by two independent reverse-mode AD systems
#     (Enzyme AND Mooncake), ruling out an Enzyme-only implementation bug.
#
# This script isolates ONE level deeper: `misc/doubleDiff.jl::withinTransform`,
# the two-way fixed-effects demeaning transform that IS the entire content of
# `newGravityMoment!`'s UoModel==1 branch (`Wτ = withinTransform(τ)`, `WA =
# withinTransform(Aod)`, `sumGrav = Σ Wτ.*WA`). It is pure, allocating,
# branch-free, mutation-free -- and uses `sum(...; dims=k)`, a well-documented
# historical pain point for Enzyme's activity analysis on reductions. If
# Enzyme NaNs on THIS in total isolation (no project code, no KNITRO, no
# moments!), that is a materially smaller, more upstream-issue-shaped
# reproducer than "the whole gravity moment."
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/minimal_enzyme_nan.jl
# ============================================================================
using ForwardDiff, LinearAlgebra, Random
import Enzyme

withinTransform(z) = begin
    lz = log.(z)
    D = size(z, 1)
    lz .- (sum(lz, dims = 2) ./ D) .- (sum(lz, dims = 1) ./ D) .+ (sum(lz) / D^2)
end

"Scalar reduction matching how newGravityMoment! consumes withinTransform's output: Σ Wa .* Wb."
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

Random.seed!(1)
D = 4
τ = rand(D, D) .+ 0.5
θ0 = vec(rand(D, D) .+ 0.5)

println(">>> primal grav_scalar = ", grav_scalar(θ0, D, τ), "  (finite: ", isfinite(grav_scalar(θ0, D, τ)), ")")

g_fd = ForwardDiff.gradient(θ -> grav_scalar(θ, D, τ), θ0)
println(">>> ForwardDiff gradient: ", g_fd)
println(">>> ForwardDiff finite everywhere: ", all(isfinite, g_fd))

# ---- Step 1: withinTransform ALONE (vector -> vector, sum output as scalar via dot with ones) ----
function within_sum_scalar(θ, D)
    A = reshape(θ, D, D)
    W = withinTransform(A)
    return sum(W)   # should be analytically ~0 by construction, but that's fine -- we want the GRADIENT, not the value
end
g_fd_1 = ForwardDiff.gradient(θ -> within_sum_scalar(θ, D), θ0)
println("\n>>> STEP 1: withinTransform alone, output=sum(W)")
println("    ForwardDiff: ", g_fd_1, "  finite=", all(isfinite, g_fd_1))
g_enz_1 = zeros(length(θ0))
try
    Enzyme.autodiff(Enzyme.Reverse, θ -> within_sum_scalar(θ, D), Enzyme.Active,
                     Enzyme.Duplicated(θ0, g_enz_1))
    println("    Enzyme:      ", g_enz_1, "  finite=", all(isfinite, g_enz_1),
            "  nan_count=", count(isnan, g_enz_1))
catch e
    println("    Enzyme FAILED to compile/run: ", sprint(showerror, e))
end

# ---- Step 2: the full grav_scalar (withinTransform on BOTH τ (Const) and Aod (Active), then Σ Wτ.*WA) ----
println("\n>>> STEP 2: grav_scalar (withinTransform(τ Const) .* withinTransform(Aod Active), summed)")
println("    ForwardDiff: ", g_fd, "  finite=", all(isfinite, g_fd))
g_enz_2 = zeros(length(θ0))
try
    Enzyme.autodiff(Enzyme.Reverse, θ -> grav_scalar(θ, D, τ), Enzyme.Active,
                     Enzyme.Duplicated(θ0, g_enz_2))
    println("    Enzyme:      ", g_enz_2, "  finite=", all(isfinite, g_enz_2),
            "  nan_count=", count(isnan, g_enz_2))
    relerr = norm(g_enz_2 .- g_fd) / max(norm(g_fd), 1e-300)
    println("    relerr vs ForwardDiff (Inf/NaN if any mismatch): ", relerr)
catch e
    println("    Enzyme FAILED to compile/run: ", sprint(showerror, e))
end

# ---- Step 3: isolate JUST the `sum(...; dims=k)` reductions, no log, no broadcast subtraction ----
function dims_sum_scalar(θ, D)
    A = reshape(θ, D, D)
    r = sum(A, dims = 2)   # D x 1
    c = sum(A, dims = 1)   # 1 x D
    g = sum(A)             # scalar
    return sum(r) + sum(c) + g
end
g_fd_3 = ForwardDiff.gradient(θ -> dims_sum_scalar(θ, D), θ0)
println("\n>>> STEP 3: bare sum(;dims=k) reductions only, no log/broadcast-subtract")
println("    ForwardDiff: ", g_fd_3, "  finite=", all(isfinite, g_fd_3))
g_enz_3 = zeros(length(θ0))
try
    Enzyme.autodiff(Enzyme.Reverse, θ -> dims_sum_scalar(θ, D), Enzyme.Active,
                     Enzyme.Duplicated(θ0, g_enz_3))
    println("    Enzyme:      ", g_enz_3, "  finite=", all(isfinite, g_enz_3),
            "  nan_count=", count(isnan, g_enz_3))
catch e
    println("    Enzyme FAILED to compile/run: ", sprint(showerror, e))
end

println("\n================ CONCLUSION ================")
println("If STEP 3 (bare sum(;dims=k)) already NaNs while step-3-equivalent scalar `sum(A)` alone")
println("would not, the culprit is Enzyme's reverse rule for `sum(A; dims=k)` on a Matrix -- a")
println("documented historical weak spot, not anything specific to this project's economics.")
println("If STEP 1 (withinTransform alone) NaNs but STEP 3 (bare dims-sums) does not, the culprit")
println("is the COMBINATION (broadcast subtraction of a dims-reduced array back against the")
println("original, i.e. `lz .- (sum(lz,dims=2)./D)`) -- an implicit broadcast/aliasing interaction,")
println("not the reduction primitive alone.")
