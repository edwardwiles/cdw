# ============================================================================
# Step 5 of the reverse-mode NaN bisection -- and the most promising lead so
# far. Steps 1-4 (sum(;dims=k), the demeaning broadcast, the dead doubleDiff
# preamble) all matched ForwardDiff exactly under Enzyme AND Mooncake. This
# step targets a STRUCTURALLY DIFFERENT pattern found by reading
# moments/moments!.jl directly (not previously isolated):
#
#   moments!.jl:130-138 (its own comment, verbatim):
#     "reuse preallocated Float64 scratch on the Float64 path; allocate Duals
#      under ForwardDiff"
#     if eltype(γ) === Float64 && size(UPow_scratch,1)==size(U,1)
#         UPow = UPow_scratch        # <- shared, mutable, externally-owned buffer
#     else
#         UPow = zeros(eltype(γ), size(U))   # <- fresh allocation
#     end
#     ... UPow .= U .^ (-μ) ...      # <- WRITES an Active(μ)-dependent value
#                                        into UPow, then G/K read UPow back.
#
# `γ` here is `copy(θ[3:2+D])` (moments!.jl:86) -- a direct function of θ. So:
#   - under ForwardDiff, eltype(γ) is Dual, NOT Float64 -> ALWAYS takes the
#     `else` (fresh-allocation) branch. ForwardDiff NEVER exercises the
#     scratch-reuse branch.
#   - under plain (non-differentiated) Float64 evaluation, eltype(γ)===Float64
#     -> TRUE -> reuses UPow_scratch. This is the intended, tested case.
#   - under ENZYME REVERSE MODE, θ is NOT promoted to a different element
#     type (Enzyme differentiates the actual Float64 code via a shadow/tape,
#     not dual numbers) -> eltype(γ)===Float64 is ALSO TRUE -> Enzyme's
#     differentiated pass takes the EXACT SAME scratch-reuse branch as plain
#     evaluation -- a branch that was never designed or tested to be
#     differentiated.
#
# Compounding this: ../ad_benchmark's Enzyme calls mark the whole `ctx`
# (containing γobj, and therefore UPow_scratch) as `Const` (required, since
# ctx also holds fixed data). But the primal code then WRITES into
# UPow_scratch a value that depends on μ (an Active/Duplicated component of
# θ), and reads it back to help compute the differentiated scalar output.
# Mutating a `Const`-marked object with an Active-dependent value, then using
# that mutated value in the thing being differentiated, is a textbook Enzyme
# activity-violation pattern (no shadow memory exists for a Const object) --
# a structurally different, and much better-motivated, candidate than
# anything tested in steps 1-4.
#
# This script builds a minimal, standalone analogue: a Const-marked context
# holding a mutable Float64 scratch buffer; a primal function that (like
# moments!.jl) branches on eltype(x)===Float64 to decide whether to reuse
# that external buffer or allocate fresh, writes an Active(mu)-dependent
# value into it, reads it back into the output. No project code loaded.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/minimal_enzyme_nan_step5_scratch.jl
# ============================================================================
using ForwardDiff, LinearAlgebra, Random
import Enzyme

mutable struct ScratchCtx
    U::Matrix{Float64}
    scratch::Matrix{Float64}   # mutable, externally-owned, reused across calls -- like UPow_scratch
end

"""
Mirrors moments!.jl:130-145's structure exactly:
  - branch on eltype(mu_and_extra) === Float64 to decide fresh-vs-reuse (mu_and_extra
    plays the role of `γ`, a vector built from theta, so its eltype tracks theta's type
    under ForwardDiff but NOT under Enzyme)
  - write U.^(-mu) into whichever buffer was chosen (an Active(mu)-dependent write)
  - read it back to build the scalar output
"""
function f_scratch(theta, ctx::ScratchCtx)
    mu = theta[1]
    extra = theta[1:1] .* 1.0   # a tiny "γ"-like vector whose eltype tracks theta's type under ForwardDiff
    if eltype(extra) === Float64 && size(ctx.scratch) == size(ctx.U)
        UPow = ctx.scratch                      # reuse external mutable buffer (the risky branch)
    else
        UPow = zeros(eltype(extra), size(ctx.U))  # fresh allocation (ForwardDiff's actual path)
    end
    @. UPow = ctx.U ^ (-mu)
    return sum(UPow)
end

Random.seed!(7)
U = rand(20, 5) .+ 0.5
ctx = ScratchCtx(U, zeros(size(U)))
theta0 = [0.2]

g_fd = ForwardDiff.gradient(θ -> f_scratch(θ, ctx), theta0)
println(">>> ForwardDiff: ", g_fd, "  finite=", all(isfinite, g_fd))

# call the primal once first (as production does before ever differentiating -- populates ctx.scratch)
f_scratch(theta0, ctx)

g_enz = zeros(1)
try
    Enzyme.autodiff(Enzyme.Reverse, θ -> f_scratch(θ, ctx), Enzyme.Active,
                     Enzyme.Duplicated(theta0, g_enz))
    println(">>> Enzyme (ctx implicitly captured, no Const annotation): ", g_enz,
            "  finite=", all(isfinite, g_enz))
catch e
    println(">>> Enzyme (implicit capture) FAILED: ", sprint(showerror, e)[1:min(300,end)])
end

# now the REALISTIC production pattern: ctx explicitly marked Const (required, since in the
# real code ctx also carries fixed non-differentiable data alongside the scratch buffer)
g_enz2 = zeros(1)
function f_scratch_arg(theta, ctx)
    f_scratch(theta, ctx)
end
try
    Enzyme.autodiff(Enzyme.Reverse, f_scratch_arg, Enzyme.Active,
                     Enzyme.Duplicated(theta0, g_enz2), Enzyme.Const(ctx))
    nnan = count(isnan, g_enz2)
    println(">>> Enzyme (ctx EXPLICITLY Const, matching production's actual annotation pattern): ",
            g_enz2, "  finite=", all(isfinite, g_enz2), "  nan_count=", nnan)
    if nnan > 0 || !isapprox(g_enz2, g_fd; rtol=1e-8)
        println(">>> REPRODUCED (or mismatched): Const-marked mutable scratch buffer, written with an")
        println(">>> Active-dependent value and read back, breaks Enzyme's reverse pass.")
    else
        println(">>> Still matches ForwardDiff here -- try TWO calls sharing ctx.scratch (below).")
    end
catch e
    println(">>> Enzyme (Const ctx) FAILED to compile/run: ", sprint(showerror, e))
end

# stress the REUSE aspect specifically: two DIFFERENT theta points, same persistent ctx.scratch,
# differentiate the SECOND call after the scratch buffer already holds stale data from the FIRST
# (this is exactly what happens across repeated inner-loop/moments! calls sharing one obj.γ in production)
theta1 = [0.35]
f_scratch(theta1, ctx)              # primal call #1 populates ctx.scratch with theta1's values
g_fd2 = ForwardDiff.gradient(θ -> f_scratch(θ, ctx), theta0)   # reference at theta0, AFTER scratch was touched by theta1
g_enz3 = zeros(1)
try
    Enzyme.autodiff(Enzyme.Reverse, f_scratch_arg, Enzyme.Active,
                     Enzyme.Duplicated(theta0, g_enz3), Enzyme.Const(ctx))
    nnan3 = count(isnan, g_enz3)
    println("\n>>> STALE-SCRATCH TEST: scratch buffer touched by theta1=$(theta1) BEFORE differentiating at theta0=$(theta0)")
    println("    ForwardDiff (fresh-alloc path, unaffected by staleness): ", g_fd2)
    println("    Enzyme (reuses the now-stale-then-overwritten scratch):  ", g_enz3,
            "  finite=", all(isfinite, g_enz3), "  nan_count=", nnan3)
catch e
    println(">>> STALE-SCRATCH TEST FAILED: ", sprint(showerror, e))
end
