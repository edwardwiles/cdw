# ============================================================================
# Part 7 driver: genuinely sample-exact full-(D+2) Frechet negative control.
#
#   DVAL=4 WVAL=8000 julia --project=. sequential_gravity/derivative_diagnostics/run_full_d2_sample_exact.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf, Random, LinearAlgebra, Statistics

println("\n" * "="^78); println(">>> PART 7: sample-exact full-(D+2) negative control at theta_ref = Frechet benchmark"); println("="^78)

frozen_ref = freeze_gravity_linearization(θr0, seq_gravcol, grad_R_theta)
@printf("theta_ref: R_mean=%.4e Rcol=%.4e ok=%s\n", frozen_ref.R, frozen_ref.Rcol, frozen_ref.ok)
frozen_moments = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen_ref.θ, frozen_ref.Rcol, frozen_ref.gcol, frozen_ref.dRdθ)

sx_moments = make_sample_exact_moments(frozen_moments, D + 2, θr0, U, γ)

# Sanity: verify uniform p=1/W exactly satisfies every moment at theta_ref.
Ktest = zeros(W); Gtest = zeros(W, D + 2)
sx_moments(Ktest, Gtest, θr0, U, (γ=γ,))
maxmean = maximum(abs, sum(Gtest, dims=1) ./ W)
@printf("max|sample mean of G| at theta_ref (should be ~0 to float precision): %.3e\n", maxmean)
@assert maxmean < 1e-10 "sample-exact recentering failed -- this is a bug, not a numerical-precision issue"

# test_full_fixed_dual_identity's own wrapping (make_frozen_gravity_moments around trade_moments_fn!)
# would double-recenter the gravity column if used here -- sx_moments ALREADY produces the full D+2
# columns (it wraps the already-frozen frozen_moments), so call test_fixed_dual_identity directly.
idr = test_fixed_dual_identity(θr0, sx_moments, D + 2, γ, U; l=length(θr0), find_smallest=true)
@printf("delta*(A*,gamma'_frechet) under sample-exact targets = %.6e  (should be ~0 up to solver tol)  nStatus=%d\n", idr.δ_star, idr.nStatus)

# ============================================================================
# Perturb A away from A* (theta_ref) at several step sizes/directions and
# confirm: (a) delta* >= 0 always, (b) delta* grows roughly monotonically
# with |perturbation|, establishing the numerical floor for genuine descent.
# ============================================================================
println("\n" * "="^78); println(">>> Perturbation scan: delta*(A) - delta*(A*) for A near A*"); println("="^78)

const Acol_offset = 3
Random.seed!(20260714)
dirs = Vector{Vector{Float64}}()
for o in 1:D
    v = zeros(D); v[o] = 1.0; push!(dirs, v)
end
for _ in 1:6
    v = randn(D); v ./= norm(v); push!(dirs, v)
end
hs = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]

results = NamedTuple[]
n_negative = 0
for (di, v) in enumerate(dirs)
    for h in hs
        θp = copy(θr0)
        @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
        obj_p = build_fixed_dual_bundle(γ, U, length(θp), D + 2, sx_moments; find_smallest=true)
        δp, _, statusp = inner_loop(obj_p, θp)
        δp < -1e-9 && (n_negative += 1)
        push!(results, (dir=di, h=h, δ=δp, status=statusp))
    end
end
@printf("scanned %d (direction,h) points around A*; negative divergences found: %d (should be 0)\n", length(results), n_negative)
min_δ = minimum(r.δ for r in results if r.status == 0)
@printf("min observed delta* over the scan: %.3e (this + the theta_ref delta*=%.3e IS the numerical floor)\n", min_δ, idr.δ_star)

@printf("\n%6s %8s %14s %8s\n", "dir", "h", "delta*", "status")
for r in results
    @printf("%6d %8.0e %14.6e %8d\n", r.dir, r.h, r.δ, r.status)
end

println("\nPART 7 sample-exact negative control DONE")
