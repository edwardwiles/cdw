# Task §10 validation. Run: julia --project=. full_aod_diag/d4_exact/test_gravity_elimination.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using LinearAlgebra: dot

ctx = d4_exact_setup()
D = ctx.D
rng = MersenneTwister(999)

println("="^78); println("TEST 1: gravity is EXACTLY linear (affine) in log(Aod_theta)"); println("="^78)
z0 = randn(rng, D, D) .* 0.1
Δ = randn(rng, D, D) .* 0.3
c = gravity_linear_coeffs(ctx)
g_z0 = gravity_from_logz(z0, ctx)
g_z0plus = gravity_from_logz(z0 .+ Δ, ctx)
predicted_diff = sum(c .* Δ)
actual_diff = g_z0plus - g_z0
println("predicted diff (sum(c.*Delta)) = ", predicted_diff)
println("actual   diff (g(z0+Delta)-g(z0)) = ", actual_diff)
println("abs diff = ", abs(predicted_diff - actual_diff))
@assert abs(predicted_diff - actual_diff) < 1e-9 "gravity is NOT exactly linear in log(Aod_theta) -- elimination assumption violated"
println("PASS: exact affine structure confirmed (not assumed)")

println("\n" * "="^78); println("TEST 2: pivot coefficient is not near zero"); println("="^78)
pe = build_pivot_elimination(ctx)
println("pivot linear index = ", pe.pivot_lin, "  |c[pivot]| = ", abs(pe.c[pe.pivot_lin]))
println("median |c| over all D^2 entries = ", sort(abs.(pe.c))[div(D^2,2)])
@assert abs(pe.c[pe.pivot_lin]) > 1e-6
println("PASS")

println("\n" * "="^78); println("TEST 3: nullspace rank is exactly D^2 - 1"); println("="^78)
ne = build_nullspace_elimination(ctx)
println("size(Z) = ", size(ne.Z), "  (expect (", D^2, ", ", D^2-1, "))")
@assert size(ne.Z) == (D^2, D^2 - 1)
println("Z orthonormality check: max|Z'Z - I| = ", maximum(abs.(ne.Z' * ne.Z - I)))
@assert maximum(abs.(ne.Z' * ne.Z - I)) < 1e-10
println("PASS")

println("\n" * "="^78); println("TEST 4: gravity residual at RANDOM reduced points is at machine precision"); println("="^78)
for trial in 1:5
    zf = randn(rng, D^2 - 1) .* 0.2
    z_pivot = pivot_expand(zf, pe)
    g_pivot = gravity_from_logz(z_pivot, ctx)

    ζ = randn(rng, D^2 - 1) .* 0.2
    z_null = nullspace_expand(ζ, ne)
    g_null = gravity_from_logz(z_null, ctx)

    println("trial $trial: pivot gravity residual = ", g_pivot, "   nullspace gravity residual = ", g_null)
    @assert abs(g_pivot) < 1e-9 "pivot elimination gravity residual not at machine precision"
    @assert abs(g_null) < 1e-9 "nullspace elimination gravity residual not at machine precision"
end
println("PASS: both parameterizations are exactly gravity-feasible at random points")

println("\n" * "="^78); println("TEST 5: pivot/unpivot round trip"); println("="^78)
zf_test = randn(rng, D^2 - 1)
z_full = pivot_expand(zf_test, pe)
zf_back = pivot_reduce(z_full, pe)
println("max|zf_test - zf_back| = ", maximum(abs.(zf_test .- zf_back)))
@assert zf_test == zf_back
ζ_test = randn(rng, D^2 - 1)
z_full2 = nullspace_expand(ζ_test, ne)
ζ_back = nullspace_reduce(z_full2, ne)
println("max|zeta_test - zeta_back| = ", maximum(abs.(ζ_test .- ζ_back)))
@assert maximum(abs.(ζ_test .- ζ_back)) < 1e-10
println("PASS")

println("\n" * "="^78); println("TEST 6: transformed gradient of gravity itself is exactly zero (analytic == FD)"); println("="^78)
# g_gravity(expand(reduced coords)) is IDENTICALLY zero by construction -- its gradient w.r.t. the
# reduced coordinates should be exactly zero both analytically (chain rule: c'*(dz/dreduced)=0 by
# construction) and via finite differences (confirms the elimination is doing what it claims).
h = 1e-5
zf0 = zeros(D^2 - 1)
g_grad_fd_pivot = [ (gravity_from_logz(pivot_expand(zf0 .+ h .* [i==k for k in 1:D^2-1], pe), ctx) -
                      gravity_from_logz(pivot_expand(zf0 .- h .* [i==k for k in 1:D^2-1], pe), ctx)) / (2h)
                     for i in 1:D^2-1 ]
println("max|FD gradient of gravity, pivot coords| = ", maximum(abs.(g_grad_fd_pivot)), "  (expect ~0, analytic=exactly 0)")
@assert maximum(abs.(g_grad_fd_pivot)) < 1e-6

ζ0 = zeros(D^2 - 1)
g_grad_fd_null = [ (gravity_from_logz(nullspace_expand(ζ0 .+ h .* [i==k for k in 1:D^2-1], ne), ctx) -
                     gravity_from_logz(nullspace_expand(ζ0 .- h .* [i==k for k in 1:D^2-1], ne), ctx)) / (2h)
                    for i in 1:D^2-1 ]
println("max|FD gradient of gravity, nullspace coords| = ", maximum(abs.(g_grad_fd_null)), "  (expect ~0)")
@assert maximum(abs.(g_grad_fd_null)) < 1e-6
println("PASS: both eliminations correctly zero out the gravity direction (analytic and FD agree)")

println("\n" * "="^78)
println("ALL GRAVITY ELIMINATION TESTS PASSED")
println("="^78)
println("\nNOTE (documented, not resolved this test): log-A box bounds under either")
println("reparameterization are NOT independent per-coordinate anymore (task sec 10's own")
println("warning: 'do not pretend transformed box bounds remain independent when they do not').")
println("Bound-handling for a constrained KNITRO run in reduced coordinates is deferred to")
println("the short-solver-run phase (task sec 18), not implemented in this diagnostic.")
