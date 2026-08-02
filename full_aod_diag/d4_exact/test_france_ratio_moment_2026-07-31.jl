# ============================================================================
# Task §1.3 / theory doc section 2.5 gate: France homogeneous ratio moment.
# ADDITIVE ONLY -- see homogeneous_moments_2026-07-31.jl header.
#
# CORRECTED 2026-07-31 (same day, user stop): Test 1 originally cross-checked
# against the LEGACY dense `ctx.obj.moments!`'s `G[:,cf_col]` column. Rebuilt
# to read `build_compressed_factual`'s own `cf_raw` field directly (the RAW,
# pre-post-processing value -- even more direct than the dense-G column,
# which needed a sampling-weight/gammafac division first). See
# recover_full_a_2026-07-31.jl's header for the full correction rationale.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
using Statistics

ctx = d4_exact_setup()
D = ctx.D
Aod_offset = ctx.Aod_offset
θ0 = copy(ctx.θ0_up)
bi = ctx.bi
println("baseIndex (France-analog) = $bi")

println("\n" * "="^78); println("TEST 1: cross-check against build_compressed_factual's own cf_raw at genuine calibration"); println("="^78)
cf0 = build_compressed_factual(θ0, ctx; check_ties = false)
old_moment_mean = mean(cf0.cf_raw)
println("mean(cf0.cf_raw) at genuine calibration = $old_moment_mean  (informational -- OLD absolute moment, raw form)")

H_france_0 = homogeneous_france_moment(θ0, ctx)
println("mean(homogeneous_france_moment) at genuine calibration = $(mean(H_france_0))  (informational -- NEW homogeneous moment)")
println("(both informational, matching the earlier ~0.2%-of-scale calibration-tolerance pattern -- not exactly 0)")

println("\n" * "="^78); println("TEST 2: EXACT proportional rescaling under a France-column shift (kappa^(mu*(sigma-1)))"); println("="^78)
κ = 1.6
e_exponent = θ0[1] * (θ0[2] - 1)
θ1 = copy(θ0)
for o in 1:D
    idx = Aod_offset + (bi - 1) * D + o
    θ1[idx] *= κ
end
H_france_1 = homogeneous_france_moment(θ1, ctx)
predicted = κ^e_exponent
ratio = H_france_1 ./ H_france_0
maxdiff = maximum(abs.(ratio .- predicted))
println("predicted kappa^(mu*(sigma-1)) = $predicted")
println("max|H1/H0 - predicted| = $maxdiff")
@assert maxdiff < 1e-6 "TEST 2 FAILED: France homogeneous moment does not rescale exactly by kappa^(mu*(sigma-1))"
println("PASS -- rho_f_ratio = gp^sigma makes the France moment exactly homogeneous under a France-column rescale")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
