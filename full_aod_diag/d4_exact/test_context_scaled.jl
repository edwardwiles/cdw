# Sanity check: d_exact_setup_scaled(D=4,W=8000) should reproduce d4_exact_setup()'s
# structural quantities (n_free, nTotalMoments, l_full, bounds) before trusting it at other D/W.
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))

ctx_ref = d4_exact_setup()
ctx_scaled = d_exact_setup_scaled(D = 4, W = 8000)

println("="^78); println("d_exact_setup_scaled(D=4,W=8000) vs d4_exact_setup() structural check"); println("="^78)
checks = [
    ("D", ctx_ref.D, ctx_scaled.D),
    ("nTotalMoments", ctx_ref.nTotalMoments, ctx_scaled.nTotalMoments),
    ("l_full", ctx_ref.l_full, ctx_scaled.l_full),
    ("n_free", CS.n_free(ctx_ref.m), CS.n_free(ctx_scaled.m)),
    ("Aod_offset", ctx_ref.Aod_offset, ctx_scaled.Aod_offset),
    ("W(U rows)", size(ctx_ref.U,1), size(ctx_scaled.U,1)),
]
all_ok = true
for (name, a, b) in checks
    ok = a == b
    ok || (all_ok = false)
    println("  $name: ref=$a scaled=$b  ", ok ? "OK" : "MISMATCH")
end

# a genuine exact evaluation at each ctx's own theta0 should succeed cleanly (not bit-identical
# across ctx's since each has independent random draws -- structural validity only)
x0_ref = CS.pack_free(ctx_ref.θ0_up, ctx_ref.m)
x0_scaled = CS.pack_free(ctx_scaled.θ0_up, ctx_scaled.m)
r_ref = evaluate_fullA(x0_ref, ctx_ref; cache = nothing, warm = false)
r_scaled = evaluate_fullA(x0_scaled, ctx_scaled; cache = nothing, warm = false)
println("\nref eval: inner_status=$(r_ref.inner_status)  gravity=$(r_ref.gravity_value)")
println("scaled eval: inner_status=$(r_scaled.inner_status)  gravity=$(r_scaled.gravity_value)")
both_solved = r_ref.inner_status in (0,-100,-101,-103) && r_scaled.inner_status in (0,-100,-101,-103)
both_gravity_ok = abs(r_ref.gravity_value) < 1e-6 && abs(r_scaled.gravity_value) < 1e-6

println("\nSTRUCTURAL MATCH: ", all_ok, "  BOTH SOLVE CLEANLY: ", both_solved, "  BOTH GRAVITY OK: ", both_gravity_ok)
(all_ok && both_solved && both_gravity_ok) || error("d_exact_setup_scaled does not reproduce d4_exact_setup's structure at D=4,W=8000 -- do not trust it at other D/W")
println("\nPASS -- d_exact_setup_scaled is trustworthy for the D/W scaling benchmark")
