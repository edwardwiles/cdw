# Smoke test for c8_nestedw_context.jl -- NOT a deliverable, just a sanity check
# run once before trusting the nested-pool machinery for the real grid.
include(joinpath(@__DIR__, "c8_nestedw_context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

println("pool size: ", size(NESTEDW_POOL))
println("pool[1:5,1] = ", NESTEDW_POOL[1:5, 1])

ctx8 = build_nested_ctx(8000; find_smallest = true)
ctx20 = build_nested_ctx(20000; find_smallest = true)
ctx80 = build_nested_ctx(80000; find_smallest = true)

# nesting check: ctx8's U must equal the first 8000 rows of ctx20's U and ctx80's U
println("nesting check (8000 vs 20000 prefix): ", ctx8.U == ctx20.U[1:8000, :])
println("nesting check (8000 vs 80000 prefix): ", ctx8.U == ctx80.U[1:8000, :])
println("nesting check (20000 vs 80000 prefix): ", ctx20.U == ctx80.U[1:20000, :])

pe = build_pivot_elimination(ctx8)
w_lfixcomposite_sr1 = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf = x_free_from_w(w_lfixcomposite_sr1)

for (label, ctx) in (("W=8000", ctx8), ("W=20000", ctx20), ("W=80000", ctx80))
    t0 = time()
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)
    dt = time() - t0
    kappa = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    println("[$label] kappa=$kappa Delta_dual=$(r.Delta_dual) Delta-delta=$(r.Delta_minus_delta) inner_status=$(r.inner_status) elapsed=$(round(dt,digits=3))s")
end
println("SMOKE TEST DONE")
