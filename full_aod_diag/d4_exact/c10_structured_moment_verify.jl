# ============================================================================
# Continuation 10, Part 2: correctness verification for structured_moment_build.jl
#
# Three checks, all against the ACTUAL CODE (not derivation alone):
#   A. structured_dense_factual(cf) vs the TRUSTED DENSE constructor
#      (obj.moments! = EK_moments_gammanorm_directgp!, the production dense
#      moment builder) -- direct comparison of the bilateral + counterfactual
#      columns, at D=4 (synthetic) first.
#   B. structured_dense_factual(cf) vs materialize_dense_factual(cf) (the
#      EXISTING compressed-then-materialize reference, already trusted per
#      docs/fullA_fully_compressed_inner_report.md) -- should be bit-identical
#      (same formula, just reorganized into rank-one + scatter).
#   C. Tie handling: structured_fill_chunk! is built ON TOP OF `cf` (from
#      `build_compressed_factual`), so it NEVER re-implements winner-search or
#      tie-detection -- any tie throws TiedWinnerError before reaching the
#      structured fill, by construction. Demonstrated empirically here by
#      forcing a synthetic exact price tie (mutating ctx.U so two origins tie
#      for a specific draw/destination) and confirming TiedWinnerError fires.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
using Printf

println("="^90)
println("Part A+B: D=4 -- structured vs dense (obj.moments!) vs materialize_dense_factual!")
println("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
obj4 = ctx4.obj
xf4 = ctx4.θ0_up[ctx4.free_idx]
θ4 = CS.reconstruct_full(xf4, ctx4.m)

# ---- (a) trusted dense construction: obj.moments! ----
W4 = size(obj4.U, 1)
K_dense = zeros(W4)
G_dense = zeros(W4, obj4.d)
obj4.moments!(K_dense, G_dense, θ4, obj4.U, obj4)
ncol4 = obj4.outer_constr_index - 1
G_dense_inner = G_dense[:, 1:ncol4]

# ---- build compressed representation (winner search + wval, shared by (b) and (c)) ----
cf4 = build_compressed_factual(θ4, ctx4; check_ties = true)
println("D4: n_tied=", cf4.n_tied, "  oci=", cf4.oci, "  ncol(inner)=", ncol4)

# ---- (b) existing materialize_dense_factual! reference ----
G_matref = materialize_dense_factual(cf4)

# ---- (c) NEW structured construction, both ger! and broadcast variants ----
G_struct_ger = structured_dense_factual(cf4; use_ger = true)
G_struct_bcast = structured_dense_factual(cf4; use_ger = false)

d_struct_vs_dense = maximum(abs.(G_struct_ger .- G_dense_inner))
d_struct_vs_matref = maximum(abs.(G_struct_ger .- G_matref))
d_matref_vs_dense = maximum(abs.(G_matref .- G_dense_inner))
d_ger_vs_bcast = maximum(abs.(G_struct_ger .- G_struct_bcast))

@printf("  max|structured(ger) - dense(obj.moments!)|      = %.3e  (bit_identical=%s)\n", d_struct_vs_dense, G_struct_ger == G_dense_inner)
@printf("  max|structured(ger) - materialize_dense_factual| = %.3e  (bit_identical=%s)\n", d_struct_vs_matref, G_struct_ger == G_matref)
@printf("  max|materialize_dense_factual - dense(obj.moments!)| = %.3e  (reference gap, pre-existing)\n", d_matref_vs_dense)
@printf("  max|structured(ger) - structured(broadcast)|     = %.3e  (bit_identical=%s)\n", d_ger_vs_bcast, G_struct_ger == G_struct_bcast)

d_struct_vs_dense < 1e-9 || error("Part A FAILED: structured construction does not match dense obj.moments! (max diff $d_struct_vs_dense)")
d_struct_vs_matref < 1e-12 || error("Part B FAILED: structured construction does not match materialize_dense_factual! (max diff $d_struct_vs_matref)")
d_ger_vs_bcast < 1e-12 || error("ger! vs broadcast variants disagree (max diff $d_ger_vs_bcast)")

println("\nParts A/B PASSED at D=4.")

println("\n", "="^90)
println("Part C: tie handling -- synthetic exact price tie forces TiedWinnerError")
println("="^90)

# Force an exact tie: pick draw s=1, destination d=1, origins o=1,2. Set U so that
# constCons[o,d]/UPow[s,o] is IDENTICAL for o=1,2 at (s=1,d=1). We approximate by
# just setting U[1, 1] = U[1, 2] AND checking whether that alone creates a tie
# (it will iff constCons[1,1]==constCons[2,1], which is not guaranteed by real data,
# so instead we directly search for a scaling that forces it: since price = constCons[o,d]/UPow[s,o]
# with UPow = U^(-mu), setting U[s,o2] = U[s,o1] * (constCons[o2,d]/constCons[o1,d])^(1/mu)
# forces price[o1] == price[o2] exactly at that (s,d).
μ4 = θ4[1]
γo4 = ctx4.γ
D4 = ctx4.D
Aod_θ4 = reshape(θ4[ctx4.Aod_offset+1:ctx4.Aod_offset+D4^2], (D4, D4))
lambda4 = reshape(γo4.P, (D4, D4))'
Aod4 = Aod_θ4 .* γo4.cHat .* (((γo4.wHat .* γo4.τ) ./ (γo4.wHat[1,1] .* γo4.τ[1,:]')) .^ (1/μ4)) .* (lambda4 ./ lambda4[1,:]')
AodPow4 = (Aod4 ./ γo4.cHat) .^ (-μ4)
constCons4 = [γo4.wHat[o] * AodPow4[o,d] * γo4.τ[o,d] for o in 1:D4, d in 1:D4]

Ubad = copy(ctx4.U)
s_tie, d_tie, o1, o2 = 1, 1, 1, 2
ratio = (constCons4[o1, d_tie] / constCons4[o2, d_tie])^(1/μ4)
Ubad[s_tie, o2] = Ubad[s_tie, o1] * ratio
# sanity: recompute prices directly
UPow_bad = Ubad .^ (-μ4)
price_o1 = constCons4[o1, d_tie] / UPow_bad[s_tie, o1]
price_o2 = constCons4[o2, d_tie] / UPow_bad[s_tie, o2]
@printf("  forced prices: origin %d -> %.15f , origin %d -> %.15f (should match)\n", o1, price_o1, o2, price_o2)
abs(price_o1 - price_o2) < 1e-9 || error("failed to construct a synthetic tie -- price mismatch too large")

θ4b = copy(θ4)   # tie construction only touches U, not θ -- ctx4b reuses θ4
ctx4b = merge(ctx4, (U = Ubad,))
function check_tie_throws(θ4b, ctx4b)
    try
        build_compressed_factual(θ4b, ctx4b; check_ties = true)
    catch e
        if e isa TiedWinnerError
            println("  build_compressed_factual correctly threw TiedWinnerError: ", sprint(showerror, e)[1:min(120,end)], "...")
            return true
        else
            rethrow(e)
        end
    end
    return false
end
threw = check_tie_throws(θ4b, ctx4b)
threw || error("Part C FAILED: build_compressed_factual did NOT throw TiedWinnerError on a forced exact tie")

println("\nPart C PASSED: tie detection fires correctly; since structured_fill_chunk! consumes the")
println("SAME `cf` object build_compressed_factual produces (never re-implementing winner search),")
println("any tie that would affect materialize_dense_factual! affects the structured construction")
println("identically and at the identical point (both are downstream of the SAME check).")

println("\nAll Part 2 correctness checks PASSED (D=4).")
