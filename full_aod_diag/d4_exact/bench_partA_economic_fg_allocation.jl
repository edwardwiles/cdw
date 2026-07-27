# Addendum Part A remediation (2026-07-26): direct microbenchmark, old allocating
# compressed_cc_value_grad vs new in-place compressed_cc_value_grad!, at real D=20/W=80,000 scale.
# Both functions are still present (old kept unchanged as reference for dual_bank.jl/theta_cplus.jl/
# benchmarks) -- this measures the SAME mathematical computation, old vs new call path.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "compressed_moments.jl",
          "structured_moment_build.jl", "compressed_cc_inner.jl", "oracle_fast.jl", "compressed_live.jl",
          "draw_design.jl"]
    include(joinpath(D4X, f))
end
using Printf, Statistics, Random

ctx = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
θ_full = ctx.θ0_up

cf = build_compressed_factual(θ_full, ctx; check_ties = true)
obj = ctx.obj
ncol = cf.oci - 1
λ = 0.001 .* randn(ncol)
ζ = 0.01
g_out = zeros(ncol)
ws = EconomicFGWorkspace(cf)

# warm-up (JIT)
compressed_cc_value_grad(ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
compressed_cc_value_grad!(ws, g_out, ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)

NREP = 200
old_allocs = Int[]; new_allocs = Int[]
old_times = Float64[]; new_times = Float64[]
for i in 1:NREP
    t0 = time(); b = @allocated (f1, gz1, gl1, q1, _) = compressed_cc_value_grad(ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    push!(old_times, time() - t0); push!(old_allocs, b)
end
for i in 1:NREP
    t0 = time(); b = @allocated (f2, gz2) = compressed_cc_value_grad!(ws, g_out, ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    push!(new_times, time() - t0); push!(new_allocs, b)
end

f1, gz1, gl1, q1, _ = compressed_cc_value_grad(ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
f2, gz2 = compressed_cc_value_grad!(ws, g_out, ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
@printf "agreement: f diff=%.3e  g_zeta diff=%.3e  g_lambda maxdiff=%.3e  q maxdiff=%.3e\n" abs(f1-f2) abs(gz1-gz2) maximum(abs.(gl1.-g_out)) maximum(abs.(q1.-ws.q))

@printf "\n==================== PART A ECONOMIC FG ALLOCATION MICROBENCHMARK (D=20, W=80000, per-call) ====================\n"
@printf "old (compressed_cc_value_grad)  : median_bytes=%.0f (%.1fKB)  median_time=%.3es\n" median(old_allocs) median(old_allocs)/1e3 median(old_times)
@printf "new (compressed_cc_value_grad!) : median_bytes=%.0f (%.1fKB)  median_time=%.3es\n" median(new_allocs) median(new_allocs)/1e3 median(new_times)
@printf "allocation reduction: %.1fx  (%.1f%% of original)\n" median(old_allocs)/max(1,median(new_allocs)) 100*median(new_allocs)/median(old_allocs)
@printf "time speedup: %.3fx\n" median(old_times)/median(new_times)
println(">>> BENCH_DONE")
