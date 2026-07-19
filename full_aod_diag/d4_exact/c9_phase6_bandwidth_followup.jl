# ============================================================================
# Continuation 9, Phase 6 follow-up: cheap bandwidth-sensitivity check at the
# ONE point (calibration) where the main c9_phase6_gradval_d20.jl run found
# poor A-block agreement (cosine 0.41-0.73, sign 70-85%). Per the standing
# brief's "pursue ONE follow-up if the base result is clearly poor" allowance
# -- picks "test a different L_fix bandwidth" (the cheapest option: reuses
# the SAME 20 directions and the ALREADY-COMPUTED optimized-value central
# secants from the main run's direction_secants.csv, so this costs only 2
# extra composite_gradient_at_fast calls, no new evaluate_fullA solves).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra, DelimitedFiles

const RESULTS_DIR = ARGS[1]   # the main run's OUTDIR, containing direction_secants.csv

ctx = d20_real_setup(W = 80000)
D = ctx.D; D2 = D^2
pe = build_pivot_elimination(ctx)
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]

Random.seed!(20260719)
dirs = Vector{Vector{Float64}}()
for i in 1:20
    v = randn(D2 - 1); v ./= norm(v)
    push!(dirs, v)
end

base0 = solve_base_state(xf_nat, ctx)

# baseline (matches the main run): h_mode=:cached (== :adaptive on first use)
g_adapt, _ = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true,
                                         h_mode = :cached, bandwidth_cache = Dict{Int,Float64}(),
                                         multi_method = :top3)
g_adapt_A = g_adapt[2:end]

# follow-up 1: larger fixed bandwidth (h0=0.05, closer to select_bandwidth's own h_ceil=0.1)
g_h05, _ = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true,
                                       h_mode = :fixed, h0 = 0.05, multi_method = :top3)
g_h05_A = g_h05[2:end]

# follow-up 2: smaller fixed bandwidth (h0=0.005, matching the small secant probe bandwidth)
g_h005, _ = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true,
                                        h_mode = :fixed, h0 = 0.005, multi_method = :top3)
g_h005_A = g_h005[2:end]

# follow-up 3: default fixed bandwidth (h0=0.01, the historical FIXED_H baseline)
g_h01, _ = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true,
                                       h_mode = :fixed, h0 = 0.01, multi_method = :top3)
g_h01_A = g_h01[2:end]

# load the ALREADY-COMPUTED central secants for calibration from the main run (both h columns)
raw, hdr = readdlm(joinpath(RESULTS_DIR, "direction_secants.csv"), ',', header = true)
cols = Dict(String(hdr[i]) => i for i in 1:length(hdr))
calib_rows_02 = [r for r in eachrow(raw) if r[cols["point"]] == "calibration" && abs(r[cols["h"]] - 0.02) < 1e-9]
calib_rows_005 = [r for r in eachrow(raw) if r[cols["point"]] == "calibration" && abs(r[cols["h"]] - 0.005) < 1e-9]
@assert length(calib_rows_02) == 20 && length(calib_rows_005) == 20

# order by dir_idx to align with `dirs`
sort!(calib_rows_02; by = r -> r[cols["dir_idx"]])
sort!(calib_rows_005; by = r -> r[cols["dir_idx"]])
central_02 = Float64[r[cols["central_secant"]] for r in calib_rows_02]
central_005 = Float64[r[cols["central_secant"]] for r in calib_rows_005]

cossim(a, b) = dot(a, b) / max(norm(a) * norm(b), 1e-300)
signfrac(a, b) = count(sign.(a) .== sign.(b)) / length(a)

function preds_for(g_A)
    [dot(g_A, v) for v in dirs]
end

println("="^90)
println("Calibration point, A-block bandwidth follow-up (reusing the 20 existing directions)")
println("="^90)
for (label, gA) in [("cached/adaptive (main run baseline)", g_adapt_A),
                     ("fixed h0=0.05", g_h05_A),
                     ("fixed h0=0.01", g_h01_A),
                     ("fixed h0=0.005", g_h005_A)]
    p = preds_for(gA)
    cs02 = cossim(p, central_02); sf02 = signfrac(p, central_02)
    cs005 = cossim(p, central_005); sf005 = signfrac(p, central_005)
    @printf("%-36s  vs secant(h=0.02): cos=%.4f sign=%.2f   vs secant(h=0.005): cos=%.4f sign=%.2f   ||g_A||=%.5f\n",
            label, cs02, sf02, cs005, sf005, norm(gA))
end
println("DONE")
