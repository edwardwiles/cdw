# Real D=20 post-omit-ROW gates for the addendum's unified fixed-theta a-space architecture
# (addendum §5). D_origin=20, D_dest=19, W=80,000, seed 20260719, 20 threads, BLAS threads=1.
using Random, LinearAlgebra, Printf

include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool; note::AbstractString = "")
    status = cond ? "PASS" : "FAIL"
    lp(rpad(status, 6), name, note == "" ? "" : "  ($note)")
    cond || push!(FAILURES, name)
end

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx.μHat
D = ctx.D; Ddest = ctx.D_dest
rsc = build_ranged_screen_context(ctx)
xy = precompute_aspace_XY(ctx)
pe_legacy = build_pivot_elimination(ctx)
pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)

x_free_calib = ctx.θ0_up[ctx.free_idx]
gp0 = x_free_calib[1]
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

lp(">>> D20 unified layout gates: theta_star=", theta_star, " D=", D, " Ddest=", Ddest)

lp(">>> Gate 1: fixed+legacy_z+raw reproduces EXISTING production driver's own numbers")
layout_z = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
w_legacy = vcat(gp0, pivot_reduce(logA_full0, pe_legacy))
d_z = decode_outer_unified(w_legacy, ctx, layout_z, pgc, xy)
xf_ref = x_free_from_w(w_legacy, pe_legacy)
check("xf agreement (unified fixed+legacy_z vs existing x_free_from_w)", isapprox(d_z.xf, xf_ref; rtol = 1e-6);
      note = "max|diff|=$(maximum(abs.(d_z.xf .- xf_ref)))")

r_unified, _ = screened_eval(d_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
r_legacy, _ = screened_eval(xf_ref, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
check("Delta_dual agreement (unified fixed+legacy_z vs existing driver)", isapprox(r_unified.Delta_dual, r_legacy.Delta_dual; rtol = 1e-6);
      note = "unified=$(r_unified.Delta_dual) legacy=$(r_legacy.Delta_dual)")
check("both feasible", r_unified.inner_status in FEASIBLE_CODES && r_legacy.inner_status in FEASIBLE_CODES)

lp(">>> Gate 2: fixed+powered_aspace+raw reproduces the SAME economic point")
layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
w_a = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_a)
d_a = decode_outer_unified(w_a, ctx, layout_a, pgc, xy)
check("xf agreement (a-space vs z-space at fixed theta_star)", isapprox(d_a.xf, xf_ref; rtol = 1e-6);
      note = "max|diff|=$(maximum(abs.(d_a.xf .- xf_ref)))")
r_a, _ = screened_eval(d_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
check("a-space feasible at real D=20 calibration point", r_a.inner_status in FEASIBLE_CODES)
check("a-space gravity residual < 1e-6", abs(r_a.gravity_value) < 1e-6; note = "gravity=$(r_a.gravity_value)")
check("a-space Delta_dual matches z-space/legacy", isapprox(r_a.Delta_dual, r_legacy.Delta_dual; rtol = 1e-6);
      note = "a=$(r_a.Delta_dual) legacy=$(r_legacy.Delta_dual)")

lp(">>> Gate 3: DECISIVE chain-rule check at real D=20 scale (fixed theta)")
rng = MersenneTwister(11)
dir = randn(rng, D * Ddest - 1); dir ./= norm(dir)
h = 1e-4
w_plus_a = copy(w_a); w_plus_a[2:end] .+= h .* dir
w_minus_a = copy(w_a); w_minus_a[2:end] .-= h .* dir
d_plus_a = decode_outer_unified(w_plus_a, ctx, layout_a, pgc, xy)
d_minus_a = decode_outer_unified(w_minus_a, ctx, layout_a, pgc, xy)
r_plus_a, _ = screened_eval(d_plus_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
r_minus_a, _ = screened_eval(d_minus_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
fd_a = (r_plus_a.Delta_dual - r_minus_a.Delta_dual) / (2h)

dir_z = (-theta_star) .* dir
w_plus_z = copy(w_legacy); w_plus_z[2:end] .+= h .* dir_z
w_minus_z = copy(w_legacy); w_minus_z[2:end] .-= h .* dir_z
d_plus_z = decode_outer_unified(w_plus_z, ctx, layout_z, pgc, xy)
d_minus_z = decode_outer_unified(w_minus_z, ctx, layout_z, pgc, xy)
r_plus_z, _ = screened_eval(d_plus_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
r_minus_z, _ = screened_eval(d_minus_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
fd_z = (r_plus_z.Delta_dual - r_minus_z.Delta_dual) / (2h)

rel_err = abs(fd_a - fd_z) / max(abs(fd_z), 1e-8)
lp("  fd_a=", fd_a, " fd_z(scaled)=", fd_z, " rel_err=", rel_err)
check("decisive chain-rule check rel_err < 1e-4", rel_err < 1e-4; note = "rel_err=$rel_err")

lp(">>> Gate 4: cache A/B/A")
exact_cache = SafeExactCache()
r1, _ = screened_eval(d_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false, exact_cache = exact_cache)
size1 = length(exact_cache)
r2, _ = screened_eval(d_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = true, exact_cache = exact_cache)
size2 = length(exact_cache)
check("cache size unchanged after identical re-eval", size1 == size2; note = "sizes: $size1 -> $size2")
check("Delta_dual bit-identical on cache hit", r1.Delta_dual === r2.Delta_dual)

lp(isempty(FAILURES) ? "ALL D20 UNIFIED-LAYOUT GATES PASS" : "D20 UNIFIED-LAYOUT FAILURES: $(join(FAILURES, "; "))")
flush(stdout)
