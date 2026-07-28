# ================================================================================================
# True no-H operator bundle task, Part B §9-11: does the production gravity-PIVOT decoder
# (gravity_elimination.jl's pivot_expand, as actually used by run_cm_upper_checkpointed -- see
# cm_checkpoint.jl:1016 `logA_full = pivot_expand(zfree_now, pe)`, invoked for BOTH
# A_coordinate_mode in (:legacy_z, :powered_aspace)) give machine-zero gravity residual at 100
# deterministic non-calibration perturbations? And does a "naive" CS.reconstruct_full-only
# perturbation (bypassing pivot_expand) NOT?
#
# This directly tests the earlier finding (5e-17 at calibration, ~0.003 after "a naive
# perturbation" via `ctx.θ0_up[ctx.free_idx]`/`CS.reconstruct_full`) against the REAL pivot
# decoder, to determine whether that finding was an invalid-point diagnostic (bypassed pivot) or a
# genuine pivot bug.
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "gravity_elimination.jl"]
    include(joinpath(_D4E, f))
end
for f in ["winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
D2 = ctx.D * Ddest

pe = build_pivot_elimination(ctx)
lp(xs...) = (println(xs...); flush(stdout))
lp("D=", ctx.D, " Ddest=", Ddest, " D*Ddest=", D2, " pivot_lin=", pe.pivot_lin, " |c[pivot]|=", abs(pe.c[pe.pivot_lin]))

# Calibration A_od block (levels, NOT log) and its pivot-reduced z_free.
Aod_calib = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2], ctx.D, Ddest)
z_calib = log.(Aod_calib)
zfree_calib = pivot_reduce(z_calib, pe)

# Round-trip sanity: pivot_expand(zfree_calib) must reconstruct z_calib to numerical precision
# (task's own standing note: calibration lies exactly on the gravity-zero hyperplane).
z_roundtrip = pivot_expand(zfree_calib, pe)
lp("Round-trip max|Δz| at calibration: ", maximum(abs.(z_roundtrip .- z_calib)))

d_gravity = ctx.obj.d   # gravity is the LAST moment column, obj.H[:, 2+d]; PMM/σ indexed by this d
pmm_g = ctx.γ.PMM[d_gravity]
lp("d_gravity (last moment index) = ", d_gravity, "   PMM[d_gravity] = ", pmm_g)

"""
    theta_full_from_logA(logA, ctx) -> θ_full

Embeds a D x Ddest log(Aod_theta) matrix into a copy of ctx.θ0_up's A_od block -- the SAME
embedding gravity_from_logz (gravity_elimination.jl) uses internally, exposed standalone so we can
also evaluate compressed_gravity_raw/fill_gravity_column_into! at the SAME θ_full.
"""
function theta_full_from_logA(logA::AbstractMatrix, ctx)
    θ_full = copy(ctx.θ0_up)
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*Ddest] .= vec(exp.(logA))
    return θ_full
end

N = 100
Random.seed!(20260728)
rows = Vector{NamedTuple}(undef, 2N)
row_i = 0

function record!(label, kind, scale, logA, θ_full)
    global row_i
    outer_gravity_eq = gravity_from_logz(logA, ctx)   # task's "outer gravity equality"
    grav_raw = compressed_gravity_raw(θ_full, ctx)
    inner_residual = grav_raw - pmm_g                  # task's "compressed_gravity_raw - pmm_g"
    grav_col = zeros(2)
    fill_gravity_column_into!(grav_col, grav_raw, ctx, d_gravity)
    row_i += 1
    rows[row_i] = (idx = row_i, label = label, kind = kind, scale = scale,
                   outer_gravity_equality = outer_gravity_eq,
                   compressed_gravity_raw_minus_pmm = inner_residual,
                   gravity_column_sample = grav_col[1])
end

lp("="^90)
lp("PIVOT-VALID points: z_free perturbed, pivot cell RE-SOLVED via pivot_expand (the real decoder)")
lp("="^90)
for i in 1:N
    scale = 0.01 + 0.001 * i    # deterministic, increasing perturbation magnitude, not all tiny
    zfree_i = zfree_calib .+ scale .* randn(length(zfree_calib))
    logA_i = pivot_expand(zfree_i, pe)
    θ_full_i = theta_full_from_logA(logA_i, ctx)
    record!("pivot_valid_$i", "pivot_valid", scale, logA_i, θ_full_i)
end

lp("="^90)
lp("NAIVE points: ALL D*Ddest log-A cells (including what pivot_expand calls the pivot) perturbed independently -- bypasses pivot_expand entirely")
lp("="^90)
for i in 1:N
    scale = 0.01 + 0.001 * i
    z_naive = z_calib .+ scale .* randn(size(z_calib))   # every cell free, no pivot re-solve
    θ_full_i = theta_full_from_logA(z_naive, ctx)
    record!("naive_$i", "naive_bypass_pivot", scale, z_naive, θ_full_i)
end

# --- Also reproduce the EXACT inherited-session pattern: ctx.θ0_up[ctx.free_idx] -> CS.reconstruct_full,
# perturbed directly in x_free (NOT in z/logA space at all), which is what "a naive perturbation"
# literally meant in that finding.
lp("="^90)
lp("Exact inherited-session-style reproduction: perturb x_free = θ0_up[free_idx], CS.reconstruct_full only")
lp("="^90)
x_free_calib = ctx.θ0_up[ctx.free_idx]
for (label, scale) in [("calibration_exact", 0.0), ("naive_perturb_1pct", 0.01), ("naive_perturb_small", 1e-4)]
    x_free_i = scale == 0.0 ? x_free_calib : x_free_calib .+ scale .* abs.(x_free_calib) .* randn(length(x_free_calib))
    θ_full_i = CS.reconstruct_full(x_free_i, ctx.m)
    logA_i = log.(reshape(θ_full_i[ctx.Aod_offset+1:ctx.Aod_offset+D2], ctx.D, Ddest))
    outer_eq = gravity_from_logz(logA_i, ctx)
    grav_raw = compressed_gravity_raw(θ_full_i, ctx)
    lp(@sprintf("  %-22s scale=%.4g  outer_gravity_equality=%.6e  compressed_gravity_raw-pmm=%.6e",
                label, scale, outer_eq, grav_raw - pmm_g))
end

# --- Summary ---
lp("="^90)
pivot_rows = rows[1:N]
naive_rows = rows[N+1:2N]
max_outer_pivot = maximum(abs.([r.outer_gravity_equality for r in pivot_rows]))
max_inner_pivot = maximum(abs.([r.compressed_gravity_raw_minus_pmm for r in pivot_rows]))
max_outer_naive = maximum(abs.([r.outer_gravity_equality for r in naive_rows]))
max_inner_naive = maximum(abs.([r.compressed_gravity_raw_minus_pmm for r in naive_rows]))
lp(@sprintf("PIVOT-VALID (n=%d): max|outer_gravity_equality|=%.3e   max|compressed_gravity_raw-pmm|=%.3e", N, max_outer_pivot, max_inner_pivot))
lp(@sprintf("NAIVE       (n=%d): max|outer_gravity_equality|=%.3e   max|compressed_gravity_raw-pmm|=%.3e", N, max_outer_naive, max_inner_naive))

pass_pivot_machine_zero = max_outer_pivot < 1e-8 && max_inner_pivot < 1e-8
pass_naive_nonzero = max_outer_naive > 1e-4 && max_inner_naive > 1e-4
lp("PASS pivot-valid points machine-zero: ", pass_pivot_machine_zero)
lp("PASS naive points genuinely nonzero: ", pass_naive_nonzero)

# --- Deterministic feasibility check (task §11): gravity column sign ---
lp("="^90)
lp("Deterministic feasibility check: gravity column sign at a NAIVE (nonzero-residual) point")
lp("="^90)
z_naive_bad = z_calib .+ 0.05 .* randn(size(z_calib))
θ_full_bad = theta_full_from_logA(z_naive_bad, ctx)
grav_raw_bad = compressed_gravity_raw(θ_full_bad, ctx)
W = size(ctx.U, 1)
grav_col_full = zeros(W)
fill_gravity_column_into!(grav_col_full, grav_raw_bad, ctx, d_gravity)
all_same_sign = all(>=(0.0), grav_col_full) || all(<=(0.0), grav_col_full)
lp("grav_raw-pmm = ", grav_raw_bad - pmm_g, "   column all one sign (or zero) = ", all_same_sign,
   "   min=", minimum(grav_col_full), " max=", maximum(grav_col_full))

# Write CSV deliverable (manual write -- no CSV.jl dependency in this project's environment)
outpath = joinpath(_D4E, "..", "..", "GRAVITY_PIVOT_VALID_COORDINATE_TESTS_2026-07-28.csv")
open(outpath, "w") do io
    println(io, "idx,label,kind,scale,outer_gravity_equality,compressed_gravity_raw_minus_pmm,gravity_column_sample")
    for r in rows
        println(io, r.idx, ",", r.label, ",", r.kind, ",", r.scale, ",",
                r.outer_gravity_equality, ",", r.compressed_gravity_raw_minus_pmm, ",", r.gravity_column_sample)
    end
end
lp("Wrote ", abspath(outpath))
lp("="^90)
lp("DONE")
