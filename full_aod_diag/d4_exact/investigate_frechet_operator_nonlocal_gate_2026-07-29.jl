# Investigation (2026-07-29): does common-Frechet's `moment_representation=:operator` (true no-H
# OperatorPsiBundle) actually reproduce the historical nStatus=-400 failure at NON-calibration
# OUTER theta points at real D=20 scale?
#
# BACKGROUND: `build_cm_frechet_production_context`'s `moment_representation` kwarg defaults to
# `:dense_reference` (cm_frechet_level.jl:345), citing "archC_frechet_base_state's HISTORY comment"
# -- two real, reproduced nStatus=-400 failures from skipping the dense CM/level column fill, found
# specifically at NON-calibration OUTER points (x_free0.*1.01 etc), root cause never identified.
# But that HISTORY comment is about a DIFFERENT, already-abandoned mechanism: `skip_cm_fill_ref`/
# `skip_fill`, which left a STILL-DENSE `PsiObjectiveBundleImplicit`'s CM/level columns stale
# in-place while running the `:cm_frechet_lookup` FG backend. `moment_representation=:operator` is
# structurally different: it builds an `OperatorPsiBundle` that has NO dense H field AT ALL -- any
# code path that still depended on those columns would hit a hard MethodError/FieldError, not a
# silent nStatus=-400, and the five-family no-H generalization session (2026-07-28,
# FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md) found and fixed exactly 3 such hard-failure bugs.
#
# THE ACTUAL GAP: every existing no-H-bundle equivalence gate for common Frechet
# (test_operator_no_H_bundle_equivalence_frechet_d20.jl, commit ad1cd43) runs the REAL full inner
# solve ONLY from the calibration outer theta (x_free_calib) -- exactly matching this family's own
# documented failure pattern ("missed at the calibration point, caught beyond it"). This script
# closes that gap: it runs the real full inner solve, dense vs operator, at the calibration point
# AND at the same non-calibration outer points (near-delta1-perturbed, hard-point x1.01) that
# historically reproduced the -400 failure under the OLDER skip_fill mechanism.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl",
          "lfix_factorized_workspace.jl", "lfix_factorized.jl", "cm_screen_bridge.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random

lp(xs...) = (println(xs...); flush(stdout))

lp("Building real D=20 context (W=80000, delta=1.0, destination_sample=:exclude_row)...")
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)
L = 50

x_free_near = x_free_calib .+ vcat(0.005, 0.01 .* randn(length(x_free_calib) - 1))
x_free_hard = x_free_calib .* 1.01

results = NamedTuple[]
ALL_OK = Ref(true)

for contrasts in (:anchored, :orthonormal)
    println("="^90); println("contrasts=$contrasts L=$L"); println("="^90); flush(stdout)

    pcx_d = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup,
        moment_representation = :dense_reference)
    pcx_o = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup,
        moment_representation = :operator)
    level_targets = pcx_d.aug.level_targets

    for (label, x_free) in (("calib", x_free_calib), ("near_delta1_perturbed", x_free_near), ("hard_point_x1.01", x_free_hard))
        print("  $label: dense solve...  "); flush(stdout)
        t0 = time()
        base_d = try
            archC_frechet_base_state(x_free, pcx_d.ctx_cm, pcx_d.cctx, level_targets)
        catch e
            println("THREW: ", sprint(showerror, e))
            nothing
        end
        dt_d = time() - t0
        nsd = base_d === nothing ? -99999 : base_d.inner_status
        @printf "nStatus=%d (%.1fs)\n" nsd dt_d

        print("  $label: operator solve...  "); flush(stdout)
        t0 = time()
        base_o = try
            archC_frechet_base_state(x_free, pcx_o.ctx_cm, pcx_o.cctx, level_targets)
        catch e
            println("THREW: ", sprint(showerror, e))
            nothing
        end
        dt_o = time() - t0
        nso = base_o === nothing ? -99999 : base_o.inner_status
        @printf "nStatus=%d (%.1fs)\n" nso dt_o

        feasd = nsd in (0, -100, -101, -102, -103, -400, -401, -402)
        feaso = nso in (0, -100, -101, -102, -103, -400, -401, -402)
        ok_status_match = nsd == nso
        zdiff = (base_d !== nothing && base_o !== nothing) ? abs(base_d.ζstar - base_o.ζstar) : NaN
        ldiff = (base_d !== nothing && base_o !== nothing) ? maximum(abs.(base_d.λstar .- base_o.λstar)) : NaN
        @printf "    status_dense=%d status_operator=%d match=%s  |Δζ*|=%.3e  max|Δλ*|=%.3e\n" nsd nso ok_status_match zdiff ldiff
        cell_ok = ok_status_match && (isnan(zdiff) || zdiff < 1e-6)
        ALL_OK[] &= cell_ok
        push!(results, (contrasts = contrasts, label = label, status_dense = nsd, status_operator = nso,
                         match = ok_status_match, zdiff = zdiff, ldiff = ldiff, t_dense = dt_d, t_operator = dt_o, pass = cell_ok))
        flush(stdout)
    end
end

println(); println("="^90); println("SUMMARY"); println("="^90)
for r in results
    @printf "%-12s %-22s status(d/o)=%d/%d match=%s |Δζ*|=%.3e pass=%s\n" String(r.contrasts) r.label r.status_dense r.status_operator r.match r.zdiff r.pass
end
println()
println(ALL_OK[] ? "ALL NON-CALIBRATION OPERATOR-VS-DENSE CHECKS PASS" : "SOME FAILURES ABOVE (see per-cell status/throw messages)")
