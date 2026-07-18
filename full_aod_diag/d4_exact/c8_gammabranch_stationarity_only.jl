# Standalone re-run of the headline-point stationarity diagnostic with the warm=false fix (see
# c8_gammabranch_core.jl's external_stationarity_check_c8 docstring note) -- avoids re-running the
# full 50-row table assembly just to fix a stale-warm-state bug in this one diagnostic.
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const OUT_DIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8)
mkpath(OUT_DIR)

# recover a genuinely-converged zfree at the low-g root by solving fresh (cold), COLD warm=false eval afterwards
res_root = profile_delta_at_gamma_c8(0.892635789502, ZFREE_INCUMBENT_C8, ctx, pe; moment_repr = :dense, maxtime_real = 20.0, hessopt_tag = "sr1")
zfree_low_g_root = res_root.best_zfree

res_lower_conf = profile_delta_at_gamma_c8(G_LOWER_INCUMBENT, ZFREE_LOWER_INCUMBENT, ctx, pe; moment_repr = :dense, maxtime_real = 20.0, hessopt_tag = "sr1")

headline = Tuple{String,Vector{Float64}}[]
zfree_low_g_root !== nothing && push!(headline, ("low_g_root", vcat(0.892635789502, zfree_low_g_root)))
push!(headline, ("high_g_lower_incumbent_registry_point", W_LOWER_INCUMBENT))
push!(headline, ("high_g_true_crossing_g0.997031", vcat(0.997030762, ZFREE_LOWER_INCUMBENT)))

stat_rows = NamedTuple[]
for (label, w) in headline
    sc = external_stationarity_check_c8(w, ctx, pe; find_smallest = true, warm = false,
            w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, n)), w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, n)))
    @printf("  [%-40s] Delta=%.10f  Delta-delta=%+.4e  eta=%.6f  eta_nonneg=%s  residual_rel=%.4e  n_active_bounds=%d  n_nonfinite_probes=%d\n",
        label, sc.Delta, sc.Delta_minus_delta, sc.eta, sc.eta_nonneg, sc.residual_relative, sc.n_active_bounds, sc.n_nonfinite_probes)
    push!(stat_rows, (label = label, Delta = sc.Delta, Delta_minus_delta = sc.Delta_minus_delta, eta = sc.eta,
                       eta_nonneg = sc.eta_nonneg, residual_relative = sc.residual_relative,
                       n_active_bounds = sc.n_active_bounds, n_nonfinite_probes = sc.n_nonfinite_probes))
end
open(joinpath(OUT_DIR, "c8_gammabranch_stationarity.csv"), "w") do io
    println(io, "label,Delta,Delta_minus_delta,eta,eta_nonneg,residual_relative,n_active_bounds,n_nonfinite_probes")
    for r in stat_rows
        println(io, r.label, ",", r.Delta, ",", r.Delta_minus_delta, ",", r.eta, ",", r.eta_nonneg, ",", r.residual_relative, ",", r.n_active_bounds, ",", r.n_nonfinite_probes)
    end
end
println("Wrote ", joinpath(OUT_DIR, "c8_gammabranch_stationarity.csv"))
