# Direct empirical check: how much memory traffic does the touched-row backend actually
# save relative to the sorted (full-copy) backend, at real D=20? Measures the ACTUAL
# touched-row count per coordinate (not assumed), comparing it to W, to give an honest
# bytes-moved estimate rather than a theoretical one.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra, Statistics
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
theta0 = melitz_reduce_theta(p20, ctx20)
n = length(theta0)
W = 80_000
D = ctx20.D
nA = D^2 - 1
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt, hessian_backend=:structured_serial)
r0 = evaluate_melitz_delta(theta0, ctx20, obj; cold=true, store_G=false)
x0 = r0.dual_x

compact = melitz_compact_columns_map(ctx20)
h = 1e-4
sorted_ctx = ctx20.sorted_tail_ctx
n_touched = zeros(Int, n)
n_link = 0
for r in 1:n
    cc = compact[r]
    if cc.touches_link
        global n_link += 1
        n_touched[r] = W   # link coordinates touch ALL W rows -- no savings possible there
        continue
    end
    ncols = length(cc.direct_cols)
    if ncols == 0
        n_touched[r] = 0
        continue
    end
    theta_p = copy(theta0); theta_p[r] += h
    theta_m = copy(theta0); theta_m[r] -= h
    Ap, fp, gpp, fjjp = melitz_expand_theta(theta_p, ctx20)
    Am, fm, gpm, fjjm = melitz_expand_theta(theta_m, ctx20)
    touched = Set{Int}()
    for (o, d) in cc.direct_cells
        Cp = melitz_C(ctx20.w[o], ctx20.tau[o, d], Ap[o, d], ctx20.sigma, ctx20.expenditure[d])
        Cm = melitz_C(ctx20.w[o], ctx20.tau[o, d], Am[o, d], ctx20.sigma, ctx20.expenditure[d])
        cutoff_p = melitz_cutoff(ctx20.w[o], fp[o, d], ctx20.sigma, Cp)
        cutoff_m = melitz_cutoff(ctx20.w[o], fm[o, d], ctx20.sigma, Cm)
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kp = melitz_active_tail_start(sorted_z_o, cutoff_p)
        km = melitz_active_tail_start(sorted_z_o, cutoff_m)
        kunion = min(kp, km)
        perm_o = @view sorted_ctx.permutation[:, o]
        for pos in kunion:W
            push!(touched, perm_o[pos])
        end
    end
    n_touched[r] = length(touched)
end

println("n_theta = ", n, "   W = ", W)
println("coordinates touching the focal link (no savings possible, touch ALL W rows): ", n_link)
non_link = n_touched[.!(getfield.(compact, :touches_link))]
println("\n== touched-row counts, NON-link coordinates (", length(non_link), " of ", n, ") ==")
println("mean touched rows: ", mean(non_link), "  (", round(100*mean(non_link)/W; digits=2), "% of W)")
println("median touched rows: ", median(non_link), "  (", round(100*median(non_link)/W; digits=2), "% of W)")
println("min touched rows: ", minimum(non_link), "  max touched rows: ", maximum(non_link), " (", round(100*maximum(non_link)/W;digits=2), "% of W)")

total_touched_all = sum(n_touched)
total_possible = n * W
println("\n== aggregate across the COMPLETE gradient (all $n coordinates) ==")
println("sum(touched rows) across all coordinates: ", total_touched_all, "  vs n_theta*W = ", total_possible)
println("touched-row backend's per-row work as a fraction of the sorted backend's full-W-per-coordinate work: ",
    round(100 * total_touched_all / total_possible; digits=2), "%")

# Old backend's own memory traffic: 2 full-W copies (u_plus,u_minus) + 2 full-W Psi evals, per
# coordinate, REGARDLESS of touched count -- exactly the ~1.02GB figure the prior closure
# session measured (2*W*n_theta*8 bytes for the copies ALONE, not counting the Psi evaluation
# pass, which is a further 2*W*n_theta*8 bytes of reads).
old_backend_bytes = 2 * W * n * 8   # copyto! traffic alone (the prior session's own cited figure)
old_backend_bytes_with_psi = old_backend_bytes * 2   # + the Psi evaluation pass over the same footprint
new_backend_bytes = 2 * total_touched_all * 8   # delta_plus/delta_minus touched-row writes + Psi evals at touched rows only (2x factor folded below)
new_backend_bytes_with_psi = new_backend_bytes * 2
println("\n== estimated memory-traffic comparison (copy/delta + Psi-evaluation passes) ==")
println(@sprintf("sorted (full-copy) backend: ~%.2f MB/gradient call", old_backend_bytes_with_psi / 1e6))
println(@sprintf("touched-row backend:        ~%.2f MB/gradient call", new_backend_bytes_with_psi / 1e6))
println(@sprintf("ratio (touched-row / sorted): %.4f  (i.e. %.1fx LESS memory traffic)",
    new_backend_bytes_with_psi/old_backend_bytes_with_psi, old_backend_bytes_with_psi/new_backend_bytes_with_psi))
