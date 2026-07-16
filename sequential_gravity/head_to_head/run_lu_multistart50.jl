# ============================================================================
# LU MULTISTART FOLLOW-UP (2026-07-15, post-comparison): the official 9-solve
# LU run showed T1/T3 solves are cheap (~20-30s, converge in 1 KNITRO eval)
# while T2 needs a real search (~430s, 18 evals) -- and T3 (the hardest
# target) found NO feasible point from any of its 3 starts. Since LU is this
# cheap, a much larger multistart is affordable: 50 shared points (see
# generate_lu_multistart_points.jl, graduated sigma 0.1-2.0) x 3 targets = 150
# solves, INDEPENDENT (no warm-starting between targets -- the point here is
# broad exploration, not the fair-comparison warm-start schedule the official
# run used). Estimated ~7.2h, dominated by T2's cost (50 x ~430s ~ 6h).
#
# Reuses run_lu.jl's own KNITRO machinery (minimize_delta_star_h2h, lu_solve,
# lu_target_winner, Acol_star_h2h, etc.) via SKIP_MAIN_LOOP=true -- does NOT
# touch or re-run the official (already-ALLDONE) 9-solve LU comparison result.
# Separate output directory (out_lu_multistart50/) and ALLDONE sentinel
# (lu_multistart50_ALLDONE) so it cannot collide with or be mistaken for the
# official result.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/run_lu_multistart50.jl
# ============================================================================
ENV["SKIP_MAIN_LOOP"] = "true"
include(joinpath(@__DIR__, "run_lu.jl"))

const POINTS_PATH = joinpath(@__DIR__, "lu_multistart_points.jld2")
@assert isfile(POINTS_PATH) "run generate_lu_multistart_points.jl first"
pdata = JLD2.load(POINTS_PATH)
@assert isapprox(pdata["Acol_star"], Acol_star_h2h; rtol = 1e-10) "lu_multistart_points.jld2's Acol_star doesn't match this run's setup -- regenerate"
const MS_POINTS = [Float64.(p) for p in pdata["points"]]   # Vector of 50 Vector{Float64}
const MS_SIGMAS = Float64.(pdata["sigmas"])
const N_POINTS = length(MS_POINTS)
@assert N_POINTS == 50

const OUT_DIR_MS = joinpath(H2H_DIR, "out_lu_multistart50")
isdir(OUT_DIR_MS) || mkpath(OUT_DIR_MS)
ms_result_path(tname, idx) = joinpath(OUT_DIR_MS, "lu_ms_$(tname)_pt$(lpad(idx, 2, '0')).jld2")

println("\n" * "="^78)
@printf(">>> LU MULTISTART50: D=%d W=%d  %d points x %d targets = %d solves\n", D, W, N_POINTS, length(TARGETS), N_POINTS * length(TARGETS))
println("="^78)
flush(stdout)

"""
    ms_solve(tname, idx, GT, Aod_init, sigma)

Same machinery as the official lu_solve, but writing to the multistart output
directory/filename scheme and recording which noise scale generated this
point.
"""
function ms_solve(tname::String, idx::Int, GT::Float64, Aod_init::Vector{Float64}, sigma::Float64)
    path = ms_result_path(tname, idx)
    existing = load_done(path)
    if existing !== nothing
        @printf("[LU-MS %s/pt%02d] ALREADY DONE (resume) -- best_feasible_delta_star=%.6g\n",
                tname, idx, existing["best_feasible_delta_star"])
        flush(stdout)
        return existing
    end
    r = minimize_delta_star_h2h(copy(Aod_init), GT; tname = tname, sname = "pt$(lpad(idx,2,'0'))", ckpt_path = path)
    d = load_done(path)
    d === nothing && error("ms_solve $tname/pt$idx: minimize_delta_star_h2h did not save done=true")
    d["sigma"] = sigma
    JLD2.save(path, d)   # re-save with sigma annotation added
    d
end

for tgt in TARGETS
    tname = tgt.name; GT = tgt.gp
    println("\n" * "-"^78)
    @printf("[LU-MS] TARGET %s: GT=gamma'_focal=%.9f (kappa=%.6f)\n", tname, GT, tgt.kappa)
    println("-"^78)
    flush(stdout)
    for idx in 1:N_POINTS
        ms_solve(tname, idx, GT, MS_POINTS[idx], MS_SIGMAS[idx])
    end
end

println("\n" * "="^78); println(">>> LU MULTISTART50 SUMMARY"); println("="^78)
for tgt in TARGETS
    tname = tgt.name
    rows = [load_done(ms_result_path(tname, idx)) for idx in 1:N_POINTS]
    feas = filter(r -> r !== nothing && isfinite(r["best_feasible_delta_star"]), rows)
    n_feas = length(feas)
    if n_feas == 0
        @printf("%s (GT=%.6f): 0/%d points feasible -- NO FEASIBLE RESULT\n", tname, tgt.gp, N_POINTS)
    else
        best = feas[argmin([r["best_feasible_delta_star"] for r in feas])]
        @printf("%s (GT=%.6f, kappa=%.6f): %d/%d points feasible, BEST delta_star=%.6g (pt%s, sigma=%.2f, relΔA=%.3f)\n",
                tname, tgt.gp, tgt.kappa, n_feas, N_POINTS, best["best_feasible_delta_star"],
                best["start"], best["sigma"], best["relDeltaA"])
    end
end

open(joinpath(H2H_DIR, "lu_multistart50_ALLDONE"), "w") do io
    println(io, string(Dates.now()))
end
println("\nLU_MULTISTART50 DONE")
