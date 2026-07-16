# ============================================================================
# Multistart screening: is the D=20 "stuck at A*" issue driven by starting at
# A_od*, i.e. would a DIFFERENT starting A find a different (possibly better)
# basin? Cheap first pass: for many random perturbations of A* at increasing
# noise scales, just check whether the INNER problem (recover_lfd + destination
# inversion + sequential gravity loop) is even feasible there at all -- before
# spending a full expensive outer KNITRO search on any of them. This directly
# tests the hypothesis (user's own prior) that the inner loop frequently fails
# away from the calibrated A*.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     REAL_DATA_DIR=.../real_data/noah_D20 \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/multistart_screening_d20.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, Random, LinearAlgebra, DelimitedFiles

const NOISE_SCALES = [0.1, 0.3, 0.6, 1.0, 2.0]   # log-normal multiplicative sigma on Acol
const N_PER_SCALE = 4
const SEED = 20260715

Random.seed!(SEED)
rows = NamedTuple[]
θbase = copy(θr0)   # gamma'_focal stays at theta_r0's own value; only Acol is perturbed
Acol_star = θbase[4:3+D]

println("="^78); println(">>> MULTISTART SCREENING: random Acol perturbations around A*, D=$D, W=$W"); println("="^78)
@printf("Acol* (A_od at the Frechet benchmark) = %s\n", Acol_star)

idx = 0
for σ in NOISE_SCALES
    for rep in 1:N_PER_SCALE
        global idx += 1
        noise = σ .* randn(D)
        Acol_pert = Acol_star .* exp.(noise)
        θtest = copy(θbase); θtest[4:3+D] .= Acol_pert
        relΔA = norm(Acol_pert .- Acol_star) / norm(Acol_star)

        t0 = time()
        local col, R, Rcol, umat, p, ok
        try
            col, R, Rcol, umat, p, ok = seq_gravcol(θtest; δ=Inf, maxit=100, tol=5e-4)
        catch e
            ok = false; R = NaN
            @printf("[%2d] sigma=%.2f relΔA=%.3f  ERRORED: %s\n", idx, σ, relΔA, sprint(showerror, e))
        end
        wall = time() - t0
        div_p = (ok && !isempty(p)) ? divergence_of(p) : NaN
        @printf("[%2d] sigma=%.2f relΔA=%.3f  gravity_ok=%-5s R_mean=%10.3e  div(p)=%10s  wall=%.1fs\n",
            idx, σ, relΔA, ok, R, ok ? @sprintf("%.4f", div_p) : "n/a", wall)
        push!(rows, (idx=idx, sigma=σ, relΔA=relΔA, ok=ok, R_mean=R, div_p=div_p, wall=wall, Acol=copy(Acol_pert)))
        flush(stdout)
    end
end

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
for σ in NOISE_SCALES
    rs = filter(r -> r.sigma == σ, rows)
    nfeas = count(r -> r.ok, rs)
    @printf("sigma=%.2f: %d/%d gravity-feasible\n", σ, nfeas, length(rs))
end
n_total_feas = count(r -> r.ok, rows)
@printf("\nTOTAL: %d/%d random Acol perturbations were gravity-feasible (delta=Inf, i.e. ignoring any divergence budget)\n",
    n_total_feas, length(rows))

open(joinpath(@__DIR__, "multistart_screening_d20_results.csv"), "w") do io
    writedlm(io, ["idx" "sigma" "relDeltaA" "gravity_ok" "R_mean" "div_p" "wall_s"], ',')
    for r in rows
        writedlm(io, [[r.idx r.sigma r.relΔA r.ok r.R_mean r.div_p r.wall]], ',')
    end
end

feas_rows = filter(r -> r.ok, rows)
if !isempty(feas_rows)
    println("\n" * "="^78); println(">>> Feasible points found -- these are candidates for a full outer search from a different basin"); println("="^78)
    sort!(feas_rows, by = r -> -r.relΔA)   # furthest-from-A* first
    for r in feas_rows[1:min(3, end)]
        @printf("idx=%d sigma=%.2f relΔA=%.3f div(p)=%.4f -- candidate for full outer search\n", r.idx, r.sigma, r.relΔA, r.div_p)
    end
else
    println("\nNo feasible points found among the tested perturbations -- consistent with the user's prior that random A rarely admits a feasible inner solve.")
end

println("\nMULTISTART SCREENING DONE")
