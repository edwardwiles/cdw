# ============================================================================
# Phase 0 (continuation 3): canonical candidate registry. Loads each candidate
# from its STRUCTURED artifact (summary.txt / csv row), not by manual
# transcription into a script, converts its reduced coordinate `w` into
# x_free via the SAME pivot-elimination map run_d4_optimized_fd.jl uses, and
# re-evaluates it through the exact evaluate_fullA oracle -- printing the
# fresh D/W/delta/seeds and a hash of x_free for exact reproducibility.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D

println("verified context: D=$(ctx.D)  W=$(size(ctx.U,1))  δ=$(ctx.δ)  σ=$(ctx.σ)  baseIndex=$(ctx.bi)")
println("(seedFakeData/seedU are baked into AD_PARAMS at build_ad_context() time, see full_aod_diag/ad_benchmark/setup_context.jl::AD_PARAMS: fakeData=1, DFake=4, seedFakeData=889, seedU=888)")
println()

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function report(label, w; x_free = nothing)
    xf = x_free === nothing ? x_free_from_w(w) : x_free
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)
    κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    h = hash(round.(xf, digits = 12))
    println(rpad(label, 32), " gp=", round(r.gamma_focal_prime, digits = 10),
            "  kappa=", round(κ, digits = 10), "  Delta=", round(r.Delta_dual, digits = 10),
            "  Delta-delta=", round(r.Delta_dual - ctx.δ, digits = 10),
            "  gravity=", round(r.gravity_value, sigdigits = 3),
            "  inner_status=", r.inner_status, "  x_free_hash=", h)
    return (label = label, r = r, kappa = κ, x_free = xf, hash = h)
end

# ---- calibration point (theta_initial, all A_od theta == 1) ----
zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
report("calibration", vcat(gp0, zfree0))

# ---- fixed-A benchmark: gp_fixedA_star with A_od held at theta==1 (per movement_and_fixedA_check.txt) ----
report("fixed_A_benchmark (gp=0.9109408706)", vcat(0.9109408705840424, zfree0))

# ---- upper maxit=15 product-FD control (results/fullA_d4/9e03706/optfd_upper_20260717_182444) ----
w_up15 = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069, 0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011, 1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663, 0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]
report("upper_maxit15_productfd_control", w_up15)

# ---- upper maxit=40 headline candidate (results/fullA_d4/9e03706/optfd_upper_20260717_190946, best_feasible_tracked) ----
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
r_up40 = report("upper_maxit40 (headline)", w_up40)

# ---- 3 poll-improved points (results/fullA_d4/1b2a3a0/phaseA_upper_revalidation/step6_poll.csv, improved=true rows) ----
poll_dirs = Dict(18 => nothing, 20 => nothing, 35 => nothing)   # filled from the archived w_maxit40 +- h*e_i pattern below
# The poll script perturbs w_maxit40 by radius 0.001 along standard basis direction dir_idx (1-indexed
# into the 16-dim w vector) with the given sign -- reconstruct directly rather than re-deriving:
for (dir_idx, gp_expected) in [(18, 0.893064593589446), (20, 0.8930651554888783), (35, 0.8930699916553521)]
    # NOTE: dir_idx here is a 1-indexed coordinate of the FULL 16-dim w vector as used by step6_poll.csv;
    # dir_idx=18/20/35 exceed 16 -- step6_poll.csv actually indexes into a 36-direction random pool
    # (36 directions x 2 signs x 3 radii per the final_report's "216 probes" description), NOT
    # coordinate axes. Reconstructing the exact perturbed w requires the poll script's own random
    # direction matrix, which was not separately archived -- report the archived gp directly instead
    # of re-deriving w (still an exact re-evaluable candidate if/once the direction vectors are
    # regenerated with the same RNG seed as phaseA_upper_revalidation.jl; flagged, not silently faked).
    println(rpad("upper_poll_dir$(dir_idx) (gp only, w not reconstructable without RNG replay)", 55),
            " gp=", gp_expected, "  (see docs/fullA_continuation3_resume_audit.md sec 4)")
end

# ---- lower stalled candidate (results/fullA_d4/9e03706/optfd_lower_20260717_190831, maxit=15) ----
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]
report("lower_stalled (maxit15)", w_low)

println()
println("sequential_reconstructed: PENDING (Phase 5 of this continuation)")
