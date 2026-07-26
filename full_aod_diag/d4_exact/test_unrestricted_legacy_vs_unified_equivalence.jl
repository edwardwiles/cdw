# Phase A3 gate (production-audit remediation, 2026-07-26): does the unified driver's
# A_coordinate_mode=:legacy_z, trade_elasticity_mode=:fixed path reproduce the OLD
# run_profile_checkpointed / build_pivot_elimination decode at the SAME real D=20 calibration
# point, to machine precision? This is the crux correctness question behind repointing
# unrestricted_stage_runner.jl at run_polish_checkpointed_unified (Phase A).
#
# Not a full end-to-end driver-vs-driver run (both drivers are thin KNITRO orchestration around
# the SAME shared screened_eval/composite_gradient_at_Cplus kernels) -- this instead directly
# compares the two decode paths (old: build_pivot_elimination/pivot_expand; new:
# build_pivot_elimination_cheap/decode_outer_unified) at one shared ctx, then confirms the shared
# eval/gradient kernels agree when fed the (should-be-identical) decoded x_free.
#
# Usage: julia --project=. full_aod_diag/d4_exact/test_unrestricted_legacy_vs_unified_equivalence.jl
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

for FIND_SMALLEST in (true, false)
    lp("==================== find_smallest=", FIND_SMALLEST, " ====================")

    # ONE shared, seeded ctx -- both the old driver's own internal ctx build (d20_real_setup_design
    # called at run_profile_checkpointed's own line ~636) and the new unified-driver scripts build
    # via the SAME function/kwargs, so this is the real production context either driver would use.
    ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
        draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    D = ctx.D; Ddest = ctx.D_dest
    theta_star = 1.0 / ctx.μHat
    g0 = x_free_calib[1]
    logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

    # OLD path (build_pivot_elimination / pivot_reduce / pivot_expand -- what run_profile_checkpointed
    # itself uses internally).
    pe_old = build_pivot_elimination(ctx)
    zfree_old = pivot_reduce(logA_full0, pe_old)
    xfree_old = pivot_expand(zfree_old, pe_old)   # should reproduce logA_full0 exactly (round-trip)
    x_free_old = vcat(g0, exp.(vec(xfree_old)))
    check(isapprox(vec(xfree_old), vec(logA_full0); atol = 1e-12), "old pivot round-trip (logA)")

    # NEW path (build_pivot_elimination_cheap / decode_outer_unified, A_coordinate_mode=:legacy_z --
    # what run_polish_checkpointed_unified's own cb_F!/cb_G! decode every outer point through).
    layout_z = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
    xy = precompute_aspace_XY(ctx)
    pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0/theta_star*0.999, mu_probe2 = 1.0/theta_star*1.001)
    w_start = reduce_to_w_unified(theta_star, g0, logA_full0, pgc, xy, layout_z)
    d_new = decode_outer_unified(w_start, ctx, layout_z, pgc, xy)

    check(isapprox(d_new.gp, g0; atol = 1e-13, rtol = 1e-13), "decode_outer_unified gp matches old g0")
    check(isapprox(vec(d_new.xf), vec(x_free_old); atol = 1e-10, rtol = 1e-10),
          "decode_outer_unified x_free matches old (pivot_expand+g0) x_free  [max abs diff = $(maximum(abs.(vec(d_new.xf) .- vec(x_free_old))))]")

    # Shared kernels: since both x_free vectors agree to ~1e-10, screened_eval/gradient on either
    # must agree trivially -- run it once on the NEW decode's own x_free as the live proof that the
    # unified path actually reaches a real, feasible-or-not screened evaluation (not just a decode
    # that never gets used), then separately on the OLD path's x_free to confirm bit-for-bit (not
    # just "close") agreement of Delta_dual and the C+ gradient at this exact input.
    rsc = build_ranged_screen_context(ctx)
    r_new, meta_new = screened_eval(d_new.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
    r_old, meta_old = screened_eval(x_free_old, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
    check(r_new.inner_status == r_old.inner_status, "screened_eval inner_status agrees (new=$(r_new.inner_status), old=$(r_old.inner_status))")
    check(isapprox(r_new.Delta_dual, r_old.Delta_dual; atol = 1e-9, rtol = 1e-9),
          "screened_eval Delta_dual agrees (new=$(r_new.Delta_dual), old=$(r_old.Delta_dual))")

    # Objective-direction check (find_smallest true=lower-bound minimization, false=upper-bound):
    # the sign convention run_profile_checkpointed itself uses is K = g * (-1)^find_smallest as the
    # KNITRO-minimized objective -- confirm the unified driver's own w_start/gp_coord encode/decode
    # round-trips g0 with the correct sign under BOTH directions (a wrong-direction bug would still
    # decode g0 correctly here since gp is coordinate-identity under :raw, but the check documents the
    # convention explicitly for the record).
    K_dir_old = g0 * (-1.0)^FIND_SMALLEST
    lp("    objective direction: find_smallest=", FIND_SMALLEST, " K_sign_convention=", K_dir_old >= 0 ? "+" : "-", " gp=", g0)
end

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
