# Phase 2/3 verification for CM+ZC-CROSS (2026-08-09 task): the outer-gradient envelope formula
# (d_delta_dual_d_nu_cross_vec / d_delta_dual_d_eta_nu_cross_vec, and the Variant D pair
# d_delta_dual_d_eta_active_and_nustar_shared_cross) plus the fixed-contribution fold
# (meanzc_cross_fixed_contribution).
#
# THE NU-GRADIENT IS THE GENUINELY NEW MATH IN THIS FAMILY -- it is NOT a mirror of anything. The
# base CM+ZC family writes ONE independent component per level (`out[k] = mean_m*total`, using
# `d_pair_dnu = -2nu`, cm_meanzc_moments.jl:589-608) because its outer Jacobian is block-diagonal in
# k. The cross grid COUPLES levels: block (k1,k2) has target nu_{k1}*nu_{k2}, so the derivative must
# ACCUMULATE -nu_{k2} into slot k1 and -nu_{k1} into slot k2. That restructure is what these checks
# exist to validate numerically, not by inspection.
#
# THREE INDEPENDENT CHECKS, none of which can pass by construction/mirroring alone:
#
#   (B) meanzc_cross_fixed_contribution: cache0.q0 .- cm_contrib0 .- meanzc_contrib0 (closed-form
#       winner-cache path) vs a FRESH, independent recompute of q at the exact solution via the raw
#       operator FG primitive (CMMeanZCOperatorState's dual_index!, cm_meanzc_lookup_kernels.jl) --
#       two completely different code paths that must agree if the fold is correct.
#       RUN FIRST, deliberately: the FD helper below re-solves at many perturbed eta points, leaving
#       cctx.cmlookup_st's cached state (core_cf_ref / zc_ws targets) at the LAST probe rather than
#       at the calibration point. Running (B) afterwards would compare base.λstar (calibration) to a
#       stale st -- a test-ordering bug, not a bug in the functions under test. Same trap, same
#       ordering fix, as verify_ozc_cross_gradient_d4_2026-08-09.jl (documented there 2026-08-09).
#
#   (A) d_delta_dual_d_eta_nu_cross_vec vs central finite differences of Delta_dual at INDEPENDENTLY
#       REOPTIMIZED perturbed eta points (each probe re-solves the inner dual from scratch -- NOT a
#       fixed-dual shortcut). This is the SAME validation pattern the base family's own
#       d_delta_dual_d_nu_vec was validated with.
#
#   (C) VARIANT D, the piece the handover explicitly flags as "FD-check it, do not assume":
#       d_delta_dual_d_eta_active_and_nustar_shared_cross's `d_delta_d_nu_star`. Under the shared
#       layout the focal nu_{k*} appears in cross-pair blocks at BOTH (k*,k2) for every k2 AND
#       (k1,k*) for every k1 -- 2*K_pair-1 blocks, versus the diagonal family's single (k*,k*). So
#       this collects strictly more terms than in the diagonal family and is NOT covered by (A).
#       Reference: the dense FD gradient at every dense coordinate, un-chain-ruled at the omitted
#       slot (d(Delta)/d(nu_star) = d(Delta)/d(eta_star) / nu_star, since eta=log(nu)) -- computed
#       against a SEPARATE aml-active context so the ragged lambda layout matches.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/verify_cmzc_cross_gradient_d4_2026-08-09.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "compressed_moments.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "compressed_factual_buffer_reuse.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "winner_pair_cross_hessian.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "threaded_cross_hessian.jl",
          "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl", "operator_hessian_weights.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 8
probs = cm_equal_grid_probs(L)
nu0_shared(K::Int) = [gamma(1 - ctx.μHat * k) for k in 1:K]

"""
Reoptimized (NOT fixed-dual) central FD of Delta_dual w.r.t. every eta_nu coordinate at a fixed
economic outer point -- the CM+ZC analog of `d_delta_dual_d_eta_origin_fd` (cm_originzc_production.jl),
written here rather than in a production file because it is validation-only. Each probe re-solves
the inner dual from scratch at the perturbed eta.
"""
function d_delta_dual_d_eta_nu_fd_local(x_free0, νvec, ctx_cm, cctx; h::Float64 = 1e-4)
    n = length(νvec)
    g = Vector{Float64}(undef, n)
    η = log.(νvec)
    for j in 1:n
        ηp = copy(η); ηp[j] += h
        ηm = copy(η); ηm[j] -= h
        _, vp = archC_meanzc_verified_state(x_free0, exp.(ηp), ctx_cm, cctx)
        _, vm = archC_meanzc_verified_state(x_free0, exp.(ηm), ctx_cm, cctx)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
    end
    return g
end

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D4 CM+ZC-CROSS gradient verification K_mean=$K_mean K_pair=$K_pair ====")
    νvec0 = nu0_shared(K_mean)
    pcx = build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs)

    base, verify = archC_meanzc_verified_state(x_free_calib, νvec0, pcx.ctx_cm, pcx.cctx)
    check("K=$K_mean/$K_pair: inner solve converged", verify.inner_status in (0, -100, -101, -103))

    # ---- Check (B) FIRST (see header for why the ordering is load-bearing) ----
    cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
    cm_contrib0 = cm_fixed_contribution_meanzc_layout(base, ctx, pcx.aug, pcx.bins)
    meanzc_contrib0 = meanzc_cross_fixed_contribution(base, pcx.aug, νvec0)
    q0_via_fold = cache0.q0 .- cm_contrib0 .- meanzc_contrib0

    st = pcx.cctx.cmlookup_st   # populated by archC_meanzc_verified_state's own inner solve above
    q0_via_operator = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
    d_q0 = maximum(abs.(q0_via_fold .- q0_via_operator))
    println("    (B) fixed-contribution-fold q0 vs independent operator-FG q0: max abs diff=$d_q0")
    check("K=$K_mean/$K_pair: (B) meanzc_cross_fixed_contribution matches independent operator recompute", d_q0 < 1e-8)

    # ---- Check (A): analytic eta-gradient vs independently reoptimized central FD ----
    # h=1e-6, NOT the 1e-4 used by OZC-CROSS's own equivalent gate. This objective is far steeper in
    # eta_nu here (|d Delta/d eta| ~ 1e1..1e3 against Delta ~ 1e-2), so central-difference truncation
    # error at h=1e-4 is ~3-5% RELATIVE -- large enough to look like a real bug, and it was
    # initially mistaken for one. Established by an explicit h-sweep run against BOTH this family
    # and the UNMODIFIED base diagonal family at the identical points
    # (control_base_cmzc_etagrad_fd_d4_2026-08-09.jl): the gap falls as O(h^2) --
    # 3.3e-2 (h=1e-4) -> 3.3e-4 (1e-5) -> 3.3e-6 (1e-6) -> 3.4e-8 (1e-7) -- with the SAME signature
    # in both families, i.e. pure FD truncation error, not an analytic-formula error (a formula
    # error would be flat in h). Tolerance below is set at 1e-4 relative, ~30x the observed h=1e-6
    # residual, so it is a real gate rather than a rubber stamp. Do NOT raise h back to 1e-4.
    # (h=1e-3 is unusable at K=3/3: the perturbed nu leaves the feasible set and the inner solve
    # returns nStatus=-300, a genuine infeasibility certificate -- also confirmed in the base-family
    # control, so it is a property of the problem, not of this family.)
    g_analytic = d_delta_dual_d_eta_nu_cross_vec(base.λstar, pcx.aug, νvec0; mean_m = verify.m_mean)
    g_fd = d_delta_dual_d_eta_nu_fd_local(x_free_calib, νvec0, pcx.ctx_cm, pcx.cctx; h = 1e-6)
    d_ag = maximum(abs.(g_analytic .- g_fd))
    rel = d_ag / max(1e-8, maximum(abs.(g_fd)))
    println("    (A) analytic eta-grad vs FD (h=1e-6): max abs diff=$d_ag  max rel diff=$rel")
    println("        analytic=$g_analytic")
    println("        fd      =$g_fd")
    check("K=$K_mean/$K_pair: (A) analytic eta-gradient matches reoptimized FD", rel < 1e-4)

    # ---- Check (A2): the cross formula must reduce EXACTLY to the base family's diagonal formula
    # when K_pair==1 (the only combo is (1,1), so accumulate-both-slots collapses to -2*nu_1*sum).
    # A direct, zero-tolerance structural check that the restructure did not change the diagonal
    # case, run against the BASE family's own unmodified function on the SAME lambda vector.
    if K_pair == 1
        g_base_formula = d_delta_dual_d_eta_nu_vec(base.λstar, pcx.aug, νvec0; mean_m = verify.m_mean)
        d_red = maximum(abs.(g_analytic .- g_base_formula))
        println("    (A2) cross formula vs base diagonal formula at K_pair=1: max abs diff=$d_red")
        check("K=$K_mean/$K_pair: (A2) cross formula reduces to base diagonal formula at K_pair=1", d_red < 1e-12)
    end
end

# ---- Check (C): VARIANT D -- d_delta_d_nu_star under the shared cross layout ----
# kstar=2 needs K_mean>=2. The focal origin is ctx.bi (the same one the production driver uses).
for (K_mean, K_pair) in [(2, 2), (3, 3)]
    kstar = 2
    println("\n==== D4 CM+ZC-CROSS Variant D verification K_mean=$K_mean K_pair=$K_pair kstar=$kstar ====")
    νvec0 = nu0_shared(K_mean)
    layout = SharedByPowerCrossLayout(K_mean, K_pair)
    aml = ActiveMeanLayout(layout, ctx.bi, kstar, D)
    check("K=$K_mean/$K_pair: aml.active", aml.active)
    check("K=$K_mean/$K_pair: dense_omit_idx == kstar (shared layout property)", aml.dense_omit_idx == kstar)
    check("K=$K_mean/$K_pair: n_eta_active == K_mean-1", aml.n_eta_active == K_mean - 1)

    pcx = build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs, aml = aml)
    check("K=$K_mean/$K_pair: n_mean is the ragged ROW count (K_mean*D - 1), not n_eta_active",
          pcx.aug.n_mean == K_mean * D - 1)

    base, verify = archC_meanzc_verified_state(x_free_calib, νvec0, pcx.ctx_cm, pcx.cctx)
    println("    inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
    if verify.inner_status ∉ (0, -100, -101, -103)
        println("    SKIP (C) at K=$K_mean/$K_pair: aml-active inner solve did not converge (nStatus=$(verify.inner_status)) -- ",
                "cannot FD-check a gradient at a point with no solution. Note D4's own sigmaHat=2.5 makes ",
                "sigma-1=1.5 non-integer, so kstar=2 is an ARTIFICIAL choice here (the real kstar=sigma-1=2 ",
                "test is at D20/sigmaHat=3.0) -- see memory ozc-cross-kpair2-grid-build for the identical ",
                "situation in OZC-CROSS's own D4 Variant D check.")
        continue
    end

    eta_grad_active, d_delta_d_nu_star = d_delta_dual_d_eta_active_and_nustar_shared_cross(
        base.λstar, pcx.aug, aml, νvec0; mean_m = verify.m_mean)
    check("K=$K_mean/$K_pair: eta_grad_active length == n_eta_active", length(eta_grad_active) == aml.n_eta_active)

    # FD reference over EVERY dense eta coordinate against this SAME aml-active context (so the
    # ragged lambda layout is the one the analytic function indexes into). gather_active_grad then
    # splits the dense FD gradient the same way the analytic function splits its own.
    g_dense_fd = d_delta_dual_d_eta_nu_fd_local(x_free_calib, νvec0, pcx.ctx_cm, pcx.cctx; h = 1e-6)   # see check (A)'s comment on why 1e-6, not 1e-4
    fd_active, fd_omit_eta = gather_active_grad(aml, g_dense_fd)
    d_act = maximum(abs.(eta_grad_active .- fd_active))
    rel_act = d_act / max(1e-8, maximum(abs.(fd_active)))
    @printf("    (C) active eta-grad vs FD (h=1e-6): max abs diff=%.4e  rel=%.4e\n", d_act, rel_act)
    check("K=$K_mean/$K_pair: (C) Variant D active eta-gradient matches FD", rel_act < 1e-4)

    # d_delta_d_nu_star is the RAW d(Delta)/d(nu_star); the FD probe is in eta-space at that same
    # dense slot, so divide out the chain-rule factor nu_star = nu_{kstar}.
    fd_nu_star = fd_omit_eta / νvec0[kstar]
    d_ns = abs(d_delta_d_nu_star - fd_nu_star)
    rel_ns = d_ns / max(1e-8, abs(fd_nu_star))
    @printf("    (C) d_delta_d_nu_star: analytic=%.10e  fd=%.10e  absdiff=%.4e  rel=%.4e\n",
            d_delta_d_nu_star, fd_nu_star, d_ns, rel_ns)
    check("K=$K_mean/$K_pair: (C) d_delta_d_nu_star matches FD (the cross-coupling term)", rel_ns < 1e-4)

    # Structural cross-check on the claim in the docstring: nu_{k*} genuinely enters 2*K_pair-1
    # distinct cross blocks. Counted directly off cross_pair_level_index, not asserted from prose.
    nblocks = count(kk -> kk[1] == kstar || kk[2] == kstar, cross_pair_level_index(K_pair))
    check("K=$K_mean/$K_pair: nu_kstar enters 2*K_pair-1 = $(2*K_pair-1) cross blocks (counted=$nblocks)",
          nblocks == 2 * K_pair - 1)
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
