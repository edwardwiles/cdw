# ================================================================================================
# Reoptimized-FD gate for the CM + pairwise-quantile family's OUTER gradient (family #7, 2026-08-12).
#
# The closed form (`cm_pq_dC_dmu` -> `d_delta_dual_d_mu_shared` -> `chain_cmpq_mass_gradient_to_raw`)
# was already gated in the standalone dense oracle against a FIXED-DUAL finite difference. That is
# necessary but not sufficient: it tests the algebra at an arbitrary `(zeta, lambda)`, not the claim
# that actually matters, which is the ENVELOPE-THEOREM one --
#
#     d(Delta*)/d(mu_c)  =  -mean_m * A_c   evaluated at the OPTIMAL (zeta*, lambda*)
#
# where the true derivative RE-SOLVES the inner problem at each probe. This file supplies that
# ground truth: every FD probe is a full, fresh `hessopt=exact` KNITRO solve.
#
# It is the direct analogue of `test_pairwise_quantile_outer_gradient_fd.jl` section 2 (the
# standalone family reaches 4.1e-10 there), with two differences that follow from this family's own
# structure rather than from any weakening of the bar:
#   * the coordinate vector is `L-1` long, not `D*(L-1)` -- the outer collapse this family exists for;
#   * the enforced point is a NON-UNIFORM mu, because at `mu = 1/L` the two-slot product rule's
#     partner index is provably unobservable (measured at 1.3e-9 in the dense oracle) and a gate run
#     there would pass with the index written the wrong way.
#
# The h-ladder exists for the established reason (memory `feedback-fd-bandwidth-mismatch-looks-like-a-bug`):
# a central FD carries truncation error falling as h^2 and inner-solver-tolerance noise rising as
# 1/h, so the response to disagreement is to move h, never to widen the threshold.
#
# TWO NEGATIVE CONTROLS, both of which must FAIL the same gate:
#   (a) a sign flip -- there is no negation at the production layer, and if one is "restored" this
#       must anti-correlate;
#   (b) the one-slot product rule (dropping the b-slot of `d(mu_a*mu_b)/d(mu_c)`) -- the natural,
#       plausible-looking error this family's own docstring warns about.
#
# Usage:  julia --project=. --threads=N full_aod_diag/d4_exact/test_cm_pairwise_quantile_outer_gradient_fd.jl \
#              <L> <G> <n_families> <contrasts>
#   e.g.  ... 5 50 2 orthonormal
# ================================================================================================

const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    flush(stdout)
    return ok
end

const USAGE = "usage: julia ... test_cm_pairwise_quantile_outer_gradient_fd.jl <L> <G> " *
              "<n_families> <contrasts:anchored|orthonormal>"
length(ARGS) == 4 || error(USAGE)
const L_ARG     = parse(Int, ARGS[1])
const G_ARG     = parse(Int, ARGS[2])
const NFAM_ARG  = parse(Int, ARGS[3])
const CONTR_ARG = Symbol(ARGS[4])
CONTR_ARG in (:anchored, :orthonormal) || error(USAGE)

println("=== CM + pairwise-quantile OUTER gradient (shared free masses): reoptimized-FD gate ===")
@printf("L=%d  G=%d  families=%d  contrasts=%s\n", L_ARG, G_ARG, NFAM_ARG, String(CONTR_ARG))
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
const PROD_OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")
cfg = CMPairwiseQuantileConfig(L = L_ARG, cm_grid_size = G_ARG, cm_moment_families = NFAM_ARG,
                               contrasts = CONTR_ARG, min_bin_count = 1, mass_start = :uniform)
cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = PROD_OPT)
ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx = true)
st = ctx_cm.cmpq_fg_state
obj = ctx_cm.obj
nc = L_ARG - 1
ncore1 = cmpq.ncore1
W = cmpq.op.W
@printf("D=%d  W=%d  n_x=%d  outer mass coords=%d (standalone PQ would need %d)\n",
        ctx.D, W, obj.outer_constr_index, nc, nc * ctx.D); flush(stdout)

"""
One full inner solve at the given raw mass coordinates. Returns `Delta* = -f(x*)`, the optimal
duals, `mean_m = (1/W) sum_w Psi'(r_w)` at that optimum, and the decoded `mu` -- everything the
closed form needs, recomputed through the FG functor itself at `x*` so nothing is read from a
stale callback buffer.
"""
function solve_at(raw::Vector{Float64})
    nStatus, xsol, objb, n_fg, n_hess = archCMPQ_base_state(x_free_calib, raw, ctx, ctx_cm;
        hess_cb_builder = cmpq_hess_builder_for(ctx_cm))
    f = st(xsol)                                  # sets st.arg0/obj.arg0 = r(x*)
    psi1 = similar(st.arg1)
    obj.dPsi!(psi1, st.arg0)
    mean_m = sum(psi1) / W
    lam_L, lam_P, _ = reshape_cmpq_duals(xsol, cmpq.op, ncore1)
    return (Delta = -f, x = copy(xsol), lam_L = collect(lam_L), lam_P = Array(lam_P),
            mean_m = mean_m, mu = copy(st.mass_state.mu), nStatus = nStatus,
            n_fg = n_fg, n_hess = n_hess)
end

"Analytic outer gradient in the RAW coordinates at a solved point."
analytic_grad(s, raw) = chain_cmpq_mass_gradient_to_raw(
    d_delta_dual_d_mu_shared(s.lam_L, s.lam_P, s.mu, cmpq.op; mean_m = s.mean_m), raw, s.mu)

"Reoptimized central FD of `Delta*` in the raw coordinates: every probe is a fresh KNITRO solve."
function reoptimized_fd(raw::Vector{Float64}, h::Float64)
    g = zeros(length(raw)); statuses = Int[]
    for k in eachindex(raw)
        rp = copy(raw); rp[k] += h
        rm = copy(raw); rm[k] -= h
        sp = solve_at(rp); sm = solve_at(rm)
        push!(statuses, sp.nStatus); push!(statuses, sm.nStatus)
        g[k] = (sp.Delta - sm.Delta) / (2h)
    end
    return g, statuses
end

function run_point(label::AbstractString, raw::Vector{Float64}, enforce::Bool, hs)
    println("\n", "-"^92)
    println("POINT: ", label)
    println("-"^92); flush(stdout)
    t0 = time()
    s = solve_at(raw)
    @printf("  base solve: nStatus=%d  n_fg=%d  n_hess=%d  Delta*=%.12g  mean_m=%.8g  (%.2fs)\n",
            s.nStatus, s.n_fg, s.n_hess, s.Delta, s.mean_m, time() - t0)
    @printf("  mu = [%s]\n", join((@sprintf("%.5f", v) for v in s.mu), ", ")); flush(stdout)
    check("[$label] base inner solve reached an expected status",
          s.nStatus in (0, -100, -101, -102, -103), "nStatus=$(s.nStatus)")
    check("[$label] the exact-Hessian callback ran in the base solve", s.n_hess > 0,
          "n_hess=$(s.n_hess)")
    if enforce && nc >= 2
        check("[$label] mu is genuinely NON-UNIFORM (the gate is blind at mu=1/L)",
              maximum(s.mu) - minimum(s.mu) > 0.01,
              @sprintf("spread %.4f", maximum(s.mu) - minimum(s.mu)))
    end

    ga = analytic_grad(s, raw)
    check("[$label] analytic gradient is all-finite", all(isfinite, ga))

    best = (h = NaN, rel = Inf, cos = NaN, g = fill(NaN, nc))
    @printf("  %10s %14s %14s %16s %9s\n", "h", "rel L2", "max|diff|", "cosine", "secs")
    for h in hs
        t = time()
        gf, sts = reoptimized_fd(raw, h)
        allok = all(x -> x in (0, -100, -101, -102, -103), sts)
        rel = norm(ga .- gf) / max(norm(gf), eps())
        cosang = dot(ga, gf) / (norm(ga) * norm(gf) + 1e-300)
        @printf("  %10.0e %14.3e %14.3e %16.12f %9.1f%s\n", h, rel, maximum(abs, ga .- gf), cosang,
                time() - t, allok ? "" : "   <- SOME PROBE SOLVES FAILED")
        flush(stdout)
        allok && rel < best.rel && (best = (h = h, rel = rel, cos = cosang, g = gf))
    end
    println("  best: h=", best.h, "  rel L2=", best.rel, "  cosine=", best.cos)
    @printf("  %6s %18s %18s %10s\n", "coord", "analytic", "reopt-FD", "ratio")
    for k in 1:nc
        @printf("  %6d %18.10e %18.10e %10.6f\n", k, ga[k], best.g[k],
                best.g[k] == 0.0 ? NaN : ga[k] / best.g[k])
    end

    if enforce
        check("[$label] cosine(analytic, reoptimized FD) > 0.9999", best.cos > 0.9999,
              @sprintf("cosine=%.12f", best.cos))
        check("[$label] relative L2 error < 1e-6", best.rel < 1e-6, @sprintf("rel=%.3e", best.rel))
        # (a) sign
        cos_flip = dot(-ga, best.g) / (norm(ga) * norm(best.g) + 1e-300)
        check("[$label] NEGATIVE CONTROL: a sign-flipped gradient anti-correlates with FD",
              cos_flip < -0.9999, @sprintf("cosine_flipped=%.12f", cos_flip))
        # (b) one-slot product rule -- the natural, plausible-looking transcription error
        if nc >= 2
            A_one = copy(s.lam_L)
            for pidx in 1:cmpq.op.npair, b in 1:nc, a in 1:nc
                A_one[a] += s.mu[b] * s.lam_P[a, b, pidx]      # b-slot deliberately dropped
            end
            g_one = chain_cmpq_mass_gradient_to_raw(-s.mean_m .* A_one, raw, s.mu)
            rel_one = norm(g_one .- best.g) / max(norm(best.g), eps())
            check("[$label] NEGATIVE CONTROL: the one-slot product rule fails the same gate",
                  rel_one > 1e-2, @sprintf("rel_one_slot=%.3e", rel_one))
        else
            println("  SKIP  one-slot negative control: L=$L_ARG gives one free bin, so the a- and " *
                    "b-slots are the same coordinate and the error is not distinguishable")
        end
    end
    return nothing
end

raw_uniform = cmpq_uniform_mass_raw(L_ARG)
# The ENFORCED point: masses pushed off uniform, differently per bin.
mu_off = [0.30, 0.10, 0.25, 0.08, 0.12, 0.06, 0.04, 0.02, 0.015][1:nc]
mu_off ./= (sum(mu_off) / 0.9)
raw_off = zeros(nc); raw_from_origin_masses!(raw_off, mu_off)

const HS = (1e-3, 1e-4, 1e-5)
run_point("mu = 1/L (the campaign's own start; REPORTED, not enforced -- blind by construction)",
          raw_uniform, false, HS)
run_point("NON-UNIFORM mu (ENFORCED)", raw_off, true, HS)

println("\n", "="^92)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^92)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_outer_gradient_fd: $(NFAIL[]) check(s) failed")
