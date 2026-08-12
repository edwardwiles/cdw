# ================================================================================================
# D=4 dense truth oracle for the CM + PAIRWISE-QUANTILE family (family #7, 2026-08-12).
#
# Gates the family's MATHEMATICAL CORE -- the moment rows, the forward/transpose contraction, the
# CM/PQ bin-superset property the whole construction rests on, the rank consequences of dropping the
# per-origin marginal rows, and the shared-mu derivative -- against NAIVE DENSE references built
# directly from raw W x n_rows indicator columns. No KNITRO, no economic block, no lookup tricks on
# the reference side.
#
# WHY THIS IS THE RIGHT FIRST GATE. The family's whole claim to be "CM + a shared level" rather than
# "something weaker that resembles it" is a statement about SPANS: the dropped per-origin marginal
# rows must be exactly implied by CM plus the reference-level rows. That is checkable numerically and
# decisively (CHECK 6), and it is checked here BEFORE any inner-solve plumbing exists, because if it
# fails, every downstream number is silently answering a different question.
#
# CHECKS
#   1  PQ cutoffs are SELECTED from CM's own threshold array (bit-identical), and agree with the
#      standalone family's independent closed form to floating point.
#   2  PQ bin assignment agrees between (a) z-space `searchsortedfirst` on Q -- the standalone
#      family's own unmodified code path -- and (b) CM's own bin index pushed through a pure INTEGER
#      map. Every (draw,origin) cell.
#   3  `cm_pq_forward!` == dense `-(G_R * lambda_R)`, at NON-UNIFORM mu.
#   4  `cm_pq_transpose!` == dense `-(1/W) * G_R' * psi`, at NON-UNIFORM mu.
#   5  CM's production lookup kernels == dense CM columns (`precalc_common_marginals_cdf`), forward
#      and transpose, for eq.35 and (when families=2) eq.36.
#   6  THE REDUNDANCY CLAIM, three ways:
#        6a  [CM | level | pair] has FULL column rank (the family as built is not singular);
#        6b  every DROPPED per-origin marginal column lies in span([CM | level]) -- LS residual ~ 0;
#        6c  putting all D per-origin marginal columns BACK makes the stack rank deficient by
#            EXACTLY D*(L-1) -- (D-1)*(L-1) implied non-reference rows plus (L-1) that duplicate the
#            level rows outright. This is the "singular KKT, not a tolerance problem" trap of the
#            handover's §5.1, measured rather than asserted.
#   7  `cm_pq_dC_dmu` (the two-slot shared-mu term) == FD of `C_lambda`, at NON-UNIFORM mu.
#   8  `d(fixed-dual f)/dmu` == `+mean_m * A` (so `d_delta_dual_d_mu_shared`'s negation is right), and
#      the chain rule to raw coordinates == FD through the stick-breaking decode.
#   9  NEGATIVE CONTROLS on the two-slot term: the natural wrong transcriptions must FAIL at
#      non-uniform mu -- and the index-slip variant must PASS at uniform mu, which is the measured
#      demonstration that a uniform-mu gate would have been fooled (handover §5.2).
#
# `L`/`G` generic: run at several (L,G) pairs with L | G. Usage:
#   julia --project=. full_aod_diag/d4_exact/test_cm_pairwise_quantile_d4_dense_oracle.jl
#
# SCOPE NOTE: validates this family's own numerics in isolation on synthetic draws. It does NOT
# exercise the economic H_EE/cross blocks or a live inner solve -- those are separate gates.
# Production code must NEVER include this file.
# ================================================================================================

const D4X = @__DIR__
using Parameters       # compressed_moments.jl -> winners.jl needs @unpack
using Random, LinearAlgebra, Printf

# `packed_pair_index`/`pair_oi_to_lin`/`frechet_power_feature` are defined here BEFORE the includes
# so that common_marginals_moments.jl's own `isdefined(Main,:frechet_power_feature) || include(...)`
# guard is satisfied without pulling cm_meanzc_moments.jl. Copied byte-for-byte from
# cm_meanzc_moments.jl, exactly as test_pairwise_quantile_d4_dense_oracle.jl already does and for the
# same reason (that file's own note applies verbatim). Production NEVER duplicates these.
function packed_pair_index(D::Int)
    pairs = Vector{Tuple{Int,Int}}(undef, div(D * (D - 1), 2))
    k = 0
    for o in 1:D-1, p in o+1:D
        k += 1
        pairs[k] = (o, p)
    end
    return pairs
end
function pair_oi_to_lin(o::Int, p::Int, D::Int)
    o, p = o < p ? (o, p) : (p, o)
    return div((o - 1) * (2D - o), 2) + (p - o)
end
function frechet_power_feature(U::AbstractMatrix{Float64}, k::Real, μ::Float64)
    Q = similar(U)
    @inbounds @. Q = exp(-μ * k * log(U))
    return Q
end
"Psi/Psi' at a single point, matching cc_algo/Psi.jl exactly (test-local; production goes through obj.Psi!)."
psi_scalar(x::Float64) = x <= 1.0 ? (exp(x) - 1.0) : (0.5 * exp(1) * (x^2 + 1.0) - 1.0)
dpsi_scalar(x::Float64) = x <= 1.0 ? exp(x) : (exp(1) * x)

for f in ["common_marginals_moments.jl", "common_marginals_interval.jl", "cm_lookup_kernels.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl"]
    include(joinpath(D4X, f))
end

const NFAIL = Ref(0)
const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    if ok
        NPASS[] += 1
        println("  PASS  ", name, isempty(detail) ? "" : "   [$detail]")
    else
        NFAIL[] += 1
        println("  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    end
    return ok
end
relerr(a, b) = norm(a .- b) / max(1.0, norm(b))

# ------------------------------------------------------------------------------------------------
# Dense (naive) references -- built from raw indicator columns, no reuse of any production kernel.
# ------------------------------------------------------------------------------------------------

"""
Dense `W x n_cmpq_restr_rows` moment matrix for this family's OWN block, from raw indicators:
column `cmpq_level_row(a)`   = `1{b_ref=a} - mu_a`
column `cmpq_pair_row(...)`  = `1{b_o=a, b_p=b} - mu_a*mu_b`
Written straight from the definition, deliberately without touching op-derived tables.
"""
function dense_cmpq_restriction_matrix(bin::AbstractMatrix{<:Integer}, pairs, D::Int, L::Int,
                                       mu::AbstractVector{Float64}, ref::Int)
    W = size(bin, 1); nc = L - 1
    nrow = n_cmpq_restr_rows(D, L)
    G = zeros(W, nrow)
    for a in 1:nc
        col = cmpq_level_row(a)
        for w in 1:W
            G[w, col] = (Int(bin[w, ref]) == a ? 1.0 : 0.0) - mu[a]
        end
    end
    for (pidx, (o, p)) in enumerate(pairs), b in 1:nc, a in 1:nc
        col = cmpq_pair_row(D, pidx, a, b, L)
        tgt = mu[a] * mu[b]
        for w in 1:W
            G[w, col] = ((Int(bin[w, o]) == a && Int(bin[w, p]) == b) ? 1.0 : 0.0) - tgt
        end
    end
    return G
end

"Dense per-origin marginal columns `1{b_o=a} - mu_a` (o = 1..D), the block this family DROPS."
function dense_per_origin_marginal_matrix(bin::AbstractMatrix{<:Integer}, D::Int, L::Int,
                                          mu::AbstractVector{Float64})
    W = size(bin, 1); nc = L - 1
    G = zeros(W, nc * D)
    for o in 1:D, a in 1:nc
        col = (o - 1) * nc + a
        for w in 1:W
            G[w, col] = (Int(bin[w, o]) == a ? 1.0 : 0.0) - mu[a]
        end
    end
    return G
end

"`C_lambda` straight from its definition -- the scalar `cm_pq_forward!` hoists out of the draw loop."
function dense_C_lambda(lambda_L, lambda_P, mu, pairs)
    nc = length(mu)
    C = 0.0
    for a in 1:nc
        C += lambda_L[a] * mu[a]
    end
    for pidx in 1:length(pairs), b in 1:nc, a in 1:nc
        C += lambda_P[a, b, pidx] * mu[a] * mu[b]
    end
    return C
end

# ------------------------------------------------------------------------------------------------
# One full configuration
# ------------------------------------------------------------------------------------------------

function run_case(; D::Int, W::Int, L::Int, G::Int, n_families::Int, seed::Int,
                    muHat::Float64, sigmaHat::Float64, ref::Int)
    println("\n", "="^96)
    @printf("CASE  D=%d W=%d  L=%d (PQ bins)  G=%d (CM k/G grid, %d levels)  families=%d  ref=%d\n",
            D, W, L, G, G - 1, n_families, ref)
    println("="^96)

    cfg = CMPairwiseQuantileConfig(L = L, cm_grid_size = G, cm_moment_families = n_families,
                                   contrasts = :anchored, min_bin_count = 5, mass_start = :uniform)
    rc = resolve_cm_pairwise_quantile_config(cfg)
    check("config resolves; L | G", rc.ratio * L == G, "ratio=$(rc.ratio)")

    rng = MersenneTwister(seed)
    U = -log.(1 .- rand(rng, W, D))          # Exp(1), exactly as draw_design.jl builds it
    Z = frechet_power_feature(U, 1, muHat)   # Frechet productivity z = U^(-muHat)

    # ---- CM side: dense reference on the EXPLICIT k/G grid -------------------------------------
    CMd, z_cm, origins = precalc_common_marginals_cdf(U, ref, rc.n_cm_levels;
        include_truncated_moment = (n_families == 2), σHat = sigmaHat, μHat = muHat,
        contrasts = rc.contrasts, probs = rc.cm_probs)
    nO = length(origins)
    ncm = size(CMd, 2)
    check("CM width == n_cm_moments", ncm == n_cm_moments(D, rc.n_cm_levels;
            include_truncated_moment = (n_families == 2)), "ncm=$ncm")
    check("CM thresholds are the theoretical k/G ones",
          z_cm == theoretical_u_threshold.(rc.cm_probs))

    # ---- PQ side: cutoffs SELECTED from CM's own thresholds ------------------------------------
    c_u = cm_pq_u_cutoffs_from_cm_grid(z_cm, G, L)
    Q = cm_pq_z_cutoffs_from_u(c_u, D; mu_frechet = muHat)
    op = PairwiseQuantileOperator(Z, L, Q)

    # CHECK 1 -- selection is bit-identical, and matches the independent closed form.
    bitident = all(c_u[r] === z_cm[cm_pq_grid_index(G, L, r)] for r in 1:(L-1))
    check("1a  U-cutoffs are bit-identical selections from CM's z array", bitident)
    closed = [(-log(r / L))^(-muHat) for r in 1:(L-1)]     # standalone family's own formula
    e1 = maximum(abs.(Q[:, 1] .- closed) ./ abs.(closed))
    check("1b  z-cutoffs match the standalone closed form", e1 < 1e-14, @sprintf("max rel %.2e", e1))

    # CHECK 2 -- bin agreement between the two independent routes, every (draw,origin) cell.
    Bidx = compute_bin_indices(U, z_cm)
    consistent = try
        r = assert_cm_pq_bin_consistency(op, Bidx, G)
        check("2   PQ bins agree: z-space searchsortedfirst vs CM's integer bin map",
              true, "$(r.n_checked) cells")
        true
    catch e
        check("2   PQ bins agree: z-space searchsortedfirst vs CM's integer bin map", false,
              sprint(showerror, e)[1:min(200, end)])
        false
    end
    occ = assert_pairwise_quantile_bins_nondegenerate(op; min_bin_count = cfg.min_bin_count)
    check("2b  bins non-degenerate", true,
          "min marginal=$(occ.min_marginal_count) min joint=$(occ.min_joint_count)")

    # ---- a deliberately NON-UNIFORM mu (see cm_pq_dC_dmu's docstring) --------------------------
    nc = L - 1
    mu_nonunif = [0.30, 0.10, 0.25, 0.08, 0.12, 0.06, 0.04, 0.02, 0.015, 0.011, 0.009, 0.007,
                  0.006, 0.005, 0.004, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0008, 0.0006,
                  0.0005, 0.0004][1:nc]
    sum(mu_nonunif) < 1.0 || error("test bug: non-uniform mu is off the simplex")
    raw_nonunif = zeros(nc); raw_from_origin_masses!(raw_nonunif, mu_nonunif)
    state = CMPQMassState(L)
    set_cmpq_masses!(state, raw_nonunif)
    check("3a  decode round-trips the non-uniform mu",
          maximum(abs.(state.mu .- mu_nonunif)) < 1e-13,
          @sprintf("max abs %.2e", maximum(abs.(state.mu .- mu_nonunif))))

    npair = op.npair
    lambda_L = randn(rng, nc)
    lambda_P = randn(rng, nc, nc, npair)
    lambda_cm = randn(rng, ncm)

    # CHECK 3 -- forward vs dense
    G_R = dense_cmpq_restriction_matrix(op.bin, op.pairs, D, L, state.mu, ref)
    lam_flat = zeros(n_cmpq_restr_rows(D, L))
    for a in 1:nc
        lam_flat[cmpq_level_row(a)] = lambda_L[a]
    end
    for pidx in 1:npair, b in 1:nc, a in 1:nc
        lam_flat[cmpq_pair_row(D, pidx, a, b, L)] = lambda_P[a, b, pidx]
    end
    fwd_dense = -(G_R * lam_flat)
    fwd_op = zeros(W)
    cm_pq_forward!(fwd_op, lambda_L, lambda_P, op, state, ref)
    e3 = relerr(fwd_op, fwd_dense)
    check("3b  cm_pq_forward! == -(G_R*lambda)", e3 < 1e-12, @sprintf("rel L2 %.3e", e3))

    # CHECK 4 -- transpose vs dense
    psi = dpsi_scalar.(fwd_dense .+ 0.3)      # arbitrary strictly-positive weights, both branches of Psi'
    g_L = zeros(nc); g_P = zeros(nc, nc, npair)
    tls = build_pairwise_quantile_thread_scratch(D, npair, L)
    scr = PairwiseQuantileTransposeScratch(D, npair, L)
    cm_pq_transpose!(g_L, g_P, psi, op, state, ref, tls, scr)
    gt_dense = -(G_R' * psi) ./ W
    g_flat = zeros(n_cmpq_restr_rows(D, L))
    for a in 1:nc
        g_flat[cmpq_level_row(a)] = g_L[a]
    end
    for pidx in 1:npair, b in 1:nc, a in 1:nc
        g_flat[cmpq_pair_row(D, pidx, a, b, L)] = g_P[a, b, pidx]
    end
    e4 = relerr(g_flat, gt_dense)
    check("4   cm_pq_transpose! == -(1/W)*G_R'*psi", e4 < 1e-12, @sprintf("rel L2 %.3e", e4))

    # CHECK 5 -- CM's production lookup kernels vs dense CM columns
    R = rc.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Lcm = rc.n_cm_levels
    ncm_cdf = nO * Lcm
    λmat_block = zeros(nO, Lcm); λmat_ext = zeros(nO, Lcm + 1); cm_contrib = zeros(W)
    apply_contrast!(λmat_block, reshape(@view(lambda_cm[1:ncm_cdf]), nO, Lcm), R)
    suffix_sums!(λmat_ext, λmat_block)
    cumulative_forward_contribution!(cm_contrib, Bidx, ref, origins, λmat_ext)
    cm_fwd_dense = CMd[:, 1:ncm_cdf] * lambda_cm[1:ncm_cdf]
    e5 = relerr(cm_contrib, cm_fwd_dense)
    check("5a  CM lookup forward == dense CM columns (eq.35)", e5 < 1e-12, @sprintf("rel L2 %.3e", e5))

    nbins = Lcm + 1
    hist_partials = [zeros(D, nbins)]; hist_h = zeros(D, nbins); Hpre = zeros(D, Lcm)
    g_block = zeros(nO, Lcm); g_stored = zeros(nO, Lcm)
    build_weighted_histogram!(hist_h, hist_partials, Bidx, psi, D, nbins)
    prefix_sums!(Hpre, hist_h, Lcm)
    cumulative_backward_gradient_from_prefix!(g_block, Hpre, ref, origins, Lcm, W)
    apply_contrast!(g_stored, g_block, R)
    cm_gt_dense = -(CMd[:, 1:ncm_cdf]' * psi) ./ W
    e5b = relerr(vec(g_stored), cm_gt_dense)
    check("5b  CM lookup transpose == dense CM columns (eq.35)", e5b < 1e-12, @sprintf("rel L2 %.3e", e5b))

    if n_families == 2
        Pow = frechet_power_feature(U, sigmaHat - 1, muHat)
        λb2 = zeros(nO, Lcm); λe2 = zeros(nO, Lcm + 1); cc2 = zeros(W)
        apply_contrast!(λb2, reshape(@view(lambda_cm[ncm_cdf+1:2*ncm_cdf]), nO, Lcm), R)
        suffix_sums!(λe2, λb2)
        cumulative_forward_contribution_pow!(cc2, Bidx, ref, origins, λe2, Pow)
        e5c = relerr(cc2, CMd[:, ncm_cdf+1:2*ncm_cdf] * lambda_cm[ncm_cdf+1:2*ncm_cdf])
        check("5c  CM lookup forward == dense CM columns (eq.36)", e5c < 1e-12, @sprintf("rel L2 %.3e", e5c))
    end

    # CHECK 6 -- THE REDUNDANCY CLAIM
    stack_cl = hcat(CMd, G_R[:, 1:nc])                 # [CM | level]
    stack_full = hcat(CMd, G_R)                        # [CM | level | pair]
    rk_full = rank(stack_full; rtol = 1e-10)
    check("6a  [CM | level | pair] has FULL column rank",
          rk_full == size(stack_full, 2), "rank=$rk_full of $(size(stack_full,2))")

    Gmarg = dense_per_origin_marginal_matrix(op.bin, D, L, state.mu)
    resid = Gmarg .- stack_cl * (stack_cl \ Gmarg)
    e6b = maximum(abs, resid) / max(1.0, maximum(abs, Gmarg))
    check("6b  every dropped per-origin marginal column lies in span([CM | level])",
          e6b < 1e-10, @sprintf("max rel LS residual %.2e", e6b))

    stack_with = hcat(stack_full, Gmarg)
    rk_with = rank(stack_with; rtol = 1e-10)
    deficiency = size(stack_with, 2) - rk_with
    check("6c  re-adding the per-origin marginal rows is deficient by EXACTLY D*(L-1)",
          deficiency == D * nc, "deficiency=$deficiency, D*(L-1)=$(D*nc), rank=$rk_with of $(size(stack_with,2))")

    # CHECK 7 -- the two-slot shared-mu term vs FD of C_lambda
    A = cm_pq_dC_dmu(lambda_L, lambda_P, state.mu, op)
    A_fd = zeros(nc)
    for c in 1:nc
        h = 1e-7 * max(1.0, abs(state.mu[c]))
        mp = copy(state.mu); mp[c] += h
        mm = copy(state.mu); mm[c] -= h
        A_fd[c] = (dense_C_lambda(lambda_L, lambda_P, mp, op.pairs) -
                   dense_C_lambda(lambda_L, lambda_P, mm, op.pairs)) / (2h)
    end
    e7 = relerr(A, A_fd)
    check("7   cm_pq_dC_dmu == FD of C_lambda (non-uniform mu)", e7 < 1e-6, @sprintf("rel L2 %.3e", e7))

    # CHECK 8 -- fixed-dual df/dmu == +mean_m*A, and the chain rule to raw
    zeta = 0.37
    function fixed_dual_f(raw::AbstractVector{Float64})
        st = CMPQMassState(L); set_cmpq_masses!(st, raw)
        a0 = fill(-zeta, W)
        cm_pq_forward!(a0, lambda_L, lambda_P, op, st, ref)
        return sum(psi_scalar.(a0)) / W + zeta
    end
    a0 = fill(-zeta, W)
    cm_pq_forward!(a0, lambda_L, lambda_P, op, state, ref)
    mean_m = sum(dpsi_scalar.(a0)) / W
    d_mu_analytic = mean_m .* A                       # df/dmu (NOT dDelta*/dmu, which negates)
    d_mu_fd = zeros(nc)
    for c in 1:nc
        # FD in mu-space, through the raw coordinates that decode to a perturbed mu
        h = 1e-7 * max(1.0, abs(state.mu[c]))
        for (sgn, dst) in ((1.0, 1), (-1.0, 2))
            mp = copy(state.mu); mp[c] += sgn * h
            rp = zeros(nc); raw_from_origin_masses!(rp, mp)
            v = fixed_dual_f(rp)
            dst == 1 ? (d_mu_fd[c] = v) : (d_mu_fd[c] -= v)
        end
        d_mu_fd[c] /= (2h)
    end
    e8 = relerr(d_mu_analytic, d_mu_fd)
    check("8a  d(fixed-dual f)/dmu == +mean_m * A", e8 < 1e-5, @sprintf("rel L2 %.3e", e8))
    dDelta = d_delta_dual_d_mu_shared(lambda_L, lambda_P, state.mu, op; mean_m = mean_m)
    check("8b  d_delta_dual_d_mu_shared == -(that)", relerr(dDelta, -d_mu_analytic) < 1e-13)
    g_raw = chain_cmpq_mass_gradient_to_raw(d_mu_analytic, raw_nonunif, state.mu)
    g_raw_fd = zeros(nc)
    for k in 1:nc
        h = 1e-6
        rp = copy(raw_nonunif); rp[k] += h
        rm = copy(raw_nonunif); rm[k] -= h
        g_raw_fd[k] = (fixed_dual_f(rp) - fixed_dual_f(rm)) / (2h)
    end
    e8c = relerr(g_raw, g_raw_fd)
    check("8c  chain rule to raw == FD through the stick-breaking decode", e8c < 1e-5,
          @sprintf("rel L2 %.3e", e8c))

    # CHECK 9 -- NEGATIVE CONTROLS on the two-slot term
    "WRONG variant 1: partner's mass read at the OWN bin index (mu[c]) instead of the partner's."
    function A_wrong_index(lambda_L, lambda_P, mu, op)
        Aw = copy(lambda_L)
        for pidx in 1:op.npair, b in 1:nc, a in 1:nc
            lp = lambda_P[a, b, pidx]
            Aw[a] += mu[a] * lp      # WRONG: should be mu[b]
            Aw[b] += mu[b] * lp      # WRONG: should be mu[a]
        end
        return Aw
    end
    "WRONG variant 2: only the a-slot kept (product rule truncated to one term)."
    function A_wrong_oneslot(lambda_L, lambda_P, mu, op)
        Aw = copy(lambda_L)
        for pidx in 1:op.npair, b in 1:nc, a in 1:nc
            Aw[a] += mu[b] * lambda_P[a, b, pidx]
        end
        return Aw
    end
    w2 = relerr(A_wrong_oneslot(lambda_L, lambda_P, state.mu, op), A_fd)
    check("9b  NEGATIVE CONTROL fires: one-slot product rule disagrees at non-uniform mu",
          w2 > 1e-2, @sprintf("rel L2 %.3f (want >> 0)", w2))
    # The INDEX-slip control needs at least two free bins to be a distinguishable error at all: at
    # L=2 there is one free bin, so `a == b == c == 1` always and `mu[a]`/`mu[b]` are the same
    # number. Reported as an explicit SKIP rather than allowed to pass vacuously -- a control that
    # cannot fire must not be counted as a control that did (this is the same failure mode as
    # memory `feedback-self-cancelling-test-convention-cannot-gate-a-sign`, one level up: there the
    # test convention cancelled the error, here the parameterization erases it).
    if nc >= 2
        w1 = relerr(A_wrong_index(lambda_L, lambda_P, state.mu, op), A_fd)
        check("9a  NEGATIVE CONTROL fires: wrong partner index disagrees at non-uniform mu",
              w1 > 1e-2, @sprintf("rel L2 %.3f (want >> 0)", w1))
        # ...and the demonstration that a UNIFORM-mu gate would have been fooled by 9a.
        mu_unif = fill(1.0 / L, nc)
        A_fd_u = zeros(nc)
        for c in 1:nc
            h = 1e-7
            mp = copy(mu_unif); mp[c] += h
            mm = copy(mu_unif); mm[c] -= h
            A_fd_u[c] = (dense_C_lambda(lambda_L, lambda_P, mp, op.pairs) -
                         dense_C_lambda(lambda_L, lambda_P, mm, op.pairs)) / (2h)
        end
        wu = relerr(A_wrong_index(lambda_L, lambda_P, mu_unif, op), A_fd_u)
        check("9c  and at UNIFORM mu that same wrong variant PASSES (the gate must be non-uniform)",
              wu < 1e-6, @sprintf("rel L2 %.2e (blind, as documented)", wu))
    else
        println("  SKIP  9a/9c index-slip control: L=$L gives one free bin, so mu[a] and mu[b] are " *
                "the same coordinate and the index slip is not a distinguishable error")
    end

    # ============================================================================================
    # CHECK 10-12 -- THE HESSIAN.
    # `r` is LINEAR in the inner variables, so the exact inner Hessian is the weighted Gram matrix
    # `H = (1/W) M' diag(h) M` with `M = [1 | E | G_R | G_CM]` and `h_w = Psi''(r_w)`. That gives a
    # dense reference for the restriction-side blocks with no derivative approximation at all: it is
    # the SAME identity the production code exploits, evaluated the slow, obvious way.
    # (Economic blocks are out of scope here, exactly as the standalone family's oracle scopes them
    # out -- they are the shared, separately-gated winner-pair backend.)
    # ============================================================================================
    ddpsi_scalar(x::Float64) = x <= 1.0 ? exp(x) : exp(1)
    r0 = copy(fwd_dense) .+ 0.3
    h = ddpsi_scalar.(r0)

    # ---- H_RR via the standalone family's machinery at a REPLICATED shared mu, then extracted ----
    npq = n_total_rows(D, L)
    nrow = n_cmpq_restr_rows(D, L)
    pq_state = PairwiseQuantileMassState(D, L)
    cmpq_replicate_shared_mu!(pq_state, state)
    check("10a replicated mu equals the shared mu in every origin row",
          all(pq_state.mu[o, a] == state.mu[a] for o in 1:D, a in 1:nc))
    tabs_pq = PairwiseQuantileHessianTables(op)
    build_pairwise_quantile_hessian_tables!(tabs_pq, op, h, tls)
    HRR_pq = zeros(npq, npq)
    fill_pairwise_quantile_hessian_raw!(HRR_pq, op, tabs_pq)
    center_and_scale_pairwise_quantile_hessian!(HRR_pq, op, pq_state, tabs_pq)
    HRR = zeros(nrow, nrow)
    extract_cmpq_HRR!(HRR, HRR_pq, D, L, ref)
    HRR_ref = (G_R' * (h .* G_R)) ./ W
    worst = 0.0
    for J in 1:nrow, I in J:nrow
        worst = max(worst, abs(HRR[I, J] - HRR_ref[I, J]))
    end
    scale = maximum(abs, HRR_ref)
    check("10b H_RR (PQ machinery @ replicated mu, extracted) == (1/W) G_R' diag(h) G_R",
          worst / scale < 1e-10, @sprintf("max abs %.3e (scale %.3e)", worst, scale))

    # ---- H_R,CM: the genuinely NEW block ---------------------------------------------------------
    tabs_x = CMPQCrossHessTables(D, npair, L, Lcm + 1; n_families = n_families)
    Pow_here = n_families == 2 ? frechet_power_feature(U, sigmaHat - 1, muHat) : nothing
    build_cmpq_cross_hess_tables!(tabs_x, op, Bidx, h, ref; Pow = Pow_here)
    HRC = zeros(nrow, ncm)
    fill_cmpq_cm_cross_block!(HRC, op, state, tabs_x, origins, ref, Lcm, R, W)
    HRC_ref = (G_R' * (h .* CMd)) ./ W
    e11 = maximum(abs, HRC .- HRC_ref) / max(1e-300, maximum(abs, HRC_ref))
    check("11  H_R,CM (NEW mixed-resolution block) == (1/W) G_R' diag(h) G_CM",
          e11 < 1e-10, @sprintf("max rel %.3e", e11))
    # Split the report by row family and by CM feature family, so a failure localizes immediately
    # instead of collapsing to one number.
    e11a = maximum(abs, HRC[1:nc, :] .- HRC_ref[1:nc, :]) / max(1e-300, maximum(abs, HRC_ref[1:nc, :]))
    e11b = maximum(abs, HRC[nc+1:end, :] .- HRC_ref[nc+1:end, :]) /
           max(1e-300, maximum(abs, HRC_ref[nc+1:end, :]))
    check("11a   ...level rows x CM", e11a < 1e-10, @sprintf("max rel %.3e", e11a))
    check("11b   ...pair rows x CM", e11b < 1e-10, @sprintf("max rel %.3e", e11b))
    if n_families == 2
        ncm_cdf = nO * Lcm
        e11c = maximum(abs, HRC[:, 1:ncm_cdf] .- HRC_ref[:, 1:ncm_cdf]) /
               max(1e-300, maximum(abs, HRC_ref[:, 1:ncm_cdf]))
        e11d = maximum(abs, HRC[:, ncm_cdf+1:end] .- HRC_ref[:, ncm_cdf+1:end]) /
               max(1e-300, maximum(abs, HRC_ref[:, ncm_cdf+1:end]))
        check("11c   ...eq.35 CM columns", e11c < 1e-10, @sprintf("max rel %.3e", e11c))
        check("11d   ...eq.36 CM columns (reflected 1{U>c} read)", e11d < 1e-10, @sprintf("max rel %.3e", e11d))
    end

    # ---- CHECK 12: a NEGATIVE CONTROL on the new block's mixed-resolution read -------------------
    # The natural error is to read the CM axis at PQ resolution -- i.e. to use the PQ-bin-level table
    # where the CM-cell-level one is needed, which is the same thing as evaluating the CM cumulative
    # indicator at the wrong threshold. Emulate it by reading the CM axis one grid level off; that
    # must disagree, or the check above is not actually testing the CM axis at all.
    if Lcm >= 2
        HRC_wrong = zeros(nrow, ncm)
        Xs = tabs_x.X; Ys = tabs_x.Y; Hs = tabs_x.Hcm
        for l in 1:Lcm
            lw = min(l + 1, Lcm)      # off-by-one on the CM threshold axis
            for oi in 1:nO
                o = origins[oi]
                rcm = Hs[o, lw] - Hs[ref, lw]
                for a in 1:nc
                    HRC_wrong[cmpq_level_row(a), (l-1)*nO+oi] =
                        ((Ys[lw, a, o] - Ys[lw, a, ref]) - state.mu[a] * rcm) / W
                end
            end
        end
        dev = maximum(abs, HRC_wrong[1:nc, 1:nO*Lcm] .- HRC_ref[1:nc, 1:nO*Lcm]) /
              max(1e-300, maximum(abs, HRC_ref[1:nc, 1:nO*Lcm]))
        check("12  NEGATIVE CONTROL fires: reading the CM axis one level off disagrees",
              dev > 1e-3, @sprintf("max rel %.3e (want >> 0)", dev))
    end
    return nothing
end

println("\nCM + PAIRWISE-QUANTILE (family #7): D=4 dense oracle")
run_case(D = 4, W = 4000, L = 5, G = 10, n_families = 1, seed = 20260812, muHat = 0.2,
         sigmaHat = 2.5, ref = 1)
run_case(D = 4, W = 4000, L = 5, G = 50, n_families = 2, seed = 20260813, muHat = 0.2,
         sigmaHat = 2.5, ref = 3)
run_case(D = 4, W = 6000, L = 10, G = 50, n_families = 2, seed = 20260814, muHat = 0.15,
         sigmaHat = 3.0, ref = 2)
run_case(D = 5, W = 6000, L = 2, G = 50, n_families = 1, seed = 20260815, muHat = 0.2,
         sigmaHat = 2.5, ref = 5)

println("\n", "="^96)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^96)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_d4_dense_oracle: $(NFAIL[]) check(s) failed")
