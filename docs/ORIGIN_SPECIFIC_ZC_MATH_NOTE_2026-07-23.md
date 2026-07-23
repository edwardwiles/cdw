# Origin-specific pairwise-zero-covariance restriction: math note (2026-07-23)

Branch: `experiment/fullA-origin-specific-zc-K12-2026-07-23`, forked from
`production/fullA-exact@d886d1d` (one commit past tag
`cm-meanzc-production-ready-2026-07-23`, `a00dc2d`).

Companion prototype (NOT reused as code, referenced only for provenance):
`archive/fullA-cm-mean-zc-prototype-2026-07-22` (worktree
`gravity-experiment-fullA-cm-pairwise-zero-cov`).

## 1. What this restriction is

For k = 1,...,K_mean, introduce **origin-specific** positive outer parameters

    nu_{o,k} = E_F[z_o(omega)^k],   o = 1,...,D

parameterized as eta_{o,k} = log(nu_{o,k}). The complete outer vector is

    [ g_p ; z_free ; eta_{1,1},...,eta_{D,1}, eta_{1,2},...,eta_{D,K_mean} ]

(level-major, origin-minor ordering — matches the task brief's own listing).

Mean-defining moments (one per origin, per level, k <= K_mean):

    E_F[ z_o^k - nu_{o,k} ] = 0      for every o, k <= K_mean

Pairwise zero-covariance moments (unordered pairs o < p, k <= K_pair <= K_mean):

    E_F[ z_o^k z_p^k - nu_{o,k} nu_{p,k} ] = 0      for every o < p, k <= K_pair

This is accurately described as **pairwise zero covariance of z_o^k and
z_p^k** — never "independence," and never "zero correlation" without an
accompanying positive-variance check (see `recovered_covariance_matrix`,
reused unchanged from `cm_meanzc_moments.jl`).

**No common-marginals restriction is imposed anywhere in this arm.** There is
no CM CDF-contrast block, no CM tail-moment block, no shared nu_k. The column
layout is

    [ economic (ncore_econ-1) | mean_1(D) ... mean_{K_mean}(D)
      | pair_1(npair) ... pair_{K_pair}(npair) | gravity ]

— identical in spirit to `cm_meanzc_moments.jl`'s layout, minus the entire
CM-grid block.

## 2. Why the mean-only arm is definitional

Without common marginals, nu_{o,k} is origin-specific and otherwise
unrestricted (positivity only). The equation `E_F[z_o^k] = nu_{o,k}` alone
does not restrict F at all: for ANY F, set nu_{o,k} to F's own o-th-origin
k-th moment. Hence

    Delta^unrestricted  ==  min_{nu_{o,k}} Delta^*_{origin moments}

up to numerical tolerance — the profiled feasible set is identical to the
unrestricted feasible set. This is used as an **implementation-equivalence
test**, not a target for a long outer campaign (task brief Section 3). The
genuine new economic restriction is the pairwise product condition
(Section 1's second equation).

Three diagnostic configurations are preserved for this reason:
`:unrestricted`, `:origin_specific_moments`, `:origin_specific_moments_zero_covariance`.

## 3. Target-layout abstraction

The existing `cm_meanzc_moments.jl` machinery hard-codes ONE nu_k shared
across all D origins (valid under common marginals, where every origin
shares a marginal and hence every raw moment). Making nu origin-specific
requires an explicit **target layout** describing how outer eta coordinates
map onto (origin, power) pairs:

  - `SharedByPowerLayout(K_mean, K_pair)` — n_eta = K_mean; every origin o at
    level k reads the SAME target index k. This is the existing CM+meanzc
    production behavior, reproduced bit-for-bit.
  - `OriginByPowerLayout(D, K_mean, K_pair)` — n_eta = K_mean*D; origin o at
    level k reads target index `(k-1)*D + o`.

Both are immutable `struct`s (see `cm_originzc_target_layout.jl`); no
mutable `Ref`, no closed-over shared state — every evaluation receives the
complete eta/nu vector as a plain argument, exactly matching
`cm_meanzc_moments.jl`'s existing "eta rides through theta_ext" discipline
(file header, `cm_meanzc_moments.jl:45-50`).

The `:anchored` moment basis (`mean_columns_anchored`) is a reparameterization
that only makes sense when every origin shares one target: subtracting a
common reference origin's contrast collapses D equality constraints against
one shared nu_k into one anchored constraint plus D-1 nu-free contrasts. With
origin-specific targets there is no such collapse (each origin's constraint
genuinely involves its own nu_{o,k}), so `OriginByPowerLayout` supports
`meanzc_basis = :direct` ONLY; `:anchored` is a hard error for this layout.

## 4. Analytic derivative of Delta_dual w.r.t. eta_{o,k}

Using the current production sign convention (`cm_meanzc_moments.jl:323-364`,
unchanged): Delta_dual = -(mean(Psi(q*)) + zeta*), q_s = -zeta* -
lambda*'G_s(theta). The envelope theorem gives, for any outer parameter
theta_j entering only through the moment columns G,

    d Delta_dual / d theta_j = -mean_m * sum_s lambda*_s * (d G_s / d theta_j)_s

(`mean_m` = `verify.m_mean`, the recovered-weight mean, matching the existing
`d_delta_dual_d_nu_vec` convention exactly — no extra sign flip introduced
here).

**Under `OriginByPowerLayout`**, nu_{o,k} enters exactly two families of
moment columns:

  1. Its own mean-defining column: `g_mean,o,k(s) = z_o(s)^k - nu_{o,k}`, so
     `d g_mean,o,k / d nu_{o,k} = -1`. (Unlike the shared case, this is the
     ONLY mean column nu_{o,k} touches — each origin has its own mean
     moment now, not a whole D-wide block sharing one target.)
  2. Every pair column involving origin o at level k <= K_pair:
     `g_pair,op,k(s) = z_o(s)^k z_p(s)^k - nu_{o,k} nu_{p,k}` for every
     p != o, so `d g_pair,op,k / d nu_{o,k} = -nu_{p,k}` (nu_{p,k} held
     fixed — a genuinely different factor per partner p, unlike the shared
     case's single `-2*nu_k`).

Therefore

    d Delta_dual / d nu_{o,k}
        = mean_m * ( lambda_mean,o,k* * (-1)
                      + sum_{p != o, k<=K_pair} lambda_pair,op,k* * (-nu_{p,k}) )
        = -mean_m * ( lambda_mean,o,k*  +  sum_{p != o} nu_{p,k} * lambda_pair,op,k* )

    d Delta_dual / d eta_{o,k} = nu_{o,k} * d Delta_dual / d nu_{o,k}      (chain rule, nu = exp(eta))

This combines "the multiplier on origin o's k-th mean-defining moment" and
"all pair-moment multipliers involving origin o at power k," weighted by the
partner's own nu_{p,k} — exactly as required (task brief Section 7).

**Consistency check against the existing shared-target formula** (not
assumed — verified by direct algebraic reduction, and independently by
finite differences at D=4, Section 11): setting nu_{o,k} = nu_{p,k} = nu_k
for all o,p and summing the per-origin derivative above over all D origins
reproduces `d_delta_dual_d_nu_vec`'s existing formula exactly:

    sum_o [ -mean_m*(lambda_mean,o,k* + nu_k * sum_{p!=o} lambda_pair,op,k*) ]
      = -mean_m*( sum_o lambda_mean,o,k*  +  nu_k * 2 * sum_{o<p} lambda_pair,op,k* )

using the double-counting identity `sum_o sum_{p!=o} lambda_pair,{o,p},k* =
2 * sum_{o<p} lambda_pair,op,k*` (each unordered pair's multiplier is visited
once from each of its two origins). This is bit-for-bit
`d_delta_dual_d_nu_vec`'s own `total = dot(lambda_mean_k, d_mean) +
dot(lambda_pair_k, d_pair_dnu(nu_k, npair))` with `d_mean = -1`,
`d_pair_dnu = -2*nu_k`. The new origin-specific formula is therefore a true
generalization, not an independent re-derivation.

## 5. Dimension counts (task brief Section 6, D=20)

| K | outer eta params | mean moments | pair moments | total new inner moments |
|---|---|---|---|---|
| 1 | 20 | 20 | 190 | 210 |
| 2 | 40 | 40 | 380 | 420 |

`npair = D(D-1)/2 = 190` at D=20, independent of K.

No L~50 common-marginals grid block exists in this arm at all — the inner
problem may therefore be smaller/faster than CM+ZC despite the larger outer
loop. Measured, not assumed (Section 9 of the release report).
