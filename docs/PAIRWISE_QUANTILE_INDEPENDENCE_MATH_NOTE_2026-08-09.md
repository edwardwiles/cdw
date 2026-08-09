# Pairwise-independence restriction via country-specific quantile cutoffs — equivalence proofs

Draft eq. (32), prototype branch `prototype/pairwise-quantile-independence-2026-08-09`. This note
proves the claims the task brief asks to "prove/document": that the enforced marginal-bin and
joint-cell moments are exactly equivalent to a full grid of quantile-crossing probability
conditions, and that the new `:all_cross` default (all `r,s=1..4` combinations) is strictly
stronger than the draft's own diagonal-only (`r=s`) condition.

## 1. Setup and notation

Fix an origin `o` with productivity draws `z_o`, and ordered cutoffs `q_{o,1}<q_{o,2}<q_{o,3}<q_{o,4}`
(with the conventions `q_{o,0}:=-∞`, `q_{o,5}:=+∞`). The bin assignment is

    b_o(z) = a   iff   q_{o,a-1} <= z < q_{o,a},   a = 1,...,5.

Write `π_{o,a} := P(b_o(z_o)=a)` for the marginal bin probability, and for a second origin `p≠o`,
`π_{op,ab} := P(b_o(z_o)=a, b_p(z_p)=b)` for the joint cell probability. The target equal-probability
grid is `p_r = r/5` for `r=1,...,4` (the task's `p=(0.2,0.4,0.6,0.8)`), with the convention `p_0:=0`,
`p_5:=1`.

**Enforced moments** (task Section 2-3): for `a=1,...,4`,

    g^M_{o,a}(z;q) = 1{b_o(z)=a} - 1/5,

and for every unordered pair `o<p` and `a,b=1,...,4`,

    g^P_{op,ab}(z;q) = 1{b_o(z)=a, b_p(z)=b} - 1/25.

Both are "active" (their sample mean is set to zero by the inner dual solve); the 5th bin/cell is
never separately constrained.

## 2. Marginal equivalence: `g^M` ⟺ `P(z_o<q_{o,r})=p_r` for every `r=1,...,4`

**Claim.** `π_{o,a}=1/5` for every `a=1,...,4` (i.e. all four `g^M_{o,a}` moments hold) **if and
only if** `P(z_o<q_{o,r})=p_r=r/5` for every `r=1,...,4`.

**Proof.** By construction of the bins, the event `{z_o<q_{o,r}}` is exactly the disjoint union of
bins `1,...,r` (since `q_{o,0}=-∞` and bin `a` covers `[q_{o,a-1},q_{o,a})`), so

    P(z_o < q_{o,r}) = Σ_{a=1}^{r} π_{o,a}          (★)

for every `r=1,...,4` — a finite telescoping (partial-sum) identity. This is exactly the same
lower-triangular all-ones cumulative map already implemented and proved in this codebase for a
different pairing (interval-vs-CDF against a *fixed reference origin*):
`common_marginals_interval.jl:139-145`, `cumulative_interval_transform_matrix(L)`. Here `(★)` is
the identical construction with `L=4` and no reference-origin subtraction. Since the map `S` with
`S[a,r]=1{a<=r}` is unit lower-triangular, it is invertible, so `(★)` is a linear bijection between
`(π_{o,1},...,π_{o,4})` and `(P(z_o<q_{o,1}),...,P(z_o<q_{o,4}))`:

- *(⟹)* If `π_{o,a}=1/5` for `a=1..4`, `(★)` gives `P(z_o<q_{o,r})=r/5=p_r`.
- *(⟸)* If `P(z_o<q_{o,r})=p_r` for `r=1..4`, then `π_{o,a} = P(z_o<q_{o,a}) - P(z_o<q_{o,a-1}) =
  p_a - p_{a-1} = 1/5` (using `p_0:=0`), for every `a=1..4`.

The 5th bin is never separately constrained because `π_{o,5} = 1 - Σ_{a=1}^4 π_{o,a}` is pinned to
`1/5` automatically by normalization once the other four are — this is precisely why the task
instructs dropping it as redundant, and the proof above shows the redundancy is exact, not
approximate. ∎

## 3. Joint equivalence: `g^P` ⟺ `P(z_o<q_{o,r}, z_p<q_{p,s})=p_r p_s` for **every** `r,s=1,...,4`

This is the genuinely new part (the draft's own eq. (32) only imposes the diagonal `r=s`
sub-case). Define `F(r,s) := P(z_o<q_{o,r}, z_p<q_{p,s})`, for `r,s=0,...,4` (with `F(0,·)=F(·,0)=0`
by the same `q_{o,0}=-∞` convention as above).

**Claim.** `π_{op,ab}=1/25` for every `a,b=1,...,4` (i.e. all sixteen `g^P_{op,ab}` moments hold)
**if and only if** `F(r,s)=p_r p_s` for **every** `r,s=1,...,4` — the full 4×4 grid, not merely the
diagonal.

**Proof.** The event `{z_o<q_{o,r}, z_p<q_{p,s}}` is the disjoint union of joint cells `(a,b)` with
`a<=r, b<=s`, so

    F(r,s) = Σ_{a=1}^{r} Σ_{b=1}^{s} π_{op,ab}          (★★)

for every `r,s=1,...,4` — the 2-index generalization of `(★)`, i.e. the Kronecker-square `S⊗S` of
the same invertible lower-triangular map (the direct 2-D analog of
`common_marginals_interval.jl`'s `full_transform_matrix(nO,L) = kron(S,I(nO))` construction, here
applied on both index axes instead of mixed with an identity on the origin axis). `S⊗S` is
invertible (Kronecker product of invertible matrices), so `(★★)` is again a bijection:

- *(⟹)* If `π_{op,ab}=1/25` for all `a,b=1..4`, `(★★)` gives `F(r,s)=Σ_{a<=r}Σ_{b<=s}(1/25) =
  rs/25 = (r/5)(s/5) = p_r p_s`.
- *(⟸)* If `F(r,s)=p_r p_s` for **all** `r,s=1,...,4` (using `F(0,s)=F(r,0)=0`, `p_0=0` at the
  boundary), the discrete mixed second difference of `(★★)` recovers each cell directly:

      π_{op,ab} = F(a,b) - F(a-1,b) - F(a,b-1) + F(a-1,b-1)
                = p_a p_b - p_{a-1} p_b - p_a p_{b-1} + p_{a-1} p_{b-1}
                = (p_a - p_{a-1})(p_b - p_{b-1})
                = (1/5)(1/5) = 1/25,

  for every `a,b=1,...,4`. ∎

**Why the draft's diagonal-only condition is strictly weaker.** The draft imposes only `F(r,r) =
p_r^2` for `r=1,...,4` — 4 scalar conditions per pair, versus the 16 independent conditions
`F(r,s)=p_r p_s` (`r,s=1,...,4`) that `(★★)`'s bijection shows are exactly equivalent to the full
`g^P` moment set. The 4 diagonal conditions alone do not determine the off-diagonal joint cells
`π_{op,ab}` (`a≠b`): e.g. a joint table with `π_{op,aa}=1/25` on the diagonal but arbitrary
(non-independent) off-diagonal mass redistributed between cells `(a,b)` and `(b,a)`, `a≠b`, in a
way that preserves both the row/column marginals (Section 2) *and* every diagonal rectangle
`F(r,r)`, is consistent with the draft's condition but is not the independent joint distribution.
Enforcing the full grid (`:all_cross`, all `r,s=1,...,4`) is therefore what the task calls "the
new default" precisely because it is the condition that is actually equivalent to pairwise
independence of the two origins' quantile positions at this 5-bin resolution, not merely
equal-quantile-crossing along the diagonal. `:draft_cumulative_diagonal` is retained only as a
replication mode for the weaker draft condition, never as the scientific default.

## 4. Verifier residuals: cumulative factorization residual from the enforced cells, no second dense pass

The verifier (Section 8 of the implementation plan) independently recomputes the full `5×5` joint
cell probability table `π̂_{op,ab}` (`a,b=1,...,5`, including the never-directly-enforced 5th
row/column) directly from the already-built `bin` array — this is required output regardless (task:
"report all five marginal-bin probabilities and all 5×5 joint-cell probabilities, even though only
the nonredundant subset is enforced"). Given that table, the cumulative factorization residual at
every `r,s=1,...,4` is obtained by the *same* partial-sum map `(★★)` used in the proof above:

    F̂(r,s) = Σ_{a=1}^{r} Σ_{b=1}^{s} π̂_{op,ab},        residual(r,s) = |F̂(r,s) - p_r p_s|,

an `O(16)` computation per pair once `π̂_{op,·,·}` is in hand — no second dense recompute, no
re-touching the draws. Section 2's marginal residuals are the 1-D special case of the same
identity. This is the exact mechanism `verify_inner_solution_operator_pairwisequantile!` uses to
report cumulative residuals alongside the raw interval-cell KKT residuals.
