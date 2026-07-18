# Incremental hard-winner algorithms: winner-margin certificates, coordinate updates, and structure

Branch `diag/fullA-d4-exact` (worktree checkout at `86caa75`). All work is **additive** in new files
under `full_aod_diag/d4_exact/`; no production code, the dense constructor, the tie-breaking rule
(`misc/smoothMinIndNew!.jl::MinInd!`), `winners.jl`, `winners_v2.jl`, `lfix_incremental.jl`,
`compressed_moments.jl`, or the exact hard estimand were modified. The **trusted reference** throughout
is `winners_v2.jl::compute_winners_fast` (== `winners.jl::compute_winners`, first-occurrence argmin,
bit-for-bit). Exact equivalence is required and verified at every tested point.

**New files**
- `winner_certificate.jl` — the certificate (Section 1), coordinate update (Section 2), draw-threaded
  variant (Section 6), and the `WinnerRefCache`.
- `test_winner_certificate.jl` — strict exact-equivalence suite (A–E), **ALL PASS**.
- `bench_winner_certificate.jl` — Section 1+2 measurements (+ CSV `results/winner_certificate/`).
- `explore_winner_structure.jl` — Sections 3–5 exploratory measurements (D=4/6/8/10).
- `bench_threading.jl` — Section 6 threading interaction.

Run environment: `demand.mit.edu`, `source .knitro_env.sh`, `JULIA_NUM_THREADS=20`, D=4/W=8000 unless
noted. Baseline equivalence suite (`test_oracle.jl`, `test_lfix_incremental.jl`,
`test_compressed_moments.jl`, `test_compressed_cc_inner.jl`) re-run first — all pass, worktree not regressed.

---

## 0. Verified score convention and signs (read from code, not assumed)

`hFunction!`/`MinInd!` pick the winner as the **argmin of price**:

```
winner_{s,d} = argmin_o  price_{s,o,d},   price_{s,o,d} = constCons_{o,d} / U_{s,o}^{-mu} = constCons_{o,d} * U_{s,o}^{mu}
constCons_{o,d} = wHat_o * AodPow_{o,d} * tau_{o,d},   AodPow_{o,d} = (Aod_{o,d}/cHat_{o,d})^{-mu}
Aod_{o,d} = Aod_theta_{o,d} * (FIXED data),   Aod_theta = exp(z),  z = the reduced log-A coordinate.
```

Only `Aod_theta` moves during the optimization. Writing the **destination-d score** `S_{sod} := log price_{s,o,d}`:

```
S_{sod} = B_{sod} + a_{od},
a_{od}  = -mu * log Aod_theta_{o,d} = -mu * z_{od}      (the ONLY optimization-varying part)
B_{sod} = log wHat_o + log tau_{o,d} + mu*log U_{s,o} + (fixed A-independent (o,d) terms).
```

The brief writes `B_{so}`; `B` genuinely also carries a `d`-dependence (through `tau_{o,d}` and the
`lambda` ratio), **but the certificate is invariant to it** — every fixed term cancels in the winner
gap. Because `a_{od}` does not depend on the draw `s`, moving `theta -> theta'` shifts **every** draw's
score for cell `(o,d)` by the **same** amount

```
delta_{od} := S'_{sod} - S_{sod} = a'_{od} - a_{od} = log constCons'_{o,d} - log constCons_{o,d} = -mu*(z'_{od} - z_{od}).
```

That single `D x D` shift matrix (O(D²), no draw loop) is what the certificate screens against. The code
verifies `delta == -mu*(z'-z)` to machine precision on every call (`shift_matrix`), and the winner is the
**argmin** of `S`, so the certificate uses the **min-form threshold** (making a competitor *cheaper*, i.e.
lower `delta`, is what threatens the incumbent minimum) — the sign-mirror of the brief's argmax schematic.

**Note vs the brief's minimal formula.** The brief proposed `m_{sd} > max_{k≠w}[(a'_{kd}-a_{kd})-(a'_{wd}-a_{wd})]`,
which is correct for an **argmax** winner. The code's winner is **argmin**, so the exact-sufficient
condition implemented here is the sign-mirror:

```
winner w unchanged if   margin_{sd} > delta_{wd} - min_{k≠w} delta_{kd}.
```

Both are the same statement under the score-sign flip; the implemented one is verified exact against the
full scan (below). A conservative one-number-per-destination `infinity-norm` variant
(`m_{sd} > max_o delta_{od} - min_o delta_{od}`) is strictly implied and would also be exact-sufficient.

---

## 1. Winner-margin certificate — **ADOPT**

`WinnerRefCache` caches, per `(draw s, destination d)`: winner, winner score, runner-up + score, third +
score (top-3), and the margin `sr - sw`. Memory is **O(W·D)** (the top-3 + `mulU_{s,o}=mu·log U_{s,o}`),
**not** the O(W·D²) dense price array — this is itself a large allocation win. `certified_winner_update`
screens each cell with the exact certificate, reuses the cached winner where certified, and **exactly
rescans only uncertified cells** (O(D) each). A `tol_far` guard falls back to a full scan automatically
when the callback point is too far from the reference. Cache use never changes the mathematical value or
introduces order dependence (winner-matrix writes are disjoint per cell; the certificate is a pure
predicate).

**Exactness.** `test_winner_certificate.jl` (A–E): the certificate winner matrix is **bit-identical**
(`maxabsdiff = 0`) to `compute_winners_fast` at every tested point, across accepted / line-search /
continuation / far step sizes, including the `tol_far` full-scan fallback path. Proof of correctness: the
certificate inequality is **strict**, so a certified cell's cached winner is the unique strict argmin at
`theta'` — it cannot disagree with the full scan.

**Measurements** (D=4, W=8000, winner-finding component; full scan = `compute_winners_fast` = 1.785 ms,
1751 KB/call; `build_winner_ref` one-time = 2.43 ms, 2253 KB):

| regime | step | certified | rescan | cert time | full time | **speedup** | cert allocs |
|---|---|---|---|---|---|---|---|
| accepted | 1e-3 | **99.9%** | 0.1% | 0.103 ms | 1.785 ms | **17.4×** | 254 KB |
| accepted | 5e-3 | 99.7% | 0.3% | 0.096 ms | 1.785 ms | 18.6× | 254 KB |
| line-search | 2e-2 | 98.8% | 1.2% | 0.104 ms | 1.785 ms | 17.2× | 254 KB |
| line-search | 5e-2 | 97.2% | 2.8% | 0.119 ms | 1.785 ms | 15.0× | 254 KB |
| continuation | 1e-1 | 95.0% | 5.0% | 0.16–0.23 ms | 1.785 ms | 8–13× | 254 KB |
| continuation | 2e-1 | 89.2% | 10.8% | 0.142 ms | 1.785 ms | 12.6× | 254 KB |
| far | 5e-1 | 77.0% | 23.0% | 0.226 ms | 1.785 ms | 7.9× | 254 KB |
| far | 1.0 | 65.6% | 34.4% | 0.199 ms | 1.785 ms | 9.0× | 254 KB |

- **Certified fraction vs step size**: monotone, ~100% at accepted-step magnitude, ~97–99% at
  line-search magnitude, ~89–95% at continuation magnitude, still ~66–77% at large "rejected/far" steps.
- **Time saved**: winner-finding drops from 1.785 ms to 0.10–0.23 ms → **8–18×** on the winner component,
  plus a **~7× allocation drop** (254 KB vs 1751 KB — the O(W·D²) price array is never built).
- **Point classes**: accepted points certify ≈100% (near-free), rejected/large line-search points still
  certify 66–99% depending on distance, profile-continuation points certify ≈89–95%. All exact.

**Verdict: ADOPT.** It is exact, cheap, and its worst case (far points) still beats a full scan and
auto-falls-back. The reference build cost (2.4 ms) is amortized over the many nearby evaluations of a
line search / FD sweep / continuation segment.

---

## 2. Coordinate-update specialization — **ADOPT (removes the last O(D) fallback)**

**Audit of the existing production path.** `composite_gradient.jl::a_block_fd_component` already uses the
O(1)-per-draw rule via `lfix_incremental.jl::dest_contrib_incremental_o1` (tier `:incremental_o1`):
- if the changed origin was **not** the winner → compare its new score with the cached winner;
- if it **was** the winner → compare with the cached runner-up.

This is `update_winner_o1`, and it is used for every A-block coordinate. **The gap**: for the
gravity-pivot reparameterization, each reduced coordinate perturbs **exactly 2 A-cells** (the direct cell
+ the pivot cell). The audit (measured, `bench_winner_certificate.jl`) shows the pivot is cell `(4,4)`
(destination 4), so:

- **coords 2–13 (12 of 15)**: the 2 changed cells fall in **two different destinations**, each with one
  changed origin → pure O(1) update, **no fallback**.
- **coords 14–16 (3 of 15)**: the direct cell is *also* in destination 4 → **two changed origins in the
  same destination** → the production code falls back to a generic **O(D) rescan**
  (`dest_contrib_incremental`, see `composite_gradient.jl:97`), because chaining two O(1) updates can
  propagate an inexact cached runner-up.

`coord_winner_update!` closes this gap with the **top-3 cache**: with ≤2 changed origins, the best
surviving *unchanged* origin is at worst rank 3, so the exact new winner = argmin over {changed origins'
new scores} ∪ {first of rank-1/2/3 not changed}. This is **exact for the 2-changed-same-destination case
with no O(D) rescan**, eliminating the remaining fallback. (`test_winner_certificate.jl` (C):
bit-identical to the full scan across all coordinates × steps, including coords 14–16.)

**Measurements** (D=4, W=8000):

| coordinate | A-cells | time | allocs | vs full |
|---|---|---|---|---|
| `gamma'` (coord 1) | 0 | 0.009 ms | 1.3 KB | ~200× (winners provably unchanged) |
| `z_free` (2 cells, ≤2 dests) | 2 | 0.131 ms | 1.8 KB | **13.6×** |

The `gamma'` coordinate touches **no** A-cell (it only moves the counterfactual column), so the winner
matrix is provably unchanged — the update is essentially free.

**Verdict: ADOPT.** Exact, and it removes the last generic O(D) winner fallback in the coordinate path.
The benefit is bounded at D=4 (3 of 15 coordinates, each an O(D=4) rescan over W draws per FD leg) but
**grows with D** — at larger D more coordinates share the pivot's destination and the rescan is O(D).

---

## 3. Pairwise threshold preprocessing — **REJECT**

For each origin pair `(o,k)` precompute+sort `R_s^{ok} = mulU_{so}-mulU_{sk}` (the draw-fixed part of the
`o`-vs-`k` breakpoint). Measured (D=4, W=8000, warmed): preprocessing **2.07 ms** for the 12 sorted
pair-arrays and **1.02 MB persistent memory** (O(D²·W)). By comparison the margin certificate needs only
the **O(D²)** shift matrix per step (~0 persistent memory) and already certifies **97–100%** of draws in
**~0.10 ms/step**. Maintaining per-pair sorted outcomes and updating winners only for the few draws whose
breakpoint is crossed cannot beat a screen that already eliminates nearly everything. Per the brief's
explicit rule ("do not adopt unless it improves measured end-to-end performance"), **REJECT**.

---

## 4. Draw-dominance partial order — **REJECT**

`g_s^{(o)} = (mulU_{so}-mulU_{sk})_{k≠o}`; if `g_{s2}^{(o)} ≤ g_{s1}^{(o)} componentwise` then `o` winning
`s1` implies `o` wins `s2` for **every** A (the RHS is A/d-dependent but s-independent — exact). Measured
(candidate origin o=1, 200k sampled draw pairs):

| D | comparable fraction | longest chain (proxy) |
|---|---|---|
| 4 | 0.497 | 2 |
| 6 | 0.331 | 1 |
| 8 | 0.249 | 1 |
| 10 | 0.201 | 1 |

The pairwise comparable fraction is non-trivial (inflated by the shared `mulU_{so}` term), **but the
useful structure — chains that let winner evaluations be skipped — is absent**: the longest chain proxy is
1–2 at every D, and comparability *falls* with D. There is no cheap transitive structure to exploit; the
O(W²) machinery needed to use it would cost far more than the O(W·D) scan it aims to shortcut. **REJECT**
(exploratory, measured as instructed).

---

## 5. Origin pruning — **REJECT (empty)**

Exact test: origin `o` is never a winner at destination `d` if some fixed `k` dominates it for all draws,
`logCC_{kd}-logCC_{od} < min_s(mulU_{so}-mulU_{sk})` — automatically invalidated when A moves (LHS is
A-dependent). Also tested over a local trust region (log-A free to move ±0.1, adversarial worst case).
Measured at a feasible point:

| D | prunable (o,d) at point | prunable over region (r=0.1) | total (o,d) |
|---|---|---|---|
| 4 | 0 (0.0%) | 0 (0.0%) | 16 |
| 6 | 0 (0.0%) | 0 (0.0%) | 36 |
| 8 | 0 (0.0%) | 0 (0.0%) | 64 |
| 10 | 0 (0.0%) | 0 (0.0%) | 100 |

Every origin is the cheapest for at least some draws (the draw dispersion in `U` is wide), so **no** origin
can be excluded at any destination — the pruning set is empty at every D. **REJECT** (exact but empty).

---

## 6. Threading interaction

Added `certified_winner_update_threaded` (draw-level `Threads.@threads`, thread-local stat buffers,
**deterministic** fixed-order reduction). Winner writes are disjoint per cell → bit-identical to the
single-threaded path (confirmed "exact? yes" at every size).

**Context A — ordinary exact value eval (draw-level threading of the certificate):**

| D | W | single | threaded | speedup | exact |
|---|---|---|---|---|---|
| 4 | 8000 | 0.206 ms | 0.136 ms | 1.52× | yes |
| 4 | 80000 | 1.046 ms | 1.215 ms | 0.86× | yes |
| 10 | 8000 | 0.329 ms | 0.395 ms | 0.83× | yes |
| 10 | 80000 | 6.45 ms | 3.47 ms | **1.86×** | yes |

Draw-level threading only pays off at **large W·D** (1.86× at D=10/W=80000); for the sub-millisecond
production D=4/W=8000 certificate the thread-spawn overhead makes it a wash-to-slight-loss. **Recommendation:
keep the certificate single-threaded at the production scale; enable draw threading only at large W·D.**

**Context B — coordinate-parallel L_fix sweep (D² FD probes):**

| discipline | time (16 coords) | exact |
|---|---|---|
| (a) outer-threaded + inner-serial (**correct**) | 0.285 ms | yes |
| (b) outer-serial + inner-threaded (nested) | 2.092 ms | yes |

Parallelizing the **outer coordinate loop** with a **single-threaded inner winner update** is **7.33×
faster** than draw-threading the inner update, confirming the brief's discipline: inner winner updates
should remain single-threaded inside a coordinate-parallel gradient to avoid oversubscription. Both are
exact.

---

## Ranked comparison (winner-computation component, D=4 / W=8000)

| rank | scheme | winner time | speedup vs full | allocs | preprocessing | exact | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **coordinate-specialized update** (Sec 2) | 0.009–0.131 ms | **13.6–200×** | 1.3–1.8 KB | O(W·D) ref (2.4 ms, amortized) | yes | **ADOPT** (FD gradient / coord descent) |
| 2 | **winner-margin certificate** (Sec 1) | 0.10–0.23 ms | **8–18×** | 254 KB | O(W·D) ref (2.4 ms, amortized) | yes | **ADOPT** (line search / continuation / any nearby point) |
| 3 | trusted full scan (`compute_winners_fast`) | 1.785 ms | 1× | 1751 KB | none | yes (reference) | keep as fallback / reference |
| 4 | pairwise breakpoint updates (Sec 3) | — | — | +1.02 MB persistent | 2.07 ms + per-pair sorts | (exact) | **REJECT** (dominated by the certificate) |
| 5 | draw-dominance pruning (Sec 4) | — | — | O(W·D) | O(W) | (exact) | **REJECT** (no usable chains) |
| 5 | origin pruning (Sec 5) | — | — | O(D²) | O(W·D²) | (exact) | **REJECT** (empty at every D) |

**Bottom line.** The two specialized schemes (coordinate update, winner-margin certificate) are exact,
cheap, and directly attack the winner-finding component of repeated nearby evaluations (FD gradients, line
searches, continuation): **8–200×** on the winner component with a **~7× allocation drop**, and the
coordinate path's top-3 cache removes the last generic O(D) fallback. The three structural ideas
(pairwise breakpoints, draw-dominance, origin pruning) are **rejected on measured grounds** — the margin
certificate already screens 97–100% at near-zero cost, the draw-dominance partial order yields no usable
chains, and no origin is ever prunable. The trusted full scan is preserved as the exact fallback and
equivalence reference throughout.

### Wiring recommendation (not yet done)
The certificate and coordinate update are standalone + equivalence-tested (same posture as
`compressed_moments.jl` / the pow-cache before wiring). The natural production path — behind an opt-in
flag, dense/full-scan default, `TiedWinnerError` → full-scan fallback — is:
1. build a `WinnerRefCache` at each accepted outer point;
2. use `coord_winner_update!` inside `a_block_fd_component` for the 3 same-destination coordinates that
   currently hit the O(D) fallback (exact, removes the fallback), and `certified_winner_update` for
   line-search / continuation winner recomputes;
3. keep both single-threaded inside the coordinate-parallel gradient (Context B), enabling draw threading
   only for standalone value evaluations at large W·D (Context A).
