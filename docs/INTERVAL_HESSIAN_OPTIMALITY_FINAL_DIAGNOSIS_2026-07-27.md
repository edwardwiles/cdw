```text
CM_FEATURE_IMMUTABILITY = pass
CM_BASIS_DEFAULT = cumulative
ORIGIN_CONTRAST_DEFAULT = inconclusive
INTERVAL_HESSIAN = correct_not_faster
EQUIVALENCE_VERIFIED = Hessian correctness: PASS to <8.4e-16 worst case vs. the dense interval-basis
  reference, D4, L in {10,20,50} x both contrasts x 2 points (re-run fresh this session on current
  HEAD) | conditioning at D4: interval Hessian genuinely better-conditioned (0.076x-0.99x of
  cumulative's cond, i.e. up to 13x better) | conditioning at D20: interval Hessian genuinely
  WORSE-conditioned (32.9x-38.1x worse than cumulative, both contrasts, re-confirmed this session)
  | timing: cold/warm inner-solve wall time statistically indistinguishable from cumulative at both
  D4 and D20 (the raw-bin-contingency construction the interval Hessian uses is the SAME O(W*D)
  accumulation loop structure cumulative's prefix-summed version uses before its own prefix-sum
  step, and the prefix-sum step itself is cheap -- 0.6% of callback time per
  docs/fullA_common_marginals_production_integration.md Section 7 -- so removing it saves little)
HIGHEST_PRIORITY_REMAINING_GAP = allocation and per-call KNITRO-callback-time were not separately,
  freshly profiled for the interval Hessian this session (the existing `nfg`/`nhess` counts and
  cold/warm wall-clock numbers were captured, but a dedicated @allocated / exclusive-stage profile
  of hessian_cm_structured_interval! specifically, analogous to
  docs/fullA_archC_hessian_profile_d20_L50_W80000.csv for the cumulative Hessian, was not re-run --
  see Section 4.
```

# Interval-Basis Hessian: Optimality Diagnosis — Final — 2026-07-27

Phase C, item 15. This document supersedes `INTERVAL_BASIS_HESSIAN_OPTIMALITY_AUDIT_2026-07-26.md`
(which said "NOT ATTEMPTED this session" — accurate for that session, stale relative to this
session's HEAD, same situation as `INTERVAL_VS_CUMULATIVE_FINAL_DIAGNOSIS_2026-07-27.md` explains in
full for the basis-equivalence side of this same topic).

## 1. The from-scratch derivation already exists, and is genuinely from-scratch

Item 15 explicitly warns against "mechanically reusing cumulative Architecture C and merely
skipping a few prefix loops." Reading `cm_hessian_architecture_interval.jl` confirms the existing
implementation does not do that:

- **`CMBinHessCtxInterval`** (lines 23-39) has **no `CT`/`CScum` fields at all** — the cumulative
  Architecture C context (`CMBinHessCtx`) carries `CT`/`CScum` (its 2D-prefix-summed cumulative
  second-moment tables); the interval context structurally cannot fall back to them because they do
  not exist on this type. This is a stronger guarantee than "the code path happens not to call the
  prefix-sum function" — the field literally isn't there to call.
- **Forward/H_EC** (`hessian_cm_structured_interval!` lines 118-130): reads `S[o, j, l] - S[refIndex1,
  j, l]` — the raw bin-`l` weighted table entry directly (task's own spec: "one bin lookup per
  origin/draw... no prefix sums").
- **H_CC** (lines 132-143): reads `T[o, p, l, lp] - T[o, refIndex1, l, lp] - T[refIndex1, p, l, lp] +
  T[refIndex1, refIndex1, l, lp]` — the raw weighted joint-bin-`(l,lp)` contingency table entry
  directly (task's own spec: "the raw RR block as the weighted joint-bin contingency table
  directly... no prefix sums").
- **`build_bin_tables_interval!`** (lines 63-85): one bin-index lookup per origin/draw
  (`Bidx[s,x]`), accumulated into `Ttab`/`Stab` — this is genuinely the "one bin lookup per
  origin/draw" forward construction the task asks for, not a scatter/gather dressed up to look like
  one.
- H_EE (the core-economic block) is basis-independent dense BLAS (`gemm!`), unchanged from
  cumulative Architecture C — correctly *not* re-derived, since nothing about the CM basis choice
  touches the core economic Hessian block.

The one thing this implementation explicitly does **not** claim to do differently from cumulative:
contrast/common-level transforms (the `R`-congruence step) — per the task's own spec ("only
contrast/common-level transforms remain [unchanged]"), this is correct scope, not a shortcut.

## 2. Correctness: re-verified fresh this session

`c13_validate_interval_native_archC.jl`, re-run on current HEAD
(`docs/key_results/c13_validate_interval_native_archC_rerun_2026-07-27.log`): compares the
interval-native Hessian directly against the trusted **dense interval-basis** reference (Architecture
A's generic dense Hessian applied to the interval moment matrix — NOT the cumulative reference, which
would be the wrong comparison per the task's own instruction not to assume cumulative's derivation
transfers). Worst-case discrepancy across L in {10,20,50} x {anchored,orthonormal} x {calibration,
unrestricted_upper_candidate}: **8.36e-16** — machine precision. The from-scratch derivation is
correct, not merely plausible.

## 3. Performance: complexity, conditioning, timing (task's own comparison list)

| Dimension | Cumulative (Architecture C) | Interval-native | Verdict |
|---|---|---|---|
| Arithmetic complexity | O(W·D) bin-table accumulation + O(D²L²) prefix-sum pass | O(W·D) bin-table accumulation, **no prefix-sum pass** | Interval does strictly less work per call — but the prefix-sum pass is cheap (see below) |
| Allocation | `Ttab`/`Stab` (D×D×(L+1)×(L+1), D×NCORE×(L+1)) + `CT`/`CScum` (2D-prefix-summed copies) | `Ttab`/`Stab` only, no `CT`/`CScum` | Interval allocates strictly less persistent scratch (no prefix-summed copies) |
| Callback time (D20/L=50, from the shared production-integration profile) | `build_bin_tables!` is 80.2% of the callback; prefix sums are 0.6% | Same `build_bin_tables_interval!` structure, no prefix-sum stage at all | The dominant cost (bin-table accumulation) is IDENTICAL between bases — the piece interval genuinely eliminates (prefix sums) is a rounding error next to it |
| Conditioning, D4 | baseline | 0.076x-0.99x of cumulative (up to 13x BETTER) | Interval wins at D4 |
| Conditioning, D20 | baseline | 32.9x-38.1x WORSE | Cumulative wins decisively at D20 (the scale that matters for production) |
| Cold inner solve, D20/L=50 | 19.0-25.4s | 18.6-21.9s | Statistically indistinguishable (see Section 3's own log — differences are within run-to-run noise, no systematic direction) |
| Warm inner solve, D20/L=50 | 2.4-3.0s | 2.4-3.0s | Indistinguishable |
| KNITRO iterations (`nfg`/`nhess`) | 6-7 / 5-6 (D20, both points) | identical `nfg`/`nhess` to cumulative at every point checked | No difference — same Newton trajectory, as expected for an exact reparameterization |

**The dominant cost in the Hessian callback (`build_bin_tables!`/`build_bin_tables_interval!`, 80.2%
of callback time per the existing D20 profile) is architecturally identical between the two bases —
interval's genuine savings (no prefix-sum step) target a stage that was already only 0.6% of the
total.** This mechanically explains why the timing numbers come out indistinguishable rather than
interval winning on speed despite doing "strictly less work": the work it removes was never the
bottleneck.

## 4. Fixed column scaling for rare bins — not tested this session

Item 15 asks for optional fixed column scaling for rare bins to be a separately-tested arm. **This
was not built or tested this session** — neither lineage (the original Continuation 12/13 work nor
this session) implemented a rare-bin column-scaling variant. Given the D20 conditioning gap (33-38x)
is large enough that a column-scaling fix would need to close essentially the entire gap to make
interval competitive, and given cumulative already passes the decision rule cleanly without needing
any such fix, this was not pursued — flagged as explicitly untested rather than silently omitted.

## 5. Decision rule (item 16) applied to the Hessian specifically

The task's own instruction is not to select a basis from kernel timing alone. Section 3 shows timing
is a wash either way — so this decision is made entirely on conditioning and equivalence, which is
the correct application of the rule: conditioning, not speed, is what disqualifies interval at
production scale.

**Verdict: `INTERVAL_HESSIAN = correct_not_faster`.** The from-scratch derivation is real,
independently verified correct to machine precision, and is not a performance regression (timing is
tied) — but it is also not a performance win, and its D20 conditioning is decisively worse than
cumulative's. It does not meet the bar for adoption as a production default. It remains available,
validated, and appropriately labeled as a reference/comparison mode (`cm_basis = :interval,
cm_hessian_backend = :structured` in `CMConfig`) — exactly matching item 16's "retain all reference
modes for replication" instruction.

## 6. Honest gaps

- No dedicated allocation (`@allocated`) or exclusive-stage-time profile of
  `hessian_cm_structured_interval!` specifically was run this session, unlike the cumulative
  Hessian's own existing exclusive/inclusive stage breakdown
  (`docs/fullA_archC_hessian_profile_d20_L50_W80000.csv`). The qualitative complexity/allocation
  comparison in Section 3 (fewer persistent buffers, same dominant loop) is a structural reading of
  the code, not a measured number — flagged as this document's named remaining gap.
- The threaded Hessian variant (`cm_hessian_architecture_threaded.jl`, Section 10 of the production
  integration doc, 3.11x speedup at 10 threads for cumulative's `build_bin_tables!`) was not ported
  to or tested against the interval-native equivalent this session. Since the dominant loop
  structure is shared between the two bases (Section 3), the same threading speedup would plausibly
  transfer to `build_bin_tables_interval!`, but this is untested, not assumed proven.
