# Addendum: Finish the shared inner-FG operator and eliminate dense G everywhere — verbatim, 2026-07-26

Saved verbatim as handed to the continuation session (per
`docs/HANDOFF_TO_NEXT_SESSION_2026-07-26_PARTB.md`'s own request that this addendum be preserved
as a file rather than living only in conversation history). This is the literal task prompt this
branch (`port/shared-inner-fg-operator-and-verification-2026-07-26`) is executing.

---

# Finish the shared inner-FG operator and eliminate dense G everywhere

Continue from:

* `frechet_lookup_and_unrestricted_allocfree_2026-07-26.zip` [via rclone here: C:\Users\edwar\Dropbox (Personal)\Gravity robustness\Analysis\Server Output]

The inherited branch already contains:

* a validated allocation-free unrestricted economic FG callback;
* a validated flexible-CM lookup FG backend, now default;
* a validated common-Fréchet CM+level lookup backend, still opt-in because complete-solve allocation remained higher;
* no CM+ZC operator FG;
* no ZC-only operator FG;
* strict verification still dependent on dense moment materialization;
* no global runtime proof that full (G) is absent from all production hot paths.

This addendum has exactly two goals:

1. Finish and productionize one shared, allocation-free economic FG operator for all five families.
2. Replace dense-(G) strict verification with operator-based verification for all five families.

Do not expand this task into Hessian cross-block optimization, interval-versus-cumulative basis experiments, dual-bank redesign, or final five-family profiling.

## 0. Branch and provenance

Fetch the latest canonical:

```text
production/fullA-exact
```

Inspect and selectively rebase the inherited commits from:

```text
port/finish-five-family-optimization-stack-2026-07-26
HEAD = fbb7d79cfe629a0cf8aefdc74603da29ad1401b3
```

Create:

```text
port/shared-inner-fg-operator-and-verification-2026-07-26
```

Do not merge the inherited branch wholesale.

Classify inherited commits:

```text
ADOPT
ADOPT_AFTER_REBASE
ADOPT_AFTER_MORE_GATES
REWORK
DROP
```

Use "merged" only after canonical ancestry and a production tag are verified.

# JOB 1 — one shared economic FG operator for every family

## 1. Preserve the validated unrestricted allocation-free kernel

The inherited unrestricted fix is the correctness and API starting point.

It introduced:

* `EconomicFGWorkspace`;
* `compressed_dual_contraction!`;
* `compressed_transpose_contraction!`;
* `compressed_cc_value_grad!`;
* direct writes into KNITRO's gradient buffer.

It reduced warmed per-callback allocation from approximately 2.57 MB to 128 bytes and was bit-identical to the old callback.

Do not rewrite this mathematics.

Rebase it onto current production, rerun its existing comprehensive integration suite, and use it as the common economic-core implementation.

## 2. Create a central economic operator interface

There must be one shared production implementation of:

[
E\lambda_E
]

and:

[
E'v,
]

where:

[
E=Q-\nu\pi'
]

is the winner-sparse economic moment block plus its low-rank target correction.

Create or finalize a central interface such as:

```julia
economic_forward!(
    draw_index,
    lambda_E,
    economic_operator,
    workspace,
)
```

and:

```julia
economic_transpose!(
    grad_E,
    draw_weights,
    economic_operator,
    workspace,
)
```

Requirements:

* winner gather for the forward pass;
* winner scatter for the transpose pass;
* exact sampling-weight and empirical-target corrections;
* no dense economic moment matrix;
* no per-callback W-scale allocation;
* one shared function used by every family;
* compatible with the same compressed winner state used by the shared (H_{EE}) backend.

Do not copy this algebra into family-specific files.

## 3. Family composition

Each family must assemble its full FG operator as:

```text
shared economic operator
+
family-specific restriction operator
```

### Unrestricted

```text
G = E
```

Use the inherited allocation-free shared economic operator.

### Flexible CM

```text
G = [E | C]
```

Use:

* shared economic operator for (E);
* immutable CM lookup/bin/contrast operator for (C).

### Common Fréchet

```text
G = [E | C | F]
```

Use:

* shared economic operator for (E);
* the same CM lookup operator for (C);
* the common-level Fréchet lookup operator for (F).

Do not use a dense country-by-country Fréchet matrix.

### Flexible CM + ZC

```text
G = [E | C | Z]
```

Use:

* shared economic operator for (E);
* CM lookup operator for (C);
* exact mean/pair operator for (Z).

Do not fall back to dense (G) because the ZC block is present.

### ZC only

```text
G = [E | Z]
```

Use:

* shared economic operator for (E);
* exact small restriction operator for (Z).

Do not materialize dense (E).

## 4. Restriction forward and transpose interfaces

Create a common family-level contract:

```julia
restriction_forward!(
    draw_index,
    lambda_R,
    restriction_operator,
    workspace,
)
```

and:

```julia
restriction_transpose!(
    grad_R,
    draw_weights,
    restriction_operator,
    workspace,
)
```

Family-specific implementations may differ internally.

### CM and Fréchet

Use precomputed bins, basis transforms, contrasts, and common-level direction.

### ZC blocks

Use immutable raw feature matrices or structured features plus direct target corrections.

For centered restrictions:

[
R=\Phi-\mathbf1t',
]

calculate:

[
R\lambda
========

\Phi\lambda-\mathbf1(t'\lambda),
]

and:

[
R'v
===

\Phi'v-t(\mathbf1'v).
]

Do not construct temporary centered matrices.

## 5. Persistent workspaces

Use persistent workspaces for:

* draw-level dual index;
* (\Psi'(r));
* economic gradient;
* restriction gradients;
* CM histograms;
* suffix/prefix buffers;
* origin transforms;
* Fréchet level operations;
* ZC contractions;
* per-thread sums.

Do not rebuild lookup-state objects per callback.

Audit whether workspaces are currently rebuilt once per inner solve. Where dimensions are campaign-fixed, prefer one workspace owned by the live context, unless lifecycle or thread safety requires otherwise.

## 6. Resolve the common-Fréchet allocation regression

The inherited common-Fréchet lookup path was 1.11–1.19× faster but allocated about 21% more per complete solve than dense reference.

Root-cause this before deciding the default.

Separate:

* allocation per FG callback;
* number of FG callbacks;
* KNITRO iteration-count differences;
* context/state construction;
* level-block workspace allocation;
* task or closure allocation.

If the per-callback operator is allocation-free and the total difference is caused only by a different iteration count, report that correctly rather than treating it as an allocation bug.

Make common-Fréchet lookup default if:

* callback allocation is no worse than dense;
* complete solve is faster;
* correctness and stability pass.

## 7. Complete CM+ZC and ZC-only operator FG

These were not implemented in the inherited work.

Implement both now.

Do not infer correctness from flexible CM.

Run family-specific D=4 and D=20 tests through the actual production contexts.

# JOB 2 — strict post-solve verification without dense G

## 8. Keep strict verification

Do not remove the post-solve verification step.

It is useful for independently checking:

* inner objective;
* full dual gradient;
* KKT residual;
* moment or feasibility residual;
* returned solver status;
* cache admission;
* incumbent promotion;
* cold verification.

The problem is not the check. The problem is that the old verifier requires dense (G).

## 9. Implement operator-based verification

Create:

```julia
verify_inner_solution_operator!(
    result,
    zeta,
    lambda,
    economic_operator,
    restriction_operator,
    fg_workspace,
    verification_workspace,
)
```

It must independently recompute:

[
r=-\zeta\mathbf1-G\lambda,
]

the exact objective, and:

[
g_\lambda=-\frac1W G'\Psi'(r)
]

using the shared economic and family-specific restriction operators.

"Independent verification" means:

* recompute from immutable/operator state;
* do not reuse stale callback output;
* do not require dense (G).

Use separate output or scratch buffers if needed to prevent accidental reliance on live callback state.

## 10. Remove dense-verification compatibility switches

The inherited CM lookup path uses a mutable switch such as:

```text
skip_cm_fill_ref
```

to avoid dense CM-column filling during ordinary solves, then re-enable it for verification.

Once operator verification passes:

* remove this switch from the production path;
* do not rematerialize dense CM columns for verification;
* retain dense verification only as an explicit reference or debug backend.

Do not leave scientific feature existence controlled by a shared mutable `Ref`.

## 11. Dense reference remains available only explicitly

Support:

```text
verification_backend = :operator
verification_backend = :dense_reference
```

The production default must be `:operator`.

The dense reference backend may be used for:

* D=4 tests;
* numerical-equivalence checks;
* debugging.

It must never be reached silently.

# MANDATORY NO-G INVARIANT

## 12. Runtime counters

Add:

```text
full_G_materializations
dense_economic_G_materializations
dense_CM_G_materializations
dense_ZC_G_materializations
generic_dense_FG_calls
operator_FG_calls
operator_forward_calls
operator_transpose_calls
operator_verification_calls
dense_reference_verification_calls
```

For ordinary production value, gradient, cache, and verification sequences require:

```text
full_G_materializations = 0
dense_economic_G_materializations = 0
generic_dense_FG_calls = 0
dense_reference_verification_calls = 0
```

Add an opt-in fail-fast mode that throws if a production hot path attempts to materialize full (G).

## 13. Audit all remaining consumers of dense G

Search the repository for every:

* dense moment builder;
* `gemv!` against draw-by-moment matrices;
* `select_G_from_H`;
* `obj.H[:, ...]`;
* generic verifier;
* diagnostic or cache scorer;
* post-processing routine.

Classify each use:

```text
PRODUCTION_HOT_PATH
PRODUCTION_SETUP_ONLY
EXPLICIT_REFERENCE
TEST_ONLY
DEAD_CODE
```

No `PRODUCTION_HOT_PATH` dense-(G) consumer may remain after this task.

If `obj.H` is a moment-feature matrix rather than a Hessian, document that clearly. Rename only if safe and bounded.

# CORRECTNESS GATES

## 14. D=4

For every family test:

* square layout;
* rectangular layout;
* non-last omitted destination;
* multiple restriction configurations;
* operator FG versus dense reference;
* operator verification versus dense verification;
* exact-cache hit;
* repeated value-then-gradient sequence.

Require machine-precision agreement.

## 15. D=20/W=80,000

Use:

* omit-ROW;
* seed `20260719`;
* fixed theta;
* transformed-(A) coordinates;
* 20 Julia threads.

At calibration, near-(\delta=1), and one hard point test:

1. unrestricted;
2. flexible CM, (L=50);
3. common Fréchet, (L=50);
4. CM+ZC, `K_mean=1,K_pair=1`;
5. ZC only, `K_mean=1,K_pair=1`.

Compare:

* draw-level dual index;
* objective;
* complete gradient;
* KNITRO iterations;
* final dual solution;
* (\Delta^*);
* KKT residual;
* moment residual;
* verification classification;
* cache admission;
* cold verification.

Require full-(G) counters to remain zero in operator mode.

# PERFORMANCE GATES

## 16. Per-callback and complete inner-solve A/B

For every family compare:

```text
dense FG + dense verification
```

against:

```text
shared economic operator
+ family restriction operator
+ operator verification
```

Report:

* economic forward time;
* restriction forward time;
* divergence time;
* economic transpose time;
* restriction transpose time;
* complete FG time;
* verification time;
* complete inner-solve time;
* allocation;
* garbage collection;
* KNITRO iterations.

## 17. Short real outer A/B

For every family whose complete inner solve improves materially, run a matched short direct-bound outer A/B from calibration.

The only intended difference is dense versus operator FG or verification.

Report verified progress and runtime counters.

# PRODUCTION DEFAULTS AND MERGE

## 18. Backend manifest

Every public driver must report:

```text
economic_fg_backend = compressed_operator
restriction_fg_backend = <family-specific>
verification_backend = operator
moment_representation = operator
full_G_materialization = disabled
```

Also print call and fallback counters.

## 19. Default rule

Make operator FG and operator verification default independently by family if:

* correctness passes;
* complete inner solve is faster or within 5%;
* allocation improves materially;
* no stability regression;
* short outer progress is preserved.

Preserve dense reference only as explicit debug mode.

Do not leave an optimized backend off by default without a measured reason.

## 20. Release structure

Use separate commits for:

1. shared economic operator;
2. unrestricted allocation-free port or rebase;
3. flexible-CM restriction-operator cleanup;
4. common-Fréchet operator-allocation fix;
5. CM+ZC operator FG;
6. ZC-only operator FG;
7. operator verification;
8. no-(G) counters or fail-fast guards;
9. public manifests and tests.

Merge only independently passing commits.

# DELIVERABLES

Return:

* `SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md`;
* `UNRESTRICTED_ALLOCATION_FREE_FG_FINAL_GATE_2026-07-26.md`;
* `FLEXIBLE_CM_OPERATOR_FG_FINAL_GATE_2026-07-26.md`;
* `COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md`;
* `CM_MEANZC_OPERATOR_FG_PORT_2026-07-26.md`;
* `ORIGIN_ZC_OPERATOR_FG_PORT_2026-07-26.md`;
* `OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md`;
* `NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md`;
* `FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`;
* runtime-counter JSON;
* timing and allocation CSVs;
* raw logs;
* exact commits, tags, and ancestry;
* clean status;
* SHA256 manifest.

Final verdict:

```text
ECONOMIC_FG_BACKEND =
    unrestricted:<backend>
    flexible_cm:<backend>
    common_frechet:<backend>
    cm_plus_zc:<backend>
    zc_only:<backend>

RESTRICTION_FG_BACKEND =
    unrestricted:not_applicable
    flexible_cm:<backend>
    common_frechet:<backend>
    cm_plus_zc:<backend>
    zc_only:<backend>

VERIFICATION_BACKEND =
    unrestricted:<backend>
    flexible_cm:<backend>
    common_frechet:<backend>
    cm_plus_zc:<backend>
    zc_only:<backend>

FULL_G_MATERIALIZATION =
    zero_all_production_families |
    present_<families>

GENERIC_DENSE_FG_CALLS = <integer>
DENSE_REFERENCE_VERIFICATION_CALLS = <integer>

PRODUCTION_MERGE =
    merged_all |
    partial_merge |
    port_ready_not_merged |
    not_ready
```

A verdict of `zero_all_production_families` requires runtime proof through ordinary inner solves, strict verification, exact-cache sequences, and all five public drivers.
