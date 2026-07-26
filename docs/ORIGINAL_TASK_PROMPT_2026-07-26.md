# Original Task Prompt — verbatim — 2026-07-26

This is the **exact, unedited** text of the task prompt the user gave at the start of the
session that produced `port/finish-five-family-optimization-stack-2026-07-26`
(base `production/fullA-exact @ f1fa8e770759c62b3f96c1024dd310f235ea463e`). It is preserved here
verbatim — including every mathematical/architectural detail it specifies (operator partitions,
the cross-Hessian formula, the interval-basis derivation requirements, the required runtime
counters, the acceptance invariants, and the final-verdict format) — so that a future session
picking up the unfinished work has the same ground truth this session worked from, rather than a
paraphrase that could silently drop a detail.

See `FIVE_FAMILY_OPTIMIZATION_COMPLETION_MASTER_REPORT_2026-07-26.md` in this same `docs/`
directory for what this session actually completed against this prompt, and
`HANDOFF_TO_NEXT_SESSION_2026-07-26.md` for the scoped continuation prompt (Phases 5-7 priority,
plus everything else this prompt asked for that was not reached).

---

Claude Code continuation task: finish the five-family production optimization stack

Continue from the work package:

production_5x7_remediation_session_2026-07-26.zip [on dropbox via rclone here: C:\Users\edwar\Dropbox (Personal)\Gravity robustness\Analysis\Server Output]

The prior remediation branch contains useful, independently committed work, but it is not canonicalproduction and it did not complete the central operator conversion.

Its reported branch was:

port/remediate-production-5x7-audit-2026-07-26
HEAD = be434bc8c6126f76b24de7223f3f94402528ddd4
base = production/fullA-exact @ f1fa8e770759c62b3f96c1024dd310f235ea463e

Nothing was pushed or merged. The working tree also contained untracked test/debug files.

Your goal is to finish the work so that all five canonical counterfactual families use the bestvalidated implementation in every applicable numerical block, with public-driver runtime proof.

The five families are:

unrestricted;

flexible common marginals;

common Fréchet CDF marginals;

flexible CM + equal means / zero covariance;

origin-specific mean / zero covariance without CM.

Do not merge the old branch wholesale. Review, rebase, gate, and selectively adopt each commit.

0. Release discipline and inherited-work audit

Fetch the latest canonical:

production/fullA-exact
origin/production/fullA-exact

Record:

current canonical commit and tags;

whether production has advanced since f1fa8e7;

clean status;

all public drivers;

active backend manifest.

Create:

port/finish-five-family-optimization-stack-2026-07-26

Inspect every inherited commit and classify it:

ADOPT_UNCHANGED
ADOPT_AFTER_REBASE
ADOPT_AFTER_MORE_GATES
KEEP_OPT_IN
REWORK
DROP

At minimum assess:

unrestricted unified stage-runner change;

flexible-CM lookup FG port;

restricted exact cache;

restricted dual bank;

restricted compressed-core workspace;

CM+ZC constructor bugfix;

manifest/doc/dead-configuration fixes.

Freeze a report before modifying code:

INHERITED_REMEDIATION_COMMIT_REVIEW_2026-07-26.md

Use these release states literally:

INHERITED_WORK_AUDITED

IMPLEMENTED_ON_FEATURE_BRANCH

VALIDATED_ALL_FAMILIES

MATCHED_AB_PASSED

FAST_FORWARDED_TO_CANONICAL_PRODUCTION

TAGGED

POST_MERGE_SMOKE_PASSED

FINAL_PROFILE_COMPLETED

Use “merged” only at state 5 or later.

1. Adopt the low-risk inherited work after complete gates

1.1 Unrestricted unified public driver

Preserve the change that sends new unrestricted campaigns through the unified direct-bound driver:

run_polish_checkpointed_unified

with transformed/powered A-space as the fixed-theta default.

Retain the legacy profile driver only for explicit replication or completing an already-startedlegacy checkpoint.

Verify through the actual CLI/stage runner:

direct upper and lower bound semantics;

transformed-A default;

legacy-z explicit mode;

old schema-4 checkpoint refusal or explicit legacy resume;

checkpoint/resume;

startup manifest;

resolved outer algorithm.

Do not silently migrate a checkpoint for a different scientific outer problem.

1.2 Exact cache

The inherited production exact-cache implementation is promising and should normally be adopted,but rerun complete gates for all four restricted families:

flexible CM;

common Fréchet;

CM+ZC;

ZC only.

Use real D=20/W=80,000 public-driver sequences:

value at point A;

gradient at identical point A;

repeated value at A;

distinct nearby point B;

return to A.

Require:

same_point_inner_resolves = 0

and exact cache hit/miss accounting.

Verify cached objects are not later mutated through aliasing.

1.3 Compressed-core workspace

Adopt the reusable CompressedFactualWorkspace only after D=20 full-family tests for all fourrestricted families, including a real origin-ZC context.

Require:

no stale data across different outer points;

no post-warm-up resize;

exact winner hashes and core values;

workspace owned once per live context;

no scratch serialization into checkpoints.

1.4 Manifest and diagnostics

Adopt the Fréchet manifest resolver, live-condition CM+ZC Hessian label, origin-ZC docstring fix,and fail-fast price_cache_backend validation.

Clean all untracked scratch files or archive them outside the repository before release.

2. Do not promote the inherited restricted dual bank blindly

The inherited restricted bank uses only scaled outer-coordinate distance. It does not use theunrestricted bank’s KKT/residual proxy and was validated only on a tiny D=4 two-point sequence.

Treat it as:

KEEP_OPT_IN

until real trajectory evidence exists.

Benchmark for every restricted family:

dual_bank = off
dual_bank = inherited_distance_only

Report:

warm/cold solve counts;

KNITRO iterations;

complete inner-solve time;

failure/status mix;

verified outer progress;

selected distances;

harmful warm starts.

Because the new operator FG work below will make full restricted residual evaluation cheap,also consider a restriction-aware bank scorer that evaluates the candidate dual against thecomplete family moment operator—not only the economic core.

Only make a bank default if it improves or preserves real outer progress and does not increasefailures.

3. Core requirement: fixed-theta CM moments must be immutable

This is a mandatory architecture invariant.

For fixed theta, fixed draws, thresholds, and contrast basis, the following are immutable acrossall A/gp outer points:

origin/bin assignments;

raw interval indicators;

cumulative-CDF features;

origin contrasts;

common-Fréchet common-level features;

Fréchet targets.

They must be built once at context initialization.

Do not rebuild or rematerialize them at each outer point merely because an old dense interfaceexpects one contiguous matrix.

Create a central immutable object, for example:

CMImmutableFeatureOperator

owning:

compact bin indices;

threshold/grid metadata;

interval targets;

cumulative transform only when that basis is selected;

origin contrast transform;

Fréchet common-level direction and targets;

persistent forward/transpose/Hessian scratch;

context fingerprint.

For flexible theta, rebuild or update this object only when theta changes—not when A or gp changes.

Add counters:

cm_feature_context_builds
cm_feature_rebuilds_due_to_theta
cm_feature_rebuilds_due_to_A_or_gp
cm_dense_feature_materializations

For fixed theta require:

cm_feature_rebuilds_due_to_A_or_gp = 0

4. Mandatory invariant: full G must never be materialized in production hot paths

The current restricted dense FG path still requires a fully materialized moment matrix. That isthe central unfinished item.

Create a production operator interface supporting:

forward!:   r = G * lambda
transpose!: grad = G' * q
hessian blocks

without materializing full (G).

The full moment matrix may exist only in:

explicit reference/debug backends;

correctness tests;

small D=4 diagnostics.

Add runtime counters and, where practical, fail-fast guards:

full_G_materializations
dense_core_G_materializations
dense_restriction_G_materializations
generic_dense_FG_calls
operator_FG_calls

For ordinary production runs require:

full_G_materializations = 0
generic_dense_FG_calls = 0

The final public manifest must report:

moment_representation = operator
full_G_materialization = disabled

for every optimized production family.

5. Finish restricted inner FG operators for all families

5.1 Flexible CM

Use compact bin lookup for the CM block and compressed winner gather/scatter for the economic core.

Support the scientific basis selected by the family without building dense CM columns.

5.2 Common Fréchet

Use the same CM operator plus the common-level Fréchet direction.

Do not create a separate direct-country dense feature matrix.

5.3 Flexible CM + ZC

Use a partitioned operator:

[G=[E\ C\ Z],]

where:

(E): compressed economic core;

(C): immutable CM bin/contrast operator;

(Z): immutable mean/pair raw features plus low-dimensional dynamic targets.

Do not fall back to dense G merely because the extension block exists.

5.4 ZC only

Implement the unfinished partitioned operator:

[G=[E\ Z].]

Use:

compressed winner gather/scatter for (E);

exact dense or structured operations only for the smaller (Z) block;

direct target corrections without rebuilding centered matrices.

5.5 Preallocation and threading

The inherited flexible-CM lookup port improved wall time modestly but allocated more and thereforeremained off by default.

Find and remove the allocation sources:

do not rebuild CMLookupState per solve if dimensions/context are fixed;

persistent weighted histograms;

persistent forward/transpose outputs;

persistent per-thread scratch;

no per-callback vectors, matrices, closures, or outer-vector copies.

Wire the histogram/operator threading to the live worker policy rather than hardcoding one thread.

Benchmark:

workers = 1, 4, 8, 10, 20

Choose from complete inner-solve performance.

5.6 Production flip rule

For each family independently, make the operator FG default only if:

D=4 and D=20 correctness passes;

complete inner solve is faster or within 5%;

allocations fall materially;

no stability regression;

full G materialization count is zero.

Preserve dense reference as an explicit backend only.

6. Improve E × restriction Hessian cross blocks using winner structure

This is a mandatory investigation and likely implementation.

The common economic block is:

[E=Q-\nu\pi',]

where (Q) is winner sparse.

For any restriction block (R):

Q'SR-\pi(\nu'SR).]

Current CM cross-block code exploits bin/prefix structure on the restriction side but may stilltreat the economic side as a generic dense block.

6.1 CM and common Fréchet winner-bin cross operator

Build an exact candidate that:

uses the active winner coordinate for each draw/destination;

accumulates weighted winner contributions directly into restriction-bin tables;

applies interval/cumulative and origin-contrast transforms after aggregation;

applies the low-rank empirical-target correction;

handles non-winner-sparse economic columns separately;

never constructs dense E or dense CM features.

Compare against current Stab/bin-prefix cross construction.

6.2 CM+ZC

Use the winner-bin cross for the CM portion.

For mean/pair restrictions, derive and benchmark:

[Q'SZ-\pi(\nu'SZ)]

using winner scatter plus immutable raw restriction features.

6.3 ZC only

Benchmark a winner-feature cross operator for (H_{EZ}) against the current dense exact crossblock.

Do not force a specialized backend if the smaller dense block is faster.

6.4 Gate

Measure:

isolated cross-block time;

complete Hessian callback;

complete inner solve;

allocation;

real short outer progress.

Adopt family by family only if the complete solve benefits.

7. Compare interval and cumulative CM bases

This is a required bounded scientific/numerical experiment.

The cumulative and interval restrictions are related by an invertible linear transformation andhave the same feasible set when specified consistently.

Implement exact production candidates for:

basis = cumulative
basis = interval

For interval basis:

store one active bin index per origin/draw;

omit the residual interval or impose the exact rank normalization;

use interval-probability targets;

flexible CM imposes equality of interval probabilities across origins;

common Fréchet adds common-level Fréchet interval-probability targets.

7.1 Test origin contrast choices

Compare:

contrasts = anchored
contrasts = orthonormal

for both cumulative and interval bases.

This gives four arms:

cumulative + anchored;

cumulative + orthonormal;

interval + anchored;

interval + orthonormal.

Verify exact feasible-set equivalence and dual-coordinate transformations.

7.2 Re-audit the Hessian under intervals

Do not assume the current cumulative Architecture-C Hessian is optimal for interval moments.

For intervals:

forward operation should be one bin lookup per origin/draw;

transpose should be one scatter per origin/draw;

RR Hessian raw block is the weighted joint-bin contingency table directly;

no prefix sums should be required;

ER cross block should use winner-bin accumulation directly;

only contrast/common-level transformations remain.

Derive the exact optimal interval-basis Hessian construction from scratch.

Compare:

arithmetic complexity;

allocations;

callback time;

KNITRO iteration count;

conditioning;

complete inner solve;

outer progress.

Consider fixed column scaling for rare bins, but keep it a separate tested option.

7.3 Decision rule

Do not choose a basis based only on kernel time.

Adopt interval/orthonormal defaults only if:

scientific equivalence passes;

complete inner solves improve or remain stable;

outer progress improves or remains stable;

conditioning/iteration counts are acceptable.

Retain all reference modes for replication.

8. Transformed-A defaults for all fixed-theta families

Promote transformed/powered A-space to the default for new fixed-theta campaigns in:

unrestricted;

flexible CM;

common Fréchet;

CM+ZC;

ZC only.

Retain legacy-z as an explicit replication mode.

Add safe checkpoint behavior:

old coordinate-mode checkpoints are never silently reinterpreted;

explicit conversion is allowed only when mathematically exact and fingerprinted;

public startup prints the coordinate mode and mapping version.

Run matched legacy-z versus transformed-A checks after all operator changes are rebased.

9. Complete family-by-family public-driver gates

For every five-family public driver, run:

D=4 square;

D=4 rectangular non-last omission;

D=20/W=80,000;

calibration;

near-(\delta=1);

hard point;

upper and lower smoke;

checkpoint save;

process-group hard kill;

fresh-process resume;

exact cache;

dual bank on/off;

backend counters;

no silent fallback.

Do not infer one family’s success from shared code.

The inherited remediation introduced a CM+ZC constructor bug because earlier gates exercised onlyplain CM. Prevent recurrence with an explicit test matrix whose rows are the five real familycontexts.

10. Merge strategy

Use separate commits for:

inherited unrestricted driver fix;

inherited exact cache;

inherited workspace;

inherited diagnostics;

restricted dual-bank decision;

immutable CM operator;

flexible-CM operator FG;

common-Fréchet operator FG;

CM+ZC operator FG;

ZC-only operator FG;

CM/Fréchet winner-bin cross block;

ZC cross-block candidates;

interval-basis implementation;

orthonormal contrast/scaling experiment;

transformed-A default promotion;

kill/resume and runtime-proof infrastructure.

Merge only independently passing commits.

Do not let an experimental interval basis or cross-block candidate block safe cache/workspace/driverimprovements.

11. Only after remediation: final profiles and matrices

Once all passing changes are canonical, run five sequential 300-second direct upper-bound profiles:

D=20;

D_dest=19;

W=80,000;

seed 20260719;

delta=1;

calibrated start;

fixed theta;

production coordinate/basis defaults;

20 Julia threads;

one process at a time.

Profile:

unrestricted;

flexible CM;

common Fréchet;

CM+ZC;

ZC only.

Attribute wall clock to:

moment construction;

outer gradient;

inner function callback;

inner gradient callback;

(H_{EE});

(H_{RR});

(H_{ER});

screens;

cache/bank;

KNITRO internal work;

checkpoint/logging;

residual.

Regenerate:

final 5×7 numerical matrix;

final supporting-plumbing matrix;

allocation report;

flexible-theta overlay.

Every cell must contain runtime call counts and fallback counts.

12. Acceptance invariants

The final production defaults should satisfy, where applicable:

fixed_theta_cm_features_immutable = true
cm_feature_rebuilds_due_to_A_or_gp = 0
full_G_materializations = 0
generic_dense_FG_calls = 0
same_point_inner_resolves = 0
silent_backend_fallbacks = 0
shared_winner_pair_H_EE = active_all_families
cross_block_backend = measured_family_specific_winner_aware_or_reference_by_design
A_coordinate_mode = transformed_a

Do not claim FULLY_OPTIMIZED if any invariant is unverified.

13. Deliverables

Return:

FIVE_FAMILY_OPTIMIZATION_COMPLETION_MASTER_REPORT_2026-07-26.md;

INHERITED_REMEDIATION_COMMIT_REVIEW_2026-07-26.md;

IMMUTABLE_CM_FEATURE_OPERATOR_2026-07-26.md;

NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md;

RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md;

RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md;

WINNER_AWARE_CROSS_HESSIAN_BENCHMARK_2026-07-26.md;

INTERVAL_VS_CUMULATIVE_CM_BASIS_2026-07-26.md;

ORTHONORMAL_ORIGIN_CONTRAST_BENCHMARK_2026-07-26.md;

INTERVAL_BASIS_HESSIAN_OPTIMALITY_AUDIT_2026-07-26.md;

TRANSFORMED_A_ALL_FAMILIES_RELEASE_2026-07-26.md;

FIVE_FAMILY_PUBLIC_DRIVER_GATE_MATRIX_2026-07-26.md;

FIVE_FAMILY_KILL_RESUME_REPORT_2026-07-26.md;

final 5×7 matrix in MD and CSV;

final supporting matrix in MD and CSV;

five-family 300-second profile;

timing/allocation CSVs;

runtime counter JSON;

raw logs;

cold verification;

exact commits/tags/ancestry;

clean status;

SHA256 manifest.

Final verdict format:

CANONICAL_PRODUCTION_MERGE = yes | no

FAMILY_STATUS =
    unrestricted:<status>
    flexible_cm:<status>
    common_frechet:<status>
    cm_plus_zc:<status>
    zc_only:<status>

INNER_FG_BACKEND =
    unrestricted:<backend>
    flexible_cm:<backend>
    common_frechet:<backend>
    cm_plus_zc:<backend>
    zc_only:<backend>

CM_FEATURE_IMMUTABILITY = pass | fail
FULL_G_MATERIALIZATION = zero | present_<families>
CROSS_HESSIAN_WINNER_STRUCTURE = <family-by-family result>
CM_BASIS_DEFAULT = <cumulative/interval>
ORIGIN_CONTRAST_DEFAULT = <anchored/orthonormal>
TRANSFORMED_A_DEFAULT = <family-by-family result>

EXACT_CACHE = <family-by-family result>
DUAL_BANK = <family-by-family result>
SILENT_FALLBACKS = <integer>
POST_MERGE_SMOKES = pass | fail

HIGHEST_PRIORITY_REMAINING_GAP = <single item or none>
