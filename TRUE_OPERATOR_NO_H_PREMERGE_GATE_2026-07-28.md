# True no-H operator bundle: pre-merge regression gate (2026-07-28, fresh rerun)

Branch: `work/true-operator-bundle-no-H-2026-07-28` @ `8b0c931940b33b7a2665659e16789aecd007cd2d`.
Per `TRUE_OPERATOR_NO_H_PRODUCTION_RECONCILIATION_2026-07-28.md`, canonical production (`cdw`
remote) has not advanced beyond this branch's base (`93f26df`) — this is a clean fast-forward, no
rebase. Every gate below was rerun **fresh, from scratch, this session** (not reused from the
branch's own prior commits) to prove the exact current tree state, not just cite history.

Environment: `julia 1.12.6` (juliaup), `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, per this
project's standing thread-hygiene requirement.

## Field inspection (structural, not just grep)

`full_aod_diag/d4_exact/operator_psi_bundle.jl`, `OperatorPsiBundle` struct fields:
`δ, find_smallest, γ, l, inequality_index, complement_index, U, M, N, outer_constr_index,
inner_loop_opt, lower_limit, use_cached_x, Psi!, dPsi!, ddPsi!, payoff, grav_col, H_save, arg0,
arg1, arg2, x, threshold_state, economic_state, restriction_state`.

```
has H field           = false
has H_copy field       = false
has G field            = false
has K field            = false  (renamed `payoff`)
has ones field         = false
has moments! field      = false
has select_G_from_H interface = false  (no such method dispatches on this type)
```

Every D=4 and D=20 gate below also runs the script's own live structural assertions
(`obj isa OperatorPsiBundle`, `!hasfield(..., :H)` etc., and `obj_o.H throws`) as PASS/FAIL lines,
not just static source reading.

## D=4 gates (all five families) — fresh rerun

Command (per family): `julia --project=. full_aod_diag/d4_exact/test_operator_no_H_bundle_equivalence_<family>.jl`

| Family | PASS | FAIL | Verdict |
|---|---|---|---|
| unrestricted | 24 | 0 | ALL OPERATOR-VS-DENSE UNRESTRICTED EQUIVALENCE GATES PASSED |
| flexible CM | 36 | 0 | ALL OPERATOR-VS-DENSE FLEXIBLE-CM EQUIVALENCE GATES PASSED |
| common Fréchet | 30 | 0 | ALL OPERATOR-VS-DENSE COMMON-FRECHET EQUIVALENCE GATES PASSED |
| CM+ZC | 30 | 0 | ALL OPERATOR-VS-DENSE CM+ZC EQUIVALENCE GATES PASSED |
| ZC-only (origin-ZC) | 30 | 0 | ALL OPERATOR-VS-DENSE ORIGIN-ZC EQUIVALENCE GATES PASSED |

Every family: field-inspection PASS block, calibration inner solve (both backends feasible, same
`nStatus`), objective/gradient/packed-Hessian agreement at `x=0`, 2-4 random probe points, the
real solved `x*` point, and a complete inner solve from `x_free_calib` with `Δζ*=0.0`,
`max|Δλ*|=0.0`. Flexible CM additionally re-verifies under `threaded_bins=true` (the actual
production default), per the prior session's lesson that the serial path alone is insufficient.

Raw logs: `d4_unrestricted.log`, `d4_flexcm.log`, `d4_frechet.log`, `d4_cmzc.log`,
`d4_originzc.log` (see log manifest below).

## Real D=20/W=100,000 gates (all five families) — fresh rerun

Command (per family): `julia --project=. full_aod_diag/d4_exact/test_operator_no_H_bundle_equivalence_<family>_d20.jl`,
context via `d20_real_setup(W=100_000, δ=1.0, find_smallest=true, destination_sample=:exclude_row)`
(production data; internal dense-vs-operator comparison built from the same in-process context, so
`d20_real_setup`'s lack of an explicit RNG seed does not affect equivalence-gate validity — see
memory `gateC-d20-real-setup-vs-d20-real-setup-design-seeding` for when seeding *does* matter,
i.e. cross-run/cross-process reproducibility, which is not what these gates check).

| Family | PASS | FAIL | Verdict |
|---|---|---|---|
| unrestricted | 23 | 0 | ALL OPERATOR-VS-DENSE UNRESTRICTED D=20/W=100000 EQUIVALENCE GATES PASSED |
| flexible CM | 22 | 0 | ALL OPERATOR-VS-DENSE FLEXIBLE-CM D=20/W=100000 EQUIVALENCE GATES PASSED |
| common Fréchet | 22 | 0 | ALL OPERATOR-VS-DENSE COMMON-FRECHET D=20/W=100000 EQUIVALENCE GATES PASSED |
| CM+ZC | 22 | 0 | ALL OPERATOR-VS-DENSE CM+ZC D=20/W=100000 EQUIVALENCE GATES PASSED |
| ZC-only (origin-ZC) | 22 | 0 | ALL OPERATOR-VS-DENSE ORIGIN-ZC D=20/W=100000 EQUIVALENCE GATES PASSED |

Every family: field-inspection PASS, complete real full inner solve at calibration for BOTH
backends (dense-reference and operator) from the identical initial dual/solver settings, same
accepted `nStatus` (`0` in every case), objective/gradient/packed-Hessian agreement at `x=0` and 2
random probe points and the real solved `x*`, and full-solve `Δζ*=0.0`, `max|Δλ*|=0.0` to the
gate's stated tolerance (`1e-8`/`1e-6`). Flexible CM's log additionally shows the operator backend
running ~2.6x faster than dense on the full real inner solve (9.19s vs 24.20s) — a side observation,
not a target of this gate (no performance tuning was reopened).

Raw logs: `d20_unrestricted.log`, `d20_flexcm.log`, `d20_frechet.log`, `d20_cmzc.log`,
`d20_originzc.log`.

## Operator verification cross-check (flexible CM) — fresh rerun

Command: `julia --project=. full_aod_diag/d4_exact/check_flexcm_d20_verification.jl`
(`archC_verified_state`, `verification_backend=:operator`, real D=20/W=100,000, calibration point).

```
                          dense                    operator                 |Δ|
inner_status              0                         0                        0
Delta_dual                0.007373918352170868      0.007373918352170868     0.0
Delta_primal              0.007373918352169704      0.007373918352169704     0.0
primal_dual_gap           1.1639994523804376e-15    1.1639994523804376e-15   0.0
weight_norm_resid         0.0                       0.0                      0.0
mean_m_resid              0.0                       0.0                      0.0
max_abs_moment_kkt_resid  4.2724999853028526e-13    4.2724999853028526e-13   0.0
m_mean                    1.0                       1.0                      0.0
m_min                     0.3295696606195748        0.3295696606195748      0.0
m_max                     2.9832435628989056        2.9832435628989056      0.0
```

All 10 verification quantities bit-identical between dense-reference and operator bundle; both
KKT/primal-dual residuals are at or below ~1e-13/1e-15, i.e. machine-precision-satisfied. `nStatus`
label (`-103`/`0` depending on the specific run's KNITRO stopping path) reflects the same
genuinely-feasible solution either way -- confirmed identical between backends in this run.

Raw log: `d20_flexcm_verification.log`. (This particular cross-check script exists only for
flexible CM per the prior session's ad-hoc addition; the other four families' D=20 equivalence
gates above already independently prove operator-vs-dense agreement on the full inner solve,
objective, gradient, packed Hessian, and dual vector to machine precision, which is the substantive
claim — a dedicated `verification_backend=:operator` cross-check script for the other 4 families
does not exist in this branch's history and was not created new here, to avoid reopening scope
beyond reconciliation.)

## Forbidden-interface checks

`obj_o.H` (accessing the `H` field on an `OperatorPsiBundle` instance) throws in every D=4 and
D=20 gate log (`PASS  obj_o.H throws`) — a real Julia `FieldError`, not a soft check.

## SHA256 manifests

Scripts (production bundle type + all 10 equivalence-gate scripts + verification cross-check):

```
ce55458700b9de4587a347714eddeded26a66b8e8f641369840b35bac71d4ae5  test_operator_no_H_bundle_equivalence_cmzc_d20.jl
6d1b4c7a218e90316e15cccb26eb916e60eae078985b7fb72b879f5dc04c59d8  test_operator_no_H_bundle_equivalence_cmzc.jl
10ce4065ee945b2200c902539ef9c79d0377f70e0330159302a3ff26b8f0fd5f  test_operator_no_H_bundle_equivalence_flexcm_d20.jl
6eae3322e7342b097ee8e055385c83a3efe07fb1b5dba07eaea1ae643ab3d52e  test_operator_no_H_bundle_equivalence_flexcm.jl
09b52cec6f467d2380467ac001c85ee06d39409b1689e7f087a3374aed30d43f  test_operator_no_H_bundle_equivalence_frechet_d20.jl
494f28bd24a528ceb734f90c50c1320788998b4440a9d0e71c32995fe1e2912b  test_operator_no_H_bundle_equivalence_frechet.jl
55afb7ae5f52c20f61900619a05ab0ecf0d6bd7ef22a0a36aa5f9bafd2aab0b7  test_operator_no_H_bundle_equivalence_originzc_d20.jl
87bb7e4af2db850cbe98b2829c42b120dacbd78b31b7332ac8032afaf3e7b542  test_operator_no_H_bundle_equivalence_originzc.jl
81f9e399fccf12ea2f6007e8f0cb092dc8840a4f6ea5de5da7318eca2d80cc3c  test_operator_no_H_bundle_equivalence_unrestricted_d20.jl
cf867f7692e173a896e03c71bddaf661dbc2a8d35b543c4b87b416f27b4129d2  test_operator_no_H_bundle_equivalence_unrestricted.jl
80c327d4485592e3fb946c1a9a9e6438e242d6efbe3b0a5cfd42bab9d14a12c9  check_flexcm_d20_verification.jl
4df7ec6d78733da357e6088869ebffbfc18353a5e73eaa452e46087ff6b870d4  operator_psi_bundle.jl
```

Raw fresh logs (SHA256, this session's output only):

```
c7be340fa68fc5d86abaf1a811a8f3b3c5019afda1caffe1f0fa9c3a5b11f8de  d4_cmzc.log
245b4e396faea2ba31df0addfd0cd0819223aa0ba4849a4f1d7fad8d069cfbdd  d4_flexcm.log
8ac930d04d79996bb3c1bac7dc25412bacc303a407cd5c348c6ef21339ce3201  d4_frechet.log
69f0e8e8a5cc99aa820f49f325eaf3045f45e9ac63954a114cf977f70089dde9  d4_originzc.log
646c458321e7f41e460ae48c0b87265bd6e65eebe835740dd30a07db19e4ef30  d4_unrestricted.log
0d790061a589d57a6f0d1d947cd764a1c326031d3f25ef2118b7a3941db479f8  d20_cmzc.log
7d55ec6f591e5745de36f305c9b03f770b5a363ccd9e9890a19fc012e20c619c  d20_flexcm.log
24aca62fb5478ae1428e22231197588d57cb10b5ea44ffdcd4a42d151f8a275d  d20_flexcm_verification.log
cc2587225c411d27f1812f3495bffb033a861970bcf6548c11b83785fea78ccd  d20_frechet.log
0fb0dd4c60675864c75a00ab8e7e4632e82b863462d390cd3d80df5ce4b8f0e5  d20_originzc.log
70bba456d1532f4fc48f7101563b1ef0c73c883b754e086e8f7d3454c3fbbcff  d20_unrestricted.log
```

## Operational note (self-correction, recorded for completeness)

Mid-run, the flexible-CM D=20 job was launched twice due to a manual `&`/`disown` mistake inside a
`run_in_background` Bash call (violates this project's own "no nohup/disown in background Bash"
guidance) — the first launch's completion notification fired immediately without the real process
finishing, and its log file was briefly shared with a second, correctly-launched process, corrupting
early log content with an interleaved SIGTERM backtrace. Caught immediately via `ps`, the stray
process was killed, the corrupted log deleted, and the flexible-CM D=20 gate was relaunched clean
end-to-end — the "flexible CM D=20" row and log above are from that clean rerun. No gate result in
this report is affected.

## Verdict

```
D4_GATES_ALL_FIVE_FAMILIES       = pass (0 failures each, fresh rerun)
D20_W100000_GATES_ALL_FIVE_FAMILIES = pass (0 failures each, fresh rerun)
OPERATOR_VERIFICATION_CROSSCHECK  = pass_bit_identical (flexible CM; other 4 families covered by
                                     their own D=20 equivalence gates above, not a dedicated
                                     verification-backend script)
FORBIDDEN_FIELDS_PRESENT          = none (structural, all 5 families)
FORBIDDEN_INTERFACE_CALLS         = none observed in any gate
READY_FOR_CANONICAL_MERGE         = yes
```
