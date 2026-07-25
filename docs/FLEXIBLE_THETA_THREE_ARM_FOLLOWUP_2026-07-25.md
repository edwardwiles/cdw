# Three-arm matched comparison + flexible_aspace gradient bug fix, 2026-07-25 (follow-up)

Follow-up to `FLEXIBLE_THETA_ASPACE_PRODUCTION_PORT_2026-07-25.md`. Requested by the user after
the port session: (1) diagnose the KNITRO hang from the port session, (2) get a reliable 300s
delta=2 comparison, (3) expand to three cases (flexible_aspace, fixed_aspace, fixed_legacyz) at
both delta=1 and delta=2.

## Real bug found and fixed: flexible_aspace gradient callback crash

`matched_comparison_three_arm.jl` (new, this session) exercises `run_polish_checkpointed_unified`
in `:flexible`/`:powered_aspace` mode for the first time under real D=20 load. It crashed
immediately (`KN_RC_CALLBACK_ERR`, wall~17-19s, n_eval=1) on every attempt.

**Root cause** (confirmed via a captured Julia backtrace, not inferred): a Julia operator-
precedence bug in `c10_d20_production_driver_unified.jl`'s `cb_G!`:

```julia
xf_reduced = vcat(d.gp, d.xf[layout.trade_elasticity_mode == :flexible ? 3 : 2:end])
```

In Julia, `cond ? a : b:c` parses as `cond ? a : (b:c)` -- the ternary's true-branch is the bare
literal `3`, not the range `3:end`. When `trade_elasticity_mode == :flexible`, this silently
indexed `d.xf[3]` (a single scalar) instead of `d.xf[3:end]` (the full 380-element powered-A
block), truncating `xf_reduced` from 381 elements to 2. This propagated into
`composite_gradient_at_Cplus` -> `lfix_factorized_workspace.jl:348`'s
`reshape(x_free0[2:end], D, Ddest)`, which failed with
`DimensionMismatch("new dimensions (20, 19) must be consistent with array length 1")` --
exactly matching the observed crash.

**Fix**: explicit parens, `d.xf[(layout.trade_elasticity_mode == :flexible ? 3 : 2):end]`.
Verified with a real 60s run showing correct gradient evaluations (n_grad_calls=3, clean
cold-verify) before relaunching the full 300s comparisons.

This bug is orthogonal to both hang theories below -- it's a silent wrong-answer/crash bug in
code that was new this session (the unified driver's flexible-mode branch), not a timing/hang
issue at all.

## KNITRO hang investigation: partial retraction

An initial diagnosis blamed the earlier port session's flexible/delta=2 hang (confirmed stuck
inside `KN_solve`/`KTR_solve` native code, unpreemptable by SIGTERM) on `prestep/iterWagesPreStep!.jl`,
a damped fixed-point wage iteration with no wall-clock abort. **This was overreach and is
retracted**: a controlled comparison (4 concurrent runs traversing the identical `d20_real_setup_design`
call, under MORE load than an isolated smoke test that appeared to hang) showed all 4 sailing
through that same setup code within a few minutes. The far more mundane explanation: this
codebase's cold-start cost (Julia startup + JIT-compiling a large include chain + real-data
context construction) is genuinely on the order of 1-3 minutes, and the smoke test's
`timeout 180 --kill-after=15s` wrapper simply wasn't generous enough for normal startup latency
-- it was killed mid-flight while still legitimately progressing, not stuck.

**The original delta=2 `KN_solve` hang's root cause remains unresolved.** The symptom is solid
(sustained ~95% CPU, stuck deep in native KNITRO code, immune to SIGTERM, needed `kill -9`) but
no deeper mechanism was confirmed -- only a plausible-but-unproven hypothesis (a single expensive
internal KNITRO iteration on a hard candidate overrunning the per-call time-limit check, blocking
in a ccall Julia cannot preempt). Mitigated operationally this session via generous
`timeout --kill-after=Ns` wrappers plus active monitoring; not root-caused at the KNITRO-internals
level.

## Three-arm matched comparison (300s budget, algorithm=auto+SR1, real D=20/W=80,000/seed=20260719)

All arms: identical calibrated start point, draws, screens, cache, incumbent logic; only
`(trade_elasticity_mode, A_coordinate_mode)` differs; all run through the same
`run_polish_checkpointed_unified` driver. All cold-verified.

| Case | delta=1 kappa | delta=1 wall / n_eval | delta=2 kappa | delta=2 wall / n_eval |
|---|---|---|---|---|
| flexible_aspace | 0.0658808549783364 | 410.4s / 22 | 0.07382467242715973 | 304.8s / 8 |
| fixed_aspace | 0.0649042169269014 | 306.8s / 15 | 0.07118083447894263 | 342.7s / 12 |
| fixed_legacyz | 0.05428070163460086 | 321.1s / 10 | 0.0686440433431843 | 329.5s / 15 |

Raw logs: `key_results/threearm_*.log` (this doc's companion push).

Directionally consistent with every earlier finding this session: a-space beats legacy z-space at
fixed theta (+19.6% delta=1, +3.7% delta=2), flexible beats fixed-aspace on top of that (+1.5%
delta=1, +3.7% delta=2). Single-run-per-cell, matched-budget signal only, not a global-optimum
claim -- same caveat as every other matched comparison this session.
