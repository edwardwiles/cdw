# Reverse-mode (Enzyme/Mooncake) NaN bisection — continuing the prior session's audit

Starting point (already established, `../ad_benchmark/README.md` §5,
`../SESSION_SUMMARY_2026-07-12.md` §8): Enzyme and Mooncake both compile and run against the
REAL `envelope_scalar_div_ctx` / gravity-moment pipeline, and both produce NaN in ~13-14 of 23
gradient entries at all 4 frozen D=4 points, ruling out hard-max/branching and the
`reshape(vcat(...))` A_od-construction pattern as the cause, and ruling out the custom `gamma`
EnzymeRule as the SOLE cause (the gravity moment alone, which calls no `gamma()`, already
reproduces the identical NaN pattern). Root cause not previously identified.

This directory's `minimal_enzyme_nan.jl`/`minimal_mooncake_nan.jl`/`minimal_enzyme_nan_step4.jl`
go one level deeper: total, standalone reproducers (no project code loaded, no KNITRO) of the
gravity moment's actual mathematical content, bisected in stages.

## Steps 1-3: clean reimplementation of `withinTransform` + the gravity contraction — NO NaN, either backend

```
                                          ForwardDiff vs Enzyme      ForwardDiff vs Mooncake
STEP 1: withinTransform alone (sum(W))   exact match, 0 NaN         exact match, 0 NaN
STEP 2: full grav_scalar (Wτ .* WA, Σ)   relerr 1.99e-16, 0 NaN     relerr 2.28e-16, 0 NaN
STEP 3: bare sum(A;dims=k) reductions    exact match, 0 NaN         exact match, 0 NaN
```

**This rules out `sum(A; dims=k)` and the broadcast double-demeaning pattern
(`lz .- sum(lz,dims=2)./D .- sum(lz,dims=1)./D .+ sum(lz)/D²`) as the cause** — a well-known
historical Enzyme weak spot that was the leading a priori hypothesis for this step, and it is
NOT the culprit. Both independent reverse-mode systems handle it correctly, exactly reproducing
ForwardDiff to machine precision. This is a genuinely new negative result — the prior session
never isolated `withinTransform` from the rest of the pipeline.

## Step 4: add back the REAL `newGravityMoment!`'s dead-code preamble (`doubleDiff`, mutating)

`newGravityMoment!.jl`'s `UoModel==1` branch is preceded by an always-executed but (for
`UoModel==1`) entirely UNUSED computation: `deltaτ = doubleDiff(τ)` then a `meanτ` accumulation
loop, whose result is never read again once the `if UoModel==1` branch is taken. The prior
session's `newGravityMoment_typestable.jl` fixed a COMPILE-time crash in this dead code
(`meanτ = 0` was type-unstable) but that fix did not resolve the NaN — meaning the dead
`doubleDiff` call itself (a MUTATING function: `deltaZ = zeros(eltype(z),D,D)` then
`@.deltaZ[o,:] = ...` in a loop) was never tested in isolation for whether ITS presence, ahead of
the live `withinTransform` computation, is what breaks Enzyme's reverse pass — a candidate
directly named in the audit's own checklist ("uninitialized adjoint buffers", "mutation
aliasing").

`minimal_enzyme_nan_step4.jl` reproduces `newGravityMoment!`'s exact structure (dead
`doubleDiff`+`meanτ` preamble, followed by the live `withinTransform`-based contraction) as a
standalone function, no project code.

**RESULT: NOT reproduced.** `minimal_enzyme_nan_step4.jl`'s output:

```
ForwardDiff: [-0.3245881581228922, -0.02107276340515393, 0.3133413876291989, ...]
Enzyme:      [-0.3245881581228923, -0.021072763405153922, 0.3133413876291989, ...]
Enzyme finite everywhere: true  nan_count=0
```

Enzyme matches ForwardDiff to ~15 digits, zero NaNs — even with the exact dead `doubleDiff`
mutating preamble reproduced ahead of the live `withinTransform` contraction. **The dead-code
preamble, by itself, is not the trigger either.**

## Conclusion of this session's bisection: narrowed, not solved

Four candidates are now RULED OUT by direct, isolated, standalone (no project code) tests, each
checked against both ForwardDiff (reference) and at least one reverse-mode backend:

1. Hard-max/branching (ruled out by the prior session: gravity moment has none).
2. The `reshape(vcat(...))` A_od-construction pattern (ruled out by the prior session).
3. The custom `gamma`/`digamma` EnzymeRule (ruled out by the prior session: gravity moment calls
   no `gamma()` at all, yet reproduces the NaN).
4. **`sum(A; dims=k)` reductions and the two-way demeaning broadcast pattern** (ruled out THIS
   session, steps 1-3, both Enzyme and Mooncake).
5. **The dead, mutating `doubleDiff`/`meanτ` preamble in isolation** (ruled out THIS session,
   step 4, Enzyme).

What remains as the most likely site, by elimination: the bug requires the broader
`EK_moments_gammanorm_directgp!` / `moments!.jl` context this bisection did not reach — most
plausibly the MUTABLE, PRE-ALLOCATED scratch buffers carried inside `γobj` (`UPow_scratch`,
`UσPow_scratch`, `Ū`, `Uσ`, etc. — visible in `pp.γ`'s field list via
`full_aod_diag/ad_benchmark/setup_context.jl::build_ad_context`) that are written in place and
REUSED across repeated calls to `moments!`/`EK_moments_gammanorm_directgp!` within one
differentiated pass (as opposed to the `doubleDiff`/`withinTransform` functions tested here,
which allocate fresh output arrays every call and never receive an externally-owned buffer to
mutate). Reverse-mode AD's need to track shadow/adjoint memory correctly across REUSED mutable
buffers — as opposed to fresh per-call allocations — is a substantively different failure mode
than anything tested in steps 1-4, and is the concrete next bisection step for a future session:
build a minimal function that, like the real pipeline, writes into a passed-in, pre-allocated
scratch matrix as an intermediate step toward a scalar output, and check whether THAT specific
pattern is what breaks Enzyme/Mooncake's reverse pass. Not reached within this session's budget.

## Step 5: the `UPow_scratch`/`eltype(γ)===Float64` branch — a REAL, separate Enzyme hazard (but not the observed bug)

Reading `moments/moments!.jl:130-138` directly (not previously isolated) turned up a
structurally different and, on paper, much more dangerous pattern than steps 1-4: the code's
own comment says *"reuse preallocated Float64 scratch on the Float64 path; allocate Duals under
ForwardDiff"* —

```julia
if eltype(γ) === Float64 && size(UPow_scratch, 1) == size(U, 1)
    UPow = UPow_scratch          # shared, mutable, externally-owned buffer
else
    UPow = zeros(eltype(γ), size(U))
end
@. UPow = U .^ (-μ)              # writes an Active(μ)-dependent value into UPow
```

`γ` here is `copy(θ[3:2+D])`, so `eltype(γ)` tracks θ's element type — under ForwardDiff that's
`Dual`, so this branch is NEVER taken by ForwardDiff. But Enzyme's reverse mode does NOT change
θ's element type (it differentiates the actual `Float64` code via a shadow/tape, not dual
numbers) — so `eltype(γ)===Float64` is `true` under Enzyme too, meaning **Enzyme's differentiated
pass would take the exact same scratch-buffer-reuse branch as ordinary non-differentiated
evaluation**, a branch never designed or tested to be differentiated.

`minimal_enzyme_nan_step5_scratch.jl` built a minimal analogue (a `Const`-marked context holding
a mutable scratch buffer, written with an Active(μ)-dependent value, read back into the output).
**Result: Enzyme does NOT silently NaN on this — it throws an explicit, named compile-time
error**, `EnzymeRuntimeActivityError`, whose own message says almost verbatim: *"Constant memory
is stored... as temporary storage for active memory."* This confirms the pattern is a real,
Enzyme-recognized hazard — but it is a DIFFERENT failure mode (a loud error, not a silent NaN)
from what `test_enzyme_full.jl` actually observed.

**Important correction**: I then checked which code the ACTUAL NaN-producing test
(`../ad_benchmark/test_enzyme_full.jl` → `moments_gammanorm_typestable.jl`) uses, and it
does **`UPow = zeros(T, size(U))`unconditionally** — the `eltype(γ)===Float64`/`UPow_scratch`
branch is never even reached by that test (the diagnostics-only "ts" simplification already
always allocates fresh). **So this hazard, while real, is NOT the cause of the already-observed
NaN** — it's a SEPARATE landmine that would only be hit if someone tried to differentiate the
untouched production `moments!.jl` directly (not the "ts" copy) with Enzyme.

**Follow-up** (`minimal_enzyme_nan_step5b_runtime_activity.jl`): Enzyme's own error message
suggests a workaround, `set_runtime_activity(Reverse)`. Tried it — the compile-time error goes
away, but **the resulting gradient is silently WRONG** (`[0.0]` vs. the correct `[5.69...]`), not
even NaN, just zero. This is worth recording precisely: the "recommended" Enzyme workaround for
this exact error pattern converts a loud crash into a silent wrong answer here — do NOT treat
`set_runtime_activity` as a safe fix for this pattern without independently validating the output
against ForwardDiff every time.

**Practical implication**: if `moments!.jl`'s `UPow_scratch`/`UσPow_scratch` optimization is ever
kept while attempting a REAL (non-"ts"-simplified) Enzyme port, it needs to be disabled for the
differentiated path (e.g. also gate it on a `Bool` flag passed by the caller, not just
`eltype(γ)===Float64`, since that check silently means something different under Enzyme than
under ForwardDiff).

## Step 6: the real `make_gravity_grad` formula (not the simplified reshape), real data

Steps 1-4 tested `withinTransform` on a bare `reshape(θ,D,D)` — but the REAL gravity gradient
(`../PsiObjectiveBundleImplicitMethodB_fullA.jl::make_gravity_grad`) builds `AodPow` through a
richer chain first: `Aod = Aod_θ .* cHat .* (((wHat.*τ)./(wHat[1,1].*τ[1,:]')).^(1/μ)) .*
(lambda./lambda[1,:]')`, `AodPow = (Aod./cHat).^(-μ)` — μ appears in TWO separate power
operations across several real-data array broadcasts before `withinTransform` is ever called.
`minimal_enzyme_nan_step6_real_gravity.jl` differentiates this EXACT formula (copy-pasted from
production) against REAL project data (`build_ad_context()`) and a REAL θ (frozen benchmark
point A), closing the gap between the clean synthetic tests (steps 1-4, no NaN) and the actual
gravity gradient object used in production.

**RESULT: NOT reproduced — and this is a significant correction to the prior session's claim.**

```
primal sumGrav at real theta (point A) = 5.115e-17  finite=true
ForwardDiff finite everywhere: true  nnan=0
Enzyme (plain Reverse): finite=true  nnan=0  relerr=3.12e-16
```

The REAL `make_gravity_grad` formula, on REAL data, at a REAL solved θ, matches ForwardDiff to
machine precision under Enzyme — **zero NaN**. This directly contradicts
`SESSION_SUMMARY_2026-07-12.md`'s claim that "the gravity moment alone" reproduces the NaN
pattern with zero branching. That claim cannot be reproduced against the actual
`make_gravity_grad` object with real data.

**Revised conclusion**: every piece of `newGravityMoment!`'s actual mathematical content —
`withinTransform`, the `sum(;dims=k)` reductions, the dead `doubleDiff` preamble, and now the
full real `Aod→AodPow→withinTransform→Σ` gravity-gradient chain itself — is now DIRECTLY
RULED OUT, tested against real production data, not just synthetic. The NaN observed in
`../ad_benchmark/test_enzyme_full.jl`/`test_mooncake2.jl` must therefore come from a part of the
FULL `envelope_scalar_div_ctx` pipeline that is NOT part of the gravity moment at all — most
likely **`hFunction!`/`hFunctionCounter!`** (the trade-share moments, which DO involve the
hard-max/`MinInd!` winner selection this session's prior work claimed was "unrelated") and/or
their interaction with the `gamma(μ*(1-σ)+1)` normalization or the N=8000-draw aggregation, none
of which this session isolated. **The "hard-max is unrelated" and "gravity alone reproduces it"
claims in the prior session's write-up should be treated as unverified pending a direct retest of
`hFunction!`/`hFunctionCounter!` in isolation** — the natural next bisection step, not reached
here.

## Decision (mega-prompt §14, unchanged from the prior session)

Regardless of step 4's outcome, per `../ad_benchmark/README.md` §7's timing analysis (direct
scalar ForwardDiff, Method B, is already ~1.7x the cost of a single inner KNITRO solve at D=10,
and reverse mode's only structural advantage — constant cost in `l` — does not matter once
forward-mode cost is this small relative to the inner solve) — **do not pursue reverse mode
further for production use**, even if step 4 (or a future deeper bisection) fully explains the
NaN. This session's contribution is narrowing the search space for anyone who DOES want to file
an upstream issue or debug it later, not fixing it now.
