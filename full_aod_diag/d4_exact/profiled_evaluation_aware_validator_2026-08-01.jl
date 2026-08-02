# ============================================================================
# Production outer bridge task (2026-08-01), §4: an evaluation-aware layout
# validator.
#
# `validate_family_layout_contract(fctx)` (profiled_outer_gradient_layout_
# contract_2026-08-01.jl) is STRUCTURAL ONLY -- it never looks at a solved
# dual vector, so it cannot catch a family adapter whose ranges are
# internally self-consistent but do not actually match what got solved (e.g.
# an economic_dual_range built for the WRONG family's beta length, or a
# restriction range that silently omits the family's last restriction
# column). That is exactly the gap task §4 asks to close: "the structural
# mock contract passes" must not be enough to let a restricted adapter run.
# ADDITIVE ONLY -- does not edit validate_family_layout_contract or anything
# else in the contract file.
# ============================================================================

isdefined(Main, :validate_family_layout_contract) ||
    error("profiled_evaluation_aware_validator_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")
isdefined(Main, :stable_layout_digest) ||
    error("profiled_evaluation_aware_validator_2026-08-01.jl requires profiled_stable_layout_digest_2026-08-01.jl to be included first.")

"""
    validate_family_layout_against_evaluation(fctx, ev; require_manifest_digest=false,
        restriction_outer_param_names=Symbol[]) -> NamedTuple

Task §4. Runs `runtime_structural_check(fctx)` first (throws on any purely
structural violation, unchanged), THEN checks every solved-vector fact the
structural check cannot see, against `ev` (an `evaluate_profiled_point`-shaped
NamedTuple: `ev.result.beta`, `ev.result.zeta`, `ev.result.x_sol`). Throws
(never warns, never silently coerces) on:

  - `economic_dual_range(fctx)` out of bounds of `ev.result.beta`;
  - any `restriction_dual_ranges(fctx)` entry out of bounds of `ev.result.beta`;
  - the economic range plus every restriction range NOT partitioning
    `1:length(ev.result.beta)` EXACTLY -- no gap, no overlap, no leftover
    trailing/leading index (task: "no unexplained trailing or missing dual
    entries", "ranges cover exactly the intended solved dual components");
  - `ev.result.zeta != ev.result.x_sol[1]` (zeta-convention drift: this
    branch's whole q=-zeta-econ-restriction formula assumes dual index 1 is
    always zeta and `beta=x[2:end]` starts immediately after it -- see
    `NORMALIZATION_CONVENTION_TAG`);
  - `family_kind(fctx)` disagreeing with `ev`'s own family tag, WHEN `ev`
    exposes one (`ev.family_kind`, a field only a real restricted-family
    evaluator would set -- see the narrow-hook note below);
  - `stable_layout_digest(fctx)` disagreeing with `ev`'s own manifest digest
    (`ev.layout_digest_manifest`), WHEN `require_manifest_digest=true` (see
    below).

**Narrow hook for the inner branch** (task §4/§8's own "if a missing inner
API blocks live integration, ... provide a narrow typed hook for the inner
branch to expose later"): the CURRENT `evaluate_profiled_point` (this
branch's own, family-agnostic) does not yet stamp `ev` with either
`family_kind` or `layout_digest_manifest` -- no live restricted-family
evaluator exists yet to populate them meaningfully. Rather than either (a)
skip these two checks silently, or (b) throw for every caller including the
already-working unrestricted regression gates, this function:
  - runs the family-kind check ONLY if `haskey(ev, :family_kind)` is true;
  - runs the manifest-digest check ONLY if `require_manifest_digest=true`
    (default `false`, so existing unrestricted call sites are unaffected)
    AND throws immediately if `require_manifest_digest=true` but `ev` does
    not expose `:layout_digest_manifest` at all -- a caller that explicitly
    asks for this guarantee must get a real answer, not a silent skip.
Once a real restricted-family evaluator lands and stamps both fields, no
caller of this function needs to change -- only the `require_manifest_digest`
default a production runner passes should flip to `true`.
"""
function validate_family_layout_against_evaluation(fctx, ev; require_manifest_digest::Bool = false,
        restriction_outer_param_names::Vector{Symbol} = Symbol[])
    v = runtime_structural_check(fctx)
    erange = v.economic_dual_range
    rranges = v.restriction_dual_ranges

    beta = ev.result.beta
    n_beta = length(beta)

    (1 <= first(erange) && last(erange) <= n_beta) ||
        error("validate_family_layout_against_evaluation($(family_kind(fctx))): economic_dual_range=$erange out of bounds of ev.result.beta (length $n_beta)")
    for r in rranges
        (1 <= first(r.range) && last(r.range) <= n_beta) ||
            error("validate_family_layout_against_evaluation($(family_kind(fctx))): restriction range :$(r.name)=$(r.range) out of bounds of ev.result.beta (length $n_beta)")
    end

    all_ranges = vcat([erange], [r.range for r in rranges])
    covered = falses(n_beta)
    for r in all_ranges, i in r
        covered[i] && error("validate_family_layout_against_evaluation($(family_kind(fctx))): dual index $i covered by more than one range (economic+restriction ranges overlap) -- $(all_ranges)")
        covered[i] = true
    end
    all(covered) ||
        error("validate_family_layout_against_evaluation($(family_kind(fctx))): economic+restriction ranges do not cover ev.result.beta exactly -- missing indices $(findall(!, covered)) out of 1:$n_beta (unexplained trailing/missing dual entries)")

    haskey(ev.result, :x_sol) && haskey(ev.result, :zeta) &&
        (ev.result.zeta == ev.result.x_sol[1] ||
         error("validate_family_layout_against_evaluation($(family_kind(fctx))): zeta convention violated -- ev.result.zeta=$(ev.result.zeta) != ev.result.x_sol[1]=$(ev.result.x_sol[1])"))

    if haskey(ev, :family_kind)
        family_kind(fctx) == ev.family_kind ||
            error("validate_family_layout_against_evaluation: family_kind(fctx)=$(family_kind(fctx)) != ev.family_kind=$(ev.family_kind)")
    end

    digest = stable_layout_digest(fctx; restriction_outer_param_names = restriction_outer_param_names)
    if require_manifest_digest
        haskey(ev, :layout_digest_manifest) ||
            error("validate_family_layout_against_evaluation($(family_kind(fctx))): require_manifest_digest=true but ev does not expose :layout_digest_manifest -- refusing to silently skip the check")
        digest == ev.layout_digest_manifest ||
            error("validate_family_layout_against_evaluation($(family_kind(fctx))): stable_layout_digest(fctx)=$digest != ev.layout_digest_manifest=$(ev.layout_digest_manifest)")
    end

    return (structural = v, economic_dual_range = erange, restriction_dual_ranges = rranges,
        n_beta = n_beta, stable_layout_digest = digest)
end
