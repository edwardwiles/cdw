#!/usr/bin/env julia
# Phase 14 static repository guard (docs/melitz_legacy_H_removal_audit_2026-07-31.md).
#
# Scans every production Melitz source file for the forbidden legacy-dense symbols
# (select_G_from_H, direct PsiObjectiveBundleDelta/Implicit construction, obj.H views,
# obj.moments! calls) and fails unless the ENCLOSING FUNCTION is on the explicit allowlist
# below -- i.e. unless the match is inside one of the already-audited, deliberately-generic
# duck-typed dense-fallback methods (the "other half" of MelitzCCBundle's own dispatch pair,
# required so build_melitz_psi_bundle(...; backend=:dense_reference) keeps working as an
# explicit diagnostic choice) or an explicitly-disclosed non-production function
# (predictor_corrector.jl's fresh_gradient).
#
# A NEW match whose enclosing function is not on this list means a genuinely new dense
# call site was added somewhere production code can reach -- the guard fails loudly rather
# than silently passing. Extending the allowlist should require the same scrutiny as this
# session's own audit (docs/melitz_legacy_H_removal_audit_2026-07-31.md Phase 0).
#
# Usage: julia scripts/melitz_static_dense_fallback_guard_2026-07-31.jl
# Exit code 0 = pass, 1 = fail (unallowlisted forbidden-symbol match found).

const MELITZ_DIR = joinpath(dirname(@__DIR__), "src", "melitz")

# (relative filename, enclosing-function-name) pairs that are KNOWN, AUDITED, deliberate
# dense-fallback / diagnostic sites (2026-07-31 audit). See the audit doc's Phase 0 table for
# the evidence trail on each entry.
const ALLOWLIST = Set([
    # cc_bundle.jl: the generic (untyped `obj`) half of every MelitzCCBundle dispatch pair --
    # required for backend=:dense_reference (PsiObjectiveBundleDelta/Implicit) to keep
    # working at all; each has a MORE SPECIFIC `obj::MelitzCCBundle` sibling method
    # immediately below it in the same file that is dispatched to instead for the real
    # production bundle type.
    ("cc_bundle.jl", "melitz_bundle_prepare_at_theta!"),
    ("cc_bundle.jl", "melitz_bundle_inner_solve!"),
    ("cc_bundle.jl", "melitz_bundle_current_G"),
    ("cc_bundle.jl", "melitz_heavy_snapshot"),
    ("cc_bundle.jl", "melitz_heavy_restore!"),
    ("cc_bundle.jl", "melitz_heavy_recompute"),
    ("cc_bundle.jl", "melitz_bundle_dense_G_at_theta"),
    ("cc_bundle.jl", "melitz_dense_G_from_operator"),   # explicit diagnostic escape hatch,
        # never called from a production-fast hot path (own docstring); materializes G FROM
        # the operator (diagnostic direction), not a construction of legacy storage.
    # delta_star.jl / finite_delta_outer.jl: the explicit backend=:dense_reference construction
    # branch, gated behind an explicit, non-default kwarg (never reached by a caller that
    # doesn't deliberately request it) -- this is the "kept fully intact for diagnostics"
    # legacy path the governing prompt's own Phase 6 permits.
    ("delta_star.jl", "build_melitz_psi_bundle"),
    ("finite_delta_outer.jl", "build_melitz_implicit_bundle"),
    ("pareto_calibration.jl", "build_melitz_psi_bundle_from_calibration"),
    # predictor_corrector.jl: explicitly disclosed as NOT part of the main production outer
    # loop (own header comment), pinned to backend=:dense_reference explicitly, zero
    # call sites anywhere else in src/melitz or scripts/ (grep-confirmed 2026-07-31).
    ("predictor_corrector.jl", "fresh_gradient"),
    ("predictor_corrector.jl", "melitz_predictor_corrector_continuation"),
    # Generic (untyped `obj`) halves of three more MelitzCCBundle dispatch pairs, each
    # confirmed (2026-07-31 audit) to have a more-specific `obj::MelitzCCBundle` sibling
    # defined in cc_bundle.jl that Julia dispatches to instead for the real production bundle:
    # melitz_recover_lfd_from_solution (cc_bundle.jl:799), _base_arg0! (cc_bundle.jl:580),
    # fstar_equal_weight_moments (cc_bundle.jl:1169).
    ("delta_star.jl", "melitz_recover_lfd_from_solution"),
    ("direct_gradient.jl", "_base_arg0!"),
    ("fstar_direct.jl", "fstar_equal_weight_moments"),
])

const FORBIDDEN_PATTERNS = [
    r"CS\.select_G_from_H\(",
    r"\bPsiObjectiveBundleDelta\(",
    r"\bPsiObjectiveBundleImplicit\(",
    r"obj\.moments!\(",
    r"obj_inner\.moments!\(",
    r"obj_like\.moments!\(",
    r"@view\(obj\.H",
    r"@view\(obj_inner\.H",
]

function enclosing_function(lines::Vector{String}, lineno::Int)
    for i in lineno:-1:1
        m = match(r"^function\s+([A-Za-z_!][A-Za-z0-9_!]*)", lines[i])
        m !== nothing && return m.captures[1]
    end
    return nothing
end

function main()
    n_checked = 0
    n_violations = 0
    for fname in sort(readdir(MELITZ_DIR))
        endswith(fname, ".jl") || continue
        path = joinpath(MELITZ_DIR, fname)
        lines = readlines(path)
        for (i, line) in enumerate(lines)
            # skip pure-comment lines (this guard checks live code, not docstrings/comments
            # discussing these symbols, which are extremely common in this heavily-annotated
            # codebase)
            stripped = lstrip(line)
            startswith(stripped, "#") && continue
            for pat in FORBIDDEN_PATTERNS
                if occursin(pat, line)
                    n_checked += 1
                    fn = enclosing_function(lines, i)
                    key = (fname, fn)
                    if key in ALLOWLIST
                        # allowed, known site
                    else
                        n_violations += 1
                        println("VIOLATION: $fname:$i in function $(fn === nothing ? "<top-level>" : fn): ", strip(line))
                    end
                end
            end
        end
    end
    println("\nStatic dense-fallback guard: $n_checked forbidden-symbol occurrences scanned, $n_violations unallowlisted.")
    if n_violations > 0
        println("FAIL")
        exit(1)
    else
        println("PASS")
        exit(0)
    end
end

main()
