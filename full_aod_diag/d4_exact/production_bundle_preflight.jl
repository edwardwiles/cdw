# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30, task §12: campaign preflight.
#
# Run BEFORE launching any production cell/campaign, in addition to (not instead of) whatever
# numerical 30-second smoke this repo's campaign scripts already run (e.g. smoke_delta1_flexcm.jl/
# _cmzc.jl/_frechet.jl -- unchanged, out of this task's scope). This file provides the structural
# half: for each requested family, build a real production context via the SAME
# prepare_production_run path the real drivers use, run the live assertion, save the backend
# manifest, and refuse to launch if the invariant fails or any dense construction happened.
#
# This is deliberately NOT wired into the 3 real drivers' own call chain (they already call
# prepare_production_run/assert_production_operator_bundle! internally, at real D=20/W=100,000
# scale, on every real call -- duplicating that here would just re-run the same expensive
# real-data setup for no new information). This file is for a CAMPAIGN LAUNCHER that wants to
# validate several families' configuration BEFORE committing to the real driver calls (task §12's
# own framing: "prepare every requested family" as a distinct step before "launching" them) --
# typically at D=4 for speed, since the invariant itself does not depend on data scale (confirmed:
# it is a structural/type check, not a numerical one).
# ================================================================================================
isdefined(Main, :prepare_production_run) || include(joinpath(@__DIR__, "production_bundle_api.jl"))

"""
    campaign_preflight(families::Vector{Symbol}, build_inner_for::Function; manifest_dir::AbstractString) -> Bool

`build_inner_for(family::Symbol) -> Function` returns the zero-arg `build_inner` closure for that
family (i.e. the caller supplies how to build each family's context -- this function does not know
family-specific kwargs any more than `prepare_production_run` itself does).

Returns `true` iff EVERY requested family passes `bundle_invariant_pass` and the cumulative
`DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[]` count is still 0 after preparing all of them. Writes one
`<family>_preflight_manifest.json` per family into `manifest_dir`, always (pass or fail), so a
failed preflight leaves a paper trail of exactly what was seen.

Per task §12: refuses to launch (`bundle_invariant_pass != true` OR
`DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] != 0` for ANY family) rather than launching a subset.
"""
function campaign_preflight(families::Vector{Symbol}, build_inner_for::Function; manifest_dir::AbstractString)
    isdir(manifest_dir) || mkpath(manifest_dir)
    ok = true
    for family in families
        println("[preflight] preparing ", family, " ...")
        try
            prepared = prepare_production_run(family, "campaign_preflight", build_inner_for(family))
            write_backend_manifest_atomic(prepared.manifest, joinpath(manifest_dir, "$(family)_preflight_manifest.json"))
            if !prepared.manifest.bundle_invariant_pass
                println("[preflight] REFUSE: ", family, " bundle_invariant_pass=false")
                ok = false
            elseif DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] != 0
                println("[preflight] REFUSE: ", family, " dense-reference construction count=",
                         DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[], " (expected 0)")
                ok = false
            else
                println("[preflight] OK: ", family, " -> ", prepared.manifest.structural.bundle_type)
            end
        catch e
            println("[preflight] REFUSE: ", family, " raised: ", sprint(showerror, e))
            ok = false
        end
    end
    println("[preflight] ", ok ? "ALL FAMILIES PASS -- launch permitted" : "REFUSED -- do not launch",
            " (dense-reference construction count = ", DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[], ")")
    return ok
end
