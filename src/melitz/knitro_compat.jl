# Melitz-owned KNITRO convenience wrappers.
#
# `cc_algo/knitro_compat.jl` extends the KNITRO module itself (type piracy, deliberate there)
# with 2-arg convenience forms of `KN_add_vars`/`KN_add_cons`/`KN_get_int_param` that the
# installed `KNITRO.jl` package does NOT provide natively (only the 3-arg core forms do,
# confirmed live: `methods(KNITRO.KN_add_vars)` shows only `(kc, nV, indexVars)`). Melitz's own
# KNITRO drivers (`cc_bundle.jl`, `matrix_free_dual_solve.jl`, `nuisance_profile.jl`,
# `finite_delta_outer.jl`) called the `cc_algo`-monkey-patched 2-arg forms directly -- an
# invisible dependency on `cc_algo` being loaded first, confirmed live by
# `test_melitz_standalone_no_cc_algo.jl`: `MethodError: no method matching KN_add_vars(::KNITRO.Model,
# ::Int64)` when only Melitz's own include path + KNITRO are loaded.
#
# These Melitz-named wrappers close that gap without extending the KNITRO module from Melitz
# code (per the governing prompt's own preference: a local, Melitz-owned name over another
# global monkey-patch) and without touching `cc_algo` at all.

"""
    melitz_kn_add_vars!(kc, nV::Integer) -> Vector{Cint}

Allocates `nV` new KNITRO variables and returns their indices -- the Melitz-owned equivalent
of `cc_algo/knitro_compat.jl`'s `KN_add_vars(kc, nV)` 2-arg convenience form, calling the
same underlying 3-arg `KNITRO.KN_add_vars(kc, nV, indexVars)` core API.
"""
function melitz_kn_add_vars!(kc, nV::Integer)
    idx = zeros(Cint, nV)
    KNITRO.KN_add_vars(kc, Cint(nV), idx)
    return idx
end

"""
    melitz_kn_add_cons!(kc, nC::Integer) -> Vector{Cint}

Allocates `nC` new KNITRO constraints and returns their indices -- the Melitz-owned equivalent
of `cc_algo/knitro_compat.jl`'s `KN_add_cons(kc, nC)` 2-arg convenience form.
"""
function melitz_kn_add_cons!(kc, nC::Integer)
    idx = zeros(Cint, nC)
    KNITRO.KN_add_cons(kc, Cint(nC), idx)
    return idx
end

"""
    melitz_kn_get_int_param(kc, name::AbstractString) -> Cint

String-named integer KNITRO parameter getter -- the Melitz-owned equivalent of
`cc_algo/knitro_compat.jl`'s `KN_get_int_param(kc, name)` 2-arg convenience form.
"""
function melitz_kn_get_int_param(kc, name::AbstractString)
    v = Ref{Cint}(0)
    KNITRO.KN_get_int_param_by_name(kc, name, v)
    return v[]
end
