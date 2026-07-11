# --- KNITRO.jl compatibility shim ---------------------------------------------
# The CC inner/outer loop code was written for the KNITRO.jl 0.13/0.14 high-level
# API. KNITRO.jl v1.x kept almost all of that API (callbacks, KN_set_con_eqbnds,
# KN_set_con_upbnd, the *_all bound setters, KN_get_solution, KN_solve, ...) but
# dropped a few convenience wrappers. These definitions restore exactly the ones
# the loop code relies on, mapping them onto the v1.2.1 low-level bindings.
# (Deliberate, narrow type piracy on KNITRO's own functions.)

# KN_add_vars(kc, nV) / KN_add_cons(kc, nC): allocate the index buffer, call the
# 3-arg C form, and return the vector of new indices.
function KNITRO.KN_add_vars(kc, nV::Integer)
    idx = zeros(Cint, nV)
    KNITRO.KN_add_vars(kc, Cint(nV), idx)
    return idx
end

function KNITRO.KN_add_cons(kc, nC::Integer)
    idx = zeros(Cint, nC)
    KNITRO.KN_add_cons(kc, Cint(nC), idx)
    return idx
end

# KN_get_int_param(kc, "name"): string-named integer parameter getter returning value.
function KNITRO.KN_get_int_param(kc, name::AbstractString)
    v = Ref{Cint}(0)
    KNITRO.KN_get_int_param_by_name(kc, name, v)
    return v[]
end

# 2-arg (all-variable) bound setters used interchangeably with the *_all forms.
KNITRO.KN_set_var_lobnds(kc, xLoBnds::AbstractVector) = KNITRO.KN_set_var_lobnds_all(kc, xLoBnds)
KNITRO.KN_set_var_upbnds(kc, xUpBnds::AbstractVector) = KNITRO.KN_set_var_upbnds_all(kc, xUpBnds)
