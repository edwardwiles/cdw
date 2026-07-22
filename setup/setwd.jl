# Closure task Phase 3E (independent assessment F9): the remediation task's first pass only
# replaced a raw SystemError with a clearer error message -- callers still needed one of a
# fixed set of hard-coded per-developer absolute paths to exist on the current host, so
# production/tests still could not run from an arbitrary checkout path. Not on the active
# full_aod_diag/d4_exact test path today (confirmed by grep: no d4_exact_setup()/master_setup()
# call chain reaches setwd() anywhere in full_aod_diag/d4_exact/*.jl), but the task requires a
# real portability fix regardless, not just a clearer error.
#
# Default behavior now: derive the repository root from @__DIR__ (this file lives at
# <repo_root>/setup/setwd.jl) and cd() there -- works from any checkout path on any host, no
# hard-coded directory required. The historical per-(server,user) absolute-path table is kept,
# opt-in only, behind GRAVITY_ROBUSTNESS_LEGACY_SETWD=1, for any interactive user who still
# wants that exact legacy convenience on one of the original dev hosts; it is never reached
# unless that env var is explicitly set, so an unrelated host/checkout is unaffected by it.
function setwd(server = nothing, user = nothing)
    if get(ENV, "GRAVITY_ROBUSTNESS_LEGACY_SETWD", "") == "1"
        (server === nothing || user === nothing) && error(
            "setwd(): GRAVITY_ROBUSTNESS_LEGACY_SETWD=1 requires explicit (server, user) args.")
        return _setwd_legacy(server, user)
    end
    target = normpath(joinpath(@__DIR__, ".."))
    isdir(target) || error("setwd(): derived repository root does not exist: $target -- " *
        "this file (setup/setwd.jl) is expected to live at <repo_root>/setup/setwd.jl.")
    cd(target)
    global folderData = target
    return nothing
end

"Historical per-developer hard-coded paths -- opt-in only via GRAVITY_ROBUSTNESS_LEGACY_SETWD=1, see setwd()'s own docstring above."
function _setwd_legacy(server, user)
    target = nothing
    if server == 1 # if working on the econ servers
        if user == 1
            target = raw"/bbkinghome/mhansari/Robustness"
            data = target
        elseif user == 2
            # Ed's modular working copy — run from here so the .opt files (loaded by
            # bare filename at runtime) and output CSVs resolve to this repo root.
            target = raw"/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
            data = target
        end
    elseif server == 0 # if working on laptop
        if user == 1
            target = raw"C:/2. MIT/Model Robustness"
            data = "C:/2. MIT/Model Robustness/Data"
        elseif user == 2
            target = raw"C:/Users/edwar/Dropbox (MIT)/Gravity robustness/Analysis/Julia Consolidated"
            data = "C:/Users/edwar/Dropbox (MIT)/Gravity robustness/Analysis/WIOD Data"
        end
    end
    target === nothing && error("_setwd_legacy(server=$server, user=$user): no path configured for this " *
        "(server, user) combination -- add one below rather than silently doing nothing.")
    isdir(target) || error("_setwd_legacy(server=$server, user=$user): target directory does not exist " *
        "on this host: $target -- this is a per-developer hard-coded path; add/update the entry " *
        "for your own host instead of running this unmodified elsewhere.")
    cd(target)
    global folderData = data
    return nothing
end