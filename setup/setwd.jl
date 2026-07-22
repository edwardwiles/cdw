# Remediation task Part E (finding F9): each branch below hard-codes an absolute per-developer
# path. `cd()` to a path that doesn't exist on the current host throws an opaque `SystemError`
# before any model code runs -- this file previously worked on the export/assessment host only
# because a stale directory happened to still exist there. This is a personal multi-developer
# environment bootstrap (legacy/interactive convenience, not on the active
# full_aod_diag/d4_exact test path -- confirmed by grep: no d4_exact_setup()/master_setup() call
# chain reaches this file on production/fullA-exact), so a full redesign is out of scope; the fix
# here is a clear, actionable error instead of a raw SystemError, plus an explicit opt-in guard
# so calling this on an unexpected host fails loudly rather than silently `cd`-ing to the wrong
# place (or crashing uninformatively).
function setwd(server, user)
    # sets working directories
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
    target === nothing && error("setwd(server=$server, user=$user): no path configured for this " *
        "(server, user) combination -- add one below rather than silently doing nothing.")
    isdir(target) || error("setwd(server=$server, user=$user): target directory does not exist " *
        "on this host: $target -- this is a per-developer hard-coded path (task Part E, F9); " *
        "add/update the entry for your own host instead of running this unmodified elsewhere.")
    cd(target)
    global folderData = data
    return nothing
end