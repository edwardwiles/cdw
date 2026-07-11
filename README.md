# Trade Model Robustness (modular)

Modular working copy implementing the CDW / Christensen–Connault robustness method.
**Main entry point: `master.jl`** (`setup → prestep → prepare_cc → moments → cc_algo → lfd`).
The legacy monolith has been moved to `legacy/`.

**Status: working.** The committed 4-country simulated example produces the distribution-agnostic
gains-from-trade bounds — at δ=1, κ ∈ **[0.0074, 0.2195]** around the gravity point estimate 0.0686
(the shape of CDW Figure 2). It's a fast example (W=8000, ~4 min); see SETUP for paper-scale settings.

Requires KNITRO, which only licenses on **`demand.mit.edu`**. To run:

```bash
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.juliaup/bin:$PATH"
julia --project=. master.jl
```

See **[SETUP_AND_FINDINGS.md](SETUP_AND_FINDINGS.md)** for the method↔code map, the KNITRO
compatibility shim, the derivative/Jacobian structure, what was fixed to make it run, the cruft
that was removed, and known/deferred issues (e.g. the common-marginals restriction bug).
