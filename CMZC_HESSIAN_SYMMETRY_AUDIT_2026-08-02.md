# CM+ZC Hessian symmetry audit — 2026-08-02

Dedicated gate: `full_aod_diag/d4_exact/test_cmzc_hessian_symmetry_audit_2026-08-02.jl`. Isolates and
proves the widened-core `H_EM` mirror fix (`cm_hessian_architectures.jl::_fill_cm_HEE!`, mirror line
`HEE[ncore+1:NCORE,1:ncore] .= transpose(HEM)`, ~line 1063) at 4 random points (not just calibration),
against two independent references, plus origin-ZC's analogous `H_EZ` block as a structural control.

## Results (fresh Julia process, D4, W=8000): ALL PASS

**Part A — CM+ZC widened core (ncore_core=14, NCORE=18, ncm=9, n_total=27), 4 random points:**
- Check 1 — `H_EM` mirror vs `transpose(H_EM)`, read directly off `Hfull` **before** the
  symmetrize-by-averaging pack step: bit-exact (`max|Δ|=0.000e+00`) at all 4 points.
- Check 2 — complete pre-pack `Hfull` symmetric: `max|Hfull-Hfull'|` between `0.0` and `1.11e-16`
  (machine epsilon) at all 4 points.
- Check 3 — packed production Hessian vs ForwardDiff and vs independent dense-truth
  `G'·Diagonal(S)·G`: `max|Δ|` 5.9e-15 to 2.2e-14 across points (both references agree with each
  other to the same precision — sanity-cross-checked). Block-by-block (EE/EM/MM/EC/MC/CC) all clean;
  critically **`H_EM` production/ForwardDiff ratio = 1.000000 (min=max) at every point** — not the
  historical ~0.5 halving signature.

**Part B — origin-ZC control (H_EZ, no widened-core mirror mechanism), 4 random points:**
- Source-confirmed structurally: `archA_partitioned_hess_cb_builder`'s packer reads only `i<=j`
  entries with no averaging step at all (a prior mirror-write was found "provably dead computation"
  and removed 2026-07-28) — this bug class cannot occur here, not merely untriggered.
- Same ForwardDiff/dense-truth cross-check regardless: clean at all 4 points, `H_EZ` ratio 1.0 exactly.

No residual asymmetry or precision loss found anywhere. No fix was needed beyond the one already
present on this branch (`H_EM` mirror line) — this gate is confirmatory, run explicitly as its own
dedicated check per the task's requirement rather than relying on incidental coverage from the
existing autodiff/outer-gradient gates.

Log: see git history / rerun the test file directly for full output (4 points × 2 families, ~90s
fresh-process run).
