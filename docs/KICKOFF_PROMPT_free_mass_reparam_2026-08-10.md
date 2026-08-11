Work in `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10` (branch
`integration/pairwise-quantile-outer-loop-2026-08-10`; start a new branch off it).

Your task is specified in full in
`docs/PAIRWISE_QUANTILE_FREE_MASS_REPARAMETERIZATION_HANDOVER_2026-08-10.md` — read it first, and
read `docs/PAIRWISE_QUANTILE_OUTER_LOOP_STATUS_2026-08-10.md` before touching code (it records two
bugs that must not be reintroduced, and one test whose convention must not be "restored").

Short version: the pairwise-independence restriction currently optimizes over the quantile CUTOFFS,
which sit inside indicator functions, so Delta* is a step function of every outer coordinate and the
gradient needs bandwidth/secant machinery that can only be validated to ~20%. Reparameterize it:
**fix the cutoffs once per campaign and make the bin MASSES mu_{o,a} the free outer parameters**, so
the moments become `1{b_o=a} - mu_{o,a}` and `1{b_o=a,b_p=b} - mu_{o,a}*mu_{p,b}`. Delta* is then
smooth and the outer gradient has an exact closed form that is structurally identical to origin-ZC's
existing, production-validated `d_delta_dual_d_eta_origin_vec` — mirror that, don't invent one.

Three things worth knowing before you plan:

1. The inner-solve edits are far smaller than they look. The indicator machinery and the T1-T4
   Hessian tables (~78% of wall-clock) are completely unchanged — only the per-row centering
   constant changes, in `forward!`, `transpose!`, `center_and_scale_..._hessian!`, the cross-Hessian
   block, and the HVP path. See handover section 2.
2. This buys exactness, NOT speed. Inner-solve cost is unchanged (measured, real D=20/W=100k:
   L=5 = 39.9 s/solve, L=10 = 1084.6 s/solve, both VerifiedSolved). L=10 outer-search performance is
   a separate task (`pairwise_quantile_hvp.jl`).
3. Because the objective is now smooth, a plain small-h reoptimized FD is a valid ground truth and
   you should expect ~1e-5 relative agreement or better, like origin-ZC's gate. A few percent is NOT
   good enough here — that would mean something is wrong.

Two modelling choices are yours to make explicit and required-with-no-default (CLAUDE.md's rule):
where the cutoffs are fixed (Fréchet quantiles was the user's suggestion), and how mu is
parameterized on the simplex (a cumulative/stick-breaking transform reuses the most existing
machinery and matches the math note, which is already written in cumulative form).

Do the version-A/version-B equivalence anchor early (handover section 4.3) — pinning mu = 1/L with
cutoffs at the empirical quantiles must reproduce version A's Delta* to machine precision, and there
are recorded reference values to check against. It is nearly free and will catch any centering or
indexing error immediately.

Do NOT touch `protocols/paper_upper_v1.toml` — it is frozen and a live campaign is running on this
host (`screen -S paperupper_resume`). Launch long jobs under `screen`, and check `ps` + `tail`
within ~60 s of starting one.
