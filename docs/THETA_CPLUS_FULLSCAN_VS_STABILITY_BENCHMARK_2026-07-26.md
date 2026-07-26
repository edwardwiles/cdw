# Fused full-scan vs winner-stability certificate — scope note, 2026-07-26

The original task brief's §4/§5/§9 asked for two candidate implementations to be benchmarked
against each other: a "fused full-scan" (exact re-scan of all D origins at both perturbed theta
values) and a "winner-stability certificate" (an exact directional-crossing-radius calculation
that lets a base winner be reused, with a selective rescan only when the certificate fails).

**Mid-task, the user sent an explicit addendum removing the stability-certificate path from this
task's scope entirely**: "The production theta derivative must remain an exact fixed-dual central
secant that recomputes the hard winner at both perturbed theta values... Remove the proposed
optional analytic stable-winner backend from this task." Per that instruction, only the fused
full-scan path was implemented (`theta_cplus_secant`, via `build_compressed_factual!` — an
unconditional, exact re-scan every call) — there is no second implementation to benchmark against
in this deliverable.

The winner-stability crossing-radius formula itself is still derived and documented, as a
reference/diagnostic (`docs/THETA_CPLUS_MATHEMATICAL_DERIVATION_2026-07-26.md` §6), in case a
future session wants it — e.g. to report how close to a winner flip a given theta probe was — but
it is not wired into any code path, not benchmarked, and explicitly not a candidate default per
the addendum.

**Given the fused full-scan alone already clears every one of the task's §11 performance targets**
(6.43x theta-block speedup against a 5x bar, 99.66% allocation reduction against a 90% bar — see
`THETA_CPLUS_ALLOCATION_PROFILE_2026-07-26.md`), there is no remaining performance case for the
stability-certificate path within this task's scope even setting the addendum aside: the
performance problem this task set out to fix is resolved by the full-scan alone, using
already-existing, already-validated production code, without the added complexity and
re-validation burden a hand-rolled selective-rescan implementation would introduce.
