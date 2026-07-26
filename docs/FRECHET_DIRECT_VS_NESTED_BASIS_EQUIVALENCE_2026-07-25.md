# Direct vs nested basis equivalence — 2026-07-25/26 (Part VI §16)

Full proof in `FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md` §3. This document is the
numerical verification record, at D=4 (`test_frechet_cm_level_basis_d4.jl`, 22/22 PASS) and real
D=20/W=80,000 (`test_frechet_d20_gates.jl` Part 1, 12/12 PASS).

## What was checked, both D

For `M = [C(R)  u]` (`C` = the anchored omit-reference contrast, optionally rotated by the
orthonormal `R`; `u = 1_D/√D`):

| Check | D=4 | D=20 (real draws) |
|---|---|---|
| `M` is `D×D` | ✓ | ✓ |
| `rank(M) = D` (full rank) | ✓ | ✓ |
| `cond(M)`, `:anchored` | `2.0000` (`=√4`) | `4.4721` (`=√20`) |
| `cond(M)`, `:orthonormal` | `1.0000` (exact) | `1.0000` (exact) |
| `max\|C(R)'q_l − CM_construction\|` | `0.0` / `1.4e-16` | `0.0` / `1.0e-15` |
| `max\|u'q_l − LEVEL_construction\|` | `1.1e-16` | `8.9e-16` |
| round-trip `q_l` reconstruction via `M^{-1}` | `2.8e-17` / `2.2e-16` | `6.1e-16` / `1.8e-15` |
| bin-lookup (Architecture-B) vs dense (Architecture-A) LEVEL | `0.0` exact | (covered separately, Part II wiring test: `1.8e-15`) |

`cond(M) = √D` for `:anchored` is a clean closed form (not asymptotic/approximate — verified exact
at both D tested), confirming the math doc's claim that both contrast modes stay well-conditioned
at any realistic D.

D=20 note: the `q_l`/`f_l(ω)` reconstruction loop is `O(W·D·L)`; the D=20 test spot-checks 5 of 10
thresholds (not all 10) to keep runtime bounded — this is not a reduced-confidence check, since
every threshold uses byte-identical machinery (no threshold-specific code path exists to miss).

## Verdict

`FRECHET_FORMULATION = cm_plus_common_level`, basis equivalence to direct country-by-country fixed
Fréchet **proven algebraically** (math doc §3) and **verified numerically to machine precision** at
both D=4 and real D=20 production data.
