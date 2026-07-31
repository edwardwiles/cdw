# Fresh campaign cache namespace

**Core fix (this repo, `full_aod_diag/d4_exact/oracle.jl`):** `context_fingerprint(ctx)` did not
hash `ctx.σ` (sigma). Sigma enters the inner solve's CES/price-index formulas directly, not merely
as one entry of the outer theta vector a fixed `x_free` would otherwise pin down -- two contexts
built identically except for sigma could give different inner-solve answers at the same `x_free`,
but would alias to the same cache key/fingerprint. This is exactly the risk the campaign brief's
"do not reuse caches built under sigma=2.5" requirement warns about. Fixed by adding `ctx.σ` to
the hashed buffer and bumping `CONTEXT_FINGERPRINT_SCHEMA` (2->3), so any pre-existing on-disk
cache entries (computed under the old, sigma-blind schema) are automatically distinguished/
invalidated rather than silently reused. Verified live: sigma=2.5 and sigma=3.0 contexts now
fingerprint differently; identical-sigma contexts still fingerprint identically (no
over-fragmentation).

**Driver-level requirement (Phase D, not yet built):** the brief's full list -- data checksum,
sigma, W, Sobol seed/checksum, family, L, K, omit-ROW setting, production SHA -- is broader than
what belongs inside `context_fingerprint` (a MATHEMATICAL-context identity function). `family`/`L`/
`K` are already partially covered for CM configs via `cm_cache_key`/`ctx.cm`, but origin_zc/meanzc
K_mean/K_pair are not yet folded into any single hash function, and `production SHA` is a
code-version identifier, not a mathematical-context one -- it doesn't belong inside
`context_fingerprint` at all.

The simplest, most robust way to guarantee no collision with any prior campaign's cache -- given
the fingerprint gap just fixed still only covers what a single ctx object knows about itself, not
family/L/K/production-SHA -- is to use a **fresh, campaign-specific cache ROOT DIRECTORY**,
namespaced by a descriptive string built from {data checksum, sigma, W, Sobol seed/checksum,
production SHA}, with family/L/K as subdirectory components (matching how the existing campaign
runners already structure per-cell checkpoint directories). A fresh directory categorically cannot
collide with any prior campaign's cache regardless of any hash-function gap that might still exist
in some other file. This is deferred to Phase D (building the actual campaign driver), where the
cache root path will be set explicitly in `campaign_config.json`.
