# CDW — gravity pipeline

Builds the three inputs a quantitative trade (gravity) model needs, for a set of
**19 named economies + ROW**, for a given year:

| Object | File | Meaning |
|---|---|---|
| `pi`  | `output/pi_<year>.csv`  | 20×20 import-share matrix, `pi[o,d] = X_{o,d} / Σ_o X_{o,d}` (incl. domestic) |
| `tau` | `output/tau_<year>.csv` | 20×20 bilateral tariff matrix, `tau[o,d] = 1 + (applied tariff d levies on o)` |
| `L`   | `output/L_<year>.csv`   | 20×1 labour vector |

It also runs the structural **gravity regression** `ln(pi)` on `ln(tau)` with
exporter/importer fixed effects (the `ln(tau)` coefficient identifies −θ).

## Sources & method
- **`pi`** — OECD ICIO inter-country input-output tables, summed to a single
  gross sector (intermediate + final, all industries), **basic prices**, domestic
  diagonal retained. Tariffs play no role here. (`stata/01_clean_icio.do`, `02_build_pi.do`)
- **`tau`** — Feodora Teti's HS6 Global Tariff Database merged onto CEPII BACI
  trade flows, trade-weighted (BACI value) to the block level (single stage; ROW
  pools all non-named countries). (`python/build_tau.py`)
- **`L`** — World Bank total labour force (`SL.TLF.TOTL.IN`); year selected by
  column label, WB regional aggregates excluded. (`stata/03_build_L.do`)
- The node set is fixed in **`countries.csv`** (single source of truth for
  membership and ordering); every non-named country folds into ROW.

`01_clean_icio.do` additionally writes bilateral **goods/services shares**
(`output/intermediate/services_share_<year>.dta`): tariffs are merchandise-only,
so the goods share can be used to rescale `tau` if applying it to all-industry flows.

## Run
1. Edit paths and `YEAR` in **`config.do`**.
2. From `stata/`: `do master.do`

Requires Stata (with `reghdfe`/`esttab`) and Python 3 (pandas). Raw inputs (ICIO
csvs, WB labour csv, Teti/BACI pickles) live outside the repo at the paths in
`config.do` and are not committed.

## Layout
```
config.do            paths + settings (edit here)
countries.csv        the 19 named nodes + ROW, with ordering
stata/
  01_clean_icio.do   ICIO -> gross flow matrix (+ goods/services shares)
  02_build_pi.do     flow matrix -> pi
  03_build_L.do      WB labour -> L
  04_gravity.do      gravity regression
  master.do          runs everything
python/
  build_tau.py       Teti + BACI -> tau
output/              pi/tau/L csvs, gravity table (committed)
```
