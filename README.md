# CDW — gravity pipeline

Builds the inputs a quantitative trade (gravity) model needs, for a set of
**19 named economies + ROW**, for a given year:

| Object | File | Meaning |
|---|---|---|
| `pi`      | `output/pi_<year>.csv`      | 20×20 import-share matrix, `pi[o,d] = X_{o,d} / Σ_o X_{o,d}` (incl. domestic) |
| `tau`     | `output/tau_<year>.csv`     | 20×20 bilateral tariff matrix, `tau[o,d] = 1 + (applied tariff d levies on o)` |
| `tau_adj` | `output/tau_adj_<year>.csv` | `tau` with the tariff rate rescaled by the bilateral **goods share** (tariffs are merchandise-only) |
| `L`       | `output/L_<year>.csv`       | 20×1 labour vector |
| `w`       | `output/w_<year>.csv`       | 20×1 balanced-trade wage vector (from `pi`, `L`; tariff-independent) |

It also runs **gravity / trade-elasticity** estimates (`output/gravity_<year>.tex`,
`.pdf`, `gravity_coef_<year>.csv`):
- **OLS**: `ln(pi)` on `ln(tau)` with exporter & importer FE (all nodes; excl. ROW).
- **France 2SLS**: restrict to French imports; regress `ln(pi_{o,fra})` on the
  endogenous `ln(w_o·tau_{o,fra})`, instrumented by `ln(tau_{o,fra})`.
- Each reported for the standard tariff and the goods-share-adjusted tariff.

## Sources & method
- **`pi`** — OECD ICIO tables, summed to a single gross sector (intermediate + final,
  all industries), **basic prices**, domestic diagonal retained. Tariff-free.
  (`stata/01_clean_icio.do`, `02_build_pi.do`)
- **`tau` / `tau_adj`** — Teti HS6 Global Tariff Database merged onto CEPII BACI,
  trade-weighted to blocks (single stage; ROW pools all non-named countries).
  `tau_adj` multiplies the rate by the ICIO bilateral goods share. (`python/build_tau.py`)
- **`L`** — World Bank total labour force (`SL.TLF.TOTL.IN`); year by column label,
  WB regional aggregates excluded. (`stata/03_build_L.do`)
- **`w`** — solves the balanced-trade fixed point `gdp = pi·gdp`, `gdp = w·L`,
  normalised `w_1 = 1` (Edward/Noah's `solveWages.m`). Uses `pi`, `L` only.
  (`python/build_wages.py`)
- The node set is fixed in **`countries.csv`** (single source of truth for membership
  and ordering); every non-named country folds into ROW.

`01_clean_icio.do` also writes bilateral goods/services shares
(`output/intermediate/services_share_<year>.dta`), which feed `tau_adj`.

## Run
1. Edit paths and `YEAR` in **`config.do`**.
2. From `stata/`: `do master.do`

Requires Stata (`reghdfe`, `ivreg2`, `esttab`) and Python 3 (pandas, numpy). Raw
inputs (ICIO csvs, WB labour csv, Teti/BACI pickles) live outside the repo at the
paths in `config.do` and are not committed.

## Layout
```
config.do            paths + settings (edit here)
countries.csv        the 19 named nodes + ROW, with ordering
stata/
  01_clean_icio.do   ICIO -> gross flow matrix (+ goods/services shares)
  02_build_pi.do     flow matrix -> pi
  03_build_L.do      WB labour -> L
  04_gravity.do      OLS + France-2SLS gravity (standard & goods-adjusted tariff)
  master.do          runs everything
python/
  build_tau.py       Teti + BACI -> tau, tau_adj
  build_wages.py     pi, L -> balanced-trade wages w
output/              pi/tau/tau_adj/L/w csvs, gravity table+pdf (committed)
```
