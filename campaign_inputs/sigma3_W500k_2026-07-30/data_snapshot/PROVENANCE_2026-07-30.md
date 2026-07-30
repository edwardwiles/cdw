# pi.csv / tau.csv replaced 2026-07-30

**Old (WITS-era) files backed up to:** `backup_wits_era_2026-07-30/`

**New files:** 2018 data, ICIO-based trade-share matrix + Teti/BACI-weighted, goods-share-adjusted
tariff matrix. 20x20, node order per `countries.csv` (aus,fra,bra,can,che,chn,deu,esp,gbr,idn,ind,
ita,jpn,kor,mex,nld,rus,tur,usa,row), row=exporter, col=importer, self-trade diagonal retained
(tau diagonal = 1).

- `pi.csv`: `pi[o,d] = X_od / sum_o X_od`, from OECD ICIO 2016-2022 extension table, year 2018
  (`cdw/stata/01_clean_icio.do` + `02_build_pi.do`, basic prices, all industries incl. services).
- `tau.csv`: `tau_adj[o,d] = 1 + (rate_od * goods_share_od)/100`, where `rate_od` is Teti GTD HS6
  "tariff" (preferred bilateral applied rate), BACI-2018-value-weighted to the 19-named+ROW blocks,
  and `goods_share_od` is the ICIO bilateral goods-vs-services flow share from the same 01_clean_icio
  step. This is the goods-share-adjusted tariff (not the raw/undiluted rate).

Selected because the "2018, international-only (no domestic diagonal in the regression sample),
ROW excluded as destination, goods-adjusted tau" spec gives theta=4.73 — see session discussion for
the full year/spec sensitivity table this was chosen from. NOTE: theta=4.73 was estimated on a
*restricted regression sample* (diagonal + ROW-destination dropped); these pi.csv/tau.csv files are
the *complete, unrestricted* 20x20 matrices (diagonal retained, ROW included) needed as full model
inputs — the sample restriction was an estimation choice for identifying theta cleanly, not a
property of the underlying data.
