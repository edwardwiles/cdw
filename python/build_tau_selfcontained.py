#!/usr/bin/env python3
"""
build_tau_selfcontained.py  --  same formula as build_tau.py, run directly off raw
Teti + BACI for a single year without first materializing merged_final_<year>.pkl
via the tariff_build/ pipeline (01_load_teti / 02_prep_baci / 04_finalize_merge).

  tau_{o,d}     = 1 + (trade-weighted applied tariff importer d levies on exporter o)/100
  tau_adj_{o,d} = 1 + (rate * goods_share_{o,d})/100      [goods-share-adjusted]

Used to extend the tariff build to years beyond 2016 (e.g. 2018) without re-running the
full multi-step tariff_build pipeline for each year. Equivalent to build_tau.py's
core weighting formula: merges BACI trade values with Teti's HS6 "tariff" (preferred
bilateral applied rate) directly on importer/exporter/hs6, fills unmatched tariffs with
0 (same convention as tariff_build/scripts/04_finalize_merge.py), then aggregates to
the 19-named+ROW blocks. Requires services_share_<year>.dta (from 01_clean_icio.do) for
the goods-share adjustment, and countries.csv for the block/node mapping.
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/bbkinghome/edav/cdw")
ap.add_argument("--teti-raw", default="/bbkinghome/edav/gravity_robustness/tariff_build/raw/teti")
ap.add_argument("--baci-raw", default="/bbkinghome/edav/gravity_robustness/tariff_build/raw/baci")
ap.add_argument("--out",  default="/bbkinghome/edav/cdw/output")
ap.add_argument("--tmp",  default="/bbkinghome/edav/cdw/output/intermediate")
ap.add_argument("--year", default="2018")
a = ap.parse_args()
y = a.year

nodes = pd.read_csv(f"{a.repo}/countries.csv")
named = set(nodes.loc[nodes.node_id <= 19, "code"])
nid = dict(zip(nodes.code, nodes.node_id))
blk = lambda s: np.where(s.str.lower().isin(named), s.str.lower(), "row")

# --- Teti HS6 tariff rates for the year ---
teti = pd.read_stata(f"{a.teti_raw}/tariff{y}_beta1-2024-12.dta", columns=["iso1","iso2","hs92","tariff"])
teti = teti.rename(columns={"iso1":"importer","iso2":"exporter","hs92":"hs6"})
teti["hs6"] = teti["hs6"].astype("int32")
teti["importer"] = teti["importer"].astype(str).str.lower()
teti["exporter"] = teti["exporter"].astype(str).str.lower()

# --- BACI trade values for the same year ---
cc = pd.read_csv(f"{a.baci_raw}/country_codes_V202601.csv")
num2iso = dict(zip(cc.country_code, cc.country_iso3.str.lower()))
baci = pd.read_csv(f"{a.baci_raw}/BACI_HS92_Y{y}_V202601.csv", dtype={"k": str})
baci = baci.rename(columns={"i":"exp_num","j":"imp_num","k":"hs6","v":"value"})
baci["hs6"] = baci["hs6"].astype("int32")
baci["exporter"] = baci["exp_num"].map(num2iso)
baci["importer"] = baci["imp_num"].map(num2iso)
baci = baci.dropna(subset=["exporter","importer"])

# --- merge + block-aggregate trade-weighted rate (unmatched tariffs fill 0, matching
#     tariff_build/scripts/04_finalize_merge.py's residual-fill convention) ---
m = baci.merge(teti, on=["importer","exporter","hs6"], how="left")
m["tariff"] = m["tariff"].fillna(0.0)
m["exp_blk"] = blk(m["exporter"].values)
m["imp_blk"] = blk(m["importer"].values)
m = m[m.exp_blk != m.imp_blk].copy()
m["_tv"] = m["tariff"].values * m["value"].values
g = m.groupby(["exp_blk","imp_blk"], observed=True)
rate = (g["_tv"].sum() / g["value"].sum()).rename("rate").reset_index()

# --- block goods share from ICIO services-share file (01_clean_icio.do) ---
ss = pd.read_stata(f"{a.tmp}/services_share_{y}.dta")
ss["exp_blk"] = blk(ss.exp_cntry); ss["imp_blk"] = blk(ss.imp_cntry)
gs = ss.groupby(["exp_blk", "imp_blk"], observed=True)[["goods_flow", "total_flow"]].sum()
gs["goods_share"] = np.where(gs.total_flow > 0, gs.goods_flow / gs.total_flow, 1.0)
gs = gs["goods_share"].reset_index()

t = rate.merge(gs, on=["exp_blk", "imp_blk"], how="left")
t["goods_share"] = t["goods_share"].fillna(1.0)
t["tau"]     = 1.0 + t["rate"] / 100.0
t["tau_adj"] = 1.0 + (t["rate"] * t["goods_share"]) / 100.0
t["exp_id"]  = t["exp_blk"].map(nid)
t["imp_id"]  = t["imp_blk"].map(nid)

def to_matrix(col):
    M = pd.DataFrame(index=range(1, 21), columns=range(1, 21), dtype=float)
    for _, r in t.iterrows():
        M.loc[int(r.exp_id), int(r.imp_id)] = r[col]
    for i in range(1, 21):
        if pd.isna(M.loc[i, i]):
            M.loc[i, i] = 1.0
    return M.fillna(1.0)

Mtau, Madj = to_matrix("tau"), to_matrix("tau_adj")
Mtau.to_csv(f"{a.out}/tau_{y}.csv", header=False, index=False)
Madj.to_csv(f"{a.out}/tau_adj_{y}.csv", header=False, index=False)

long = (Mtau.stack().rename_axis(["exp_id", "imp_id"]).rename("tau").reset_index()
        .merge(Madj.stack().rename_axis(["exp_id", "imp_id"]).rename("tau_adj").reset_index(),
               on=["exp_id", "imp_id"]))
long.to_csv(f"{a.tmp}/tau_long_{y}.csv", index=False)

off = t[t.exp_blk != t.imp_blk]
print(f"[tau] saved tau_{y}.csv and tau_adj_{y}.csv; mean off-diag rate={off.rate.mean():.2f}% "
      f"-> adjusted {(off.rate*off.goods_share).mean():.2f}% (mean goods share={off.goods_share.mean():.2f})")
