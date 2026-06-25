#!/usr/bin/env python3
"""
build_tau.py  --  block tariff matrices from Teti HS6 tariffs + BACI weights.

  tau_{o,d}     = 1 + (trade-weighted applied tariff importer d levies on exporter o)/100
  tau_adj_{o,d} = 1 + (rate * goods_share_{o,d})/100      [goods-share-adjusted]

Tariffs are merchandise-only, but they are applied to all-industry flows; the
goods share (ICIO, step 01) rescales the rate to the portion of the o->d flow that
is actually tariffable. Aggregated to 19 named blocks + ROW. Orientation matches pi:
rows = exporter node o, cols = importer node d. Writes tau_<y>.csv, tau_adj_<y>.csv
(20x20, node order, no header) and tau_long_<y>.csv (exp_id,imp_id,tau,tau_adj).
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/bbkinghome/edav/cdw")
ap.add_argument("--work", default="/bbkinghome/edav/gravity_robustness/tariff_build/work")
ap.add_argument("--out",  default="/bbkinghome/edav/cdw/output")
ap.add_argument("--tmp",  default="/bbkinghome/edav/cdw/output/intermediate")
ap.add_argument("--year", default="2016")
a = ap.parse_args()
y = a.year

nodes = pd.read_csv(f"{a.repo}/countries.csv")
named = set(nodes.loc[nodes.node_id <= 19, "code"])
nid = dict(zip(nodes.code, nodes.node_id))
blk = lambda s: np.where(s.str.lower().isin(named), s.str.lower(), "row")

# --- trade-weighted Teti tariff rate by (exporter block, importer block) ---
m = pd.read_pickle(f"{a.work}/merged_final_{y}.pkl")        # importer, exporter, value, tariff (%)
m["imp_blk"] = blk(m.importer); m["exp_blk"] = blk(m.exporter)
m["_tv"] = m["tariff"].values * m["value"].values
g = m.groupby(["exp_blk", "imp_blk"], observed=True)
rate = (g["_tv"].sum() / g["value"].sum()).rename("rate").reset_index()   # rate_{o,d} (%)

# --- block goods share from ICIO services-share file (step 01) ---
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
    for i in range(1, 21):                       # named diagonal = domestic = no tariff
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
