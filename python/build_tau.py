#!/usr/bin/env python3
"""
build_tau.py  --  block tariff matrix tau from Teti HS6 tariffs + BACI weights.

tau_{o,d} = 1 + (trade-weighted applied tariff that importer d levies on exporter o)/100,
aggregated to the 19 named blocks + ROW (single-stage BACI-weighted; ROW pools all
non-named countries). Orientation matches pi: rows = exporter node o, cols = importer node d.
Reads merged BACI-Teti flows (work/merged_final_<year>.pkl); writes tau_<year>.csv
(20x20, node order, no header) and tau_long_<year>.csv (exp_id,imp_id,tau).
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

nodes = pd.read_csv(f"{a.repo}/countries.csv")             # node_id, code, name
order = nodes.sort_values("node_id")["code"].tolist()      # length 20, ...,'row'
named = set(nodes.loc[nodes.node_id <= 19, "code"])        # lower-case ISO3
nid = dict(zip(nodes.code, nodes.node_id))

m = pd.read_pickle(f"{a.work}/merged_final_{y}.pkl")        # importer, exporter, value, tariff (percent)
m["imp_blk"] = np.where(m.importer.str.lower().isin(named), m.importer.str.lower(), "row")
m["exp_blk"] = np.where(m.exporter.str.lower().isin(named), m.exporter.str.lower(), "row")

# trade-weighted applied tariff by (importer block, exporter block)
m["_tv"] = m["tariff"].values * m["value"].values
g = m.groupby(["imp_blk", "exp_blk"], observed=True)
twt = (g["_tv"].sum() / g["value"].sum()).rename("tariff_w").reset_index()
twt["tau"] = 1.0 + twt["tariff_w"] / 100.0

# long form in pi orientation: exp_id = exporter, imp_id = importer
twt["exp_id"] = twt["exp_blk"].map(nid)
twt["imp_id"] = twt["imp_blk"].map(nid)
long = twt[["exp_id", "imp_id", "tau"]].copy()

# full 20x20, node order; named diagonal = 1 (domestic, no tariff); fill any gaps with 1
M = pd.DataFrame(index=range(1, 21), columns=range(1, 21), dtype=float)
for _, r in long.iterrows():
    M.loc[int(r.exp_id), int(r.imp_id)] = r.tau
for i in range(1, 21):
    if pd.isna(M.loc[i, i]):
        M.loc[i, i] = 1.0
n_filled = int(M.isna().sum().sum())
M = M.fillna(1.0)            # any remaining empty block-pair -> neutral tau=1

M.to_csv(f"{a.out}/tau_{y}.csv", header=False, index=False)
# rebuild long (incl diagonal) for the gravity merge
longout = M.stack().rename_axis(["exp_id", "imp_id"]).rename("tau").reset_index()
longout.to_csv(f"{a.tmp}/tau_long_{y}.csv", index=False)
print(f"[tau] saved tau_{y}.csv (20x20); filled {n_filled} empty block-pairs with tau=1; "
      f"mean off-diag tariff={twt.loc[twt.exp_blk!=twt.imp_blk,'tariff_w'].mean():.2f}%")
