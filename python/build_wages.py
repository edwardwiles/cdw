#!/usr/bin/env python3
"""
build_wages.py  --  back out balanced-trade wages w_o from trade flows only.

Solves the market-clearing / trade-balance fixed point (Noah/Edward's solveWages.m):
    gdp = pi @ gdp ,   gdp_o = w_o * L_o ,   normalise  w_1 = 1
i.e. each origin's output equals the sum over destinations of its import share times
destination expenditure (=gdp under balanced trade). Uses pi and L only -> independent
of the tariff construction. Writes w_<year>.csv (node order) and w_long (node_id,w).
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/bbkinghome/edav/cdw/output")
ap.add_argument("--tmp", default="/bbkinghome/edav/cdw/output/intermediate")
ap.add_argument("--year", default="2016")
a = ap.parse_args()
y = a.year

pi = np.loadtxt(f"{a.out}/pi_{y}.csv", delimiter=",")   # rows = exporter o, cols = importer d
L  = np.loadtxt(f"{a.out}/L_{y}.csv",  delimiter=",")
n = pi.shape[0]

# gdp is the right eigenvector of pi for eigenvalue 1 (pi is column-stochastic -> exists)
vals, vecs = np.linalg.eig(pi)
k = int(np.argmin(np.abs(vals - 1.0)))
gdp = np.real(vecs[:, k])
if gdp.sum() < 0:
    gdp = -gdp
assert (gdp > 0).all(), "non-positive gdp eigenvector; check pi"

w = gdp / L
w = w / w[0]                                            # normalise w_1 = 1

resid = np.abs(gdp - pi @ gdp).max() / gdp.max()
print(f"[wages] eigval used={vals[k].real:.6f}  fixed-point resid={resid:.2e}  "
      f"w range [{w.min():.3f}, {w.max():.3f}]")

pd.Series(w).to_csv(f"{a.out}/w_{y}.csv", header=False, index=False)
pd.DataFrame({"node_id": np.arange(1, n + 1), "w": w}).to_csv(
    f"{a.tmp}/w_long_{y}.csv", index=False)
