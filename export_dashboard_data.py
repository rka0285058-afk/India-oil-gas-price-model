#!/usr/bin/env python3
"""Export simulated paths + quantile grids from IOGPM v1 for the browser dashboard (dashboard/data.json)."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import india_oil_gas_price_model as M

cfg = dict(M.CFG)
cfg["fx_beta"] = M.calibrate_fx_beta(cfg)[2]
tot = sum(s["w"] for s in M.SCEN.values())
for s in M.SCEN.values():
    s["w"] /= tot
dates = pd.date_range(cfg["start"], cfg["end"], freq="MS")
N, T = cfg["n_paths"], len(dates)
names = list(M.SCEN)
sims = {n: M.simulate(M.SCEN[n], cfg, N, np.random.default_rng(cfg["seed"] + i), dates) for i, n in enumerate(names)}

P = [0.005] + [round(0.025 * k, 3) for k in range(1, 40)] + [0.995]
VARS = [
    ("brent", "Brent crude", "$/bbl", 0, 0.02, 100),
    ("icb", "Indian basket", "$/bbl", 0, 0.02, 100),
    ("fx", "USD/INR", "₹/US$", 1, 0.0, 100),
    ("petrol", "Delhi petrol", "₹/litre", 1, 0.04, 110),
    ("diesel", "Delhi diesel", "₹/litre", 1, 0.04, 105),
    ("apm", "APM domestic gas", "$/MMBtu", 2, 0.02, 8),
    ("lng", "Landed LNG", "$/MMBtu", 1, 0.02, 12),
    ("jkm", "JKM spot LNG", "$/MMBtu", 1, 0.02, 15),
    ("hh", "Henry Hub", "$/MMBtu", 2, 0.02, 4),
    ("lpg", "LPG cylinder", "₹/cylinder", 0, 0.04, 1000),
    ("cng", "Delhi CNG", "₹/kg", 1, 0.04, 90),
]
years = list(range(2027, 2041))
yrs = np.array([d.year for d in dates]); mon = np.array([d.month for d in dates])
lines = []
meta = dict(t="meta", dates=[d.strftime("%Y-%m") for d in dates], P=P, years=years, n_paths=N,
            scen=names, asof="20-23 Nov 2026",
            vars=[dict(id=a, label=b, unit=c, dec=d, cpi=e, thr=f) for a, b, c, d, e, f in VARS],
            regimes=M.REG)
lines.append(meta)
r2 = lambda a: np.round(np.asarray(a, dtype=float), 2).tolist()
for i, n in enumerate(names):
    sm = sims[n]
    for vid, *_ in VARS:
        a = sm[vid].astype(float)
        q = np.quantile(a, P, axis=0).T
        ann = np.array([np.quantile(a[:, yrs == y].mean(1), P) for y in years])
        lines.append(dict(t="q", s=i, v=vid, q=r2(q), ann=r2(ann), paths=r2(a[:30])))
    reg = np.stack([(sm["regime"] == r).mean(0) for r in range(4)], axis=1)
    lines.append(dict(t="reg", s=i, p=np.round(reg, 4).tolist()))
    b, pe, di, fx, lng, apm, lp, cg = (sm[k].astype(float) for k in ("brent", "petrol", "diesel", "fx", "lng", "apm", "lpg", "cng"))
    e27 = (yrs == 2027) & (mon == 12)
    cap27 = M.apm_cap(pd.Timestamp("2027-12-01"), cfg)
    ev = [
        ("Brent at or above $120 in some month of 2027", (b[:, yrs == 2027] >= 120).any(1)),
        ("Brent at or above $150 in some month before 2031", (b[:, yrs <= 2030] >= 150).any(1)),
        ("Brent at or below $60 in some month of 2027-2030", (b[:, (yrs >= 2027) & (yrs <= 2030)] <= 60).any(1)),
        ("Delhi petrol at or above ₹110/l by Dec 2027", pe[:, e27].ravel() >= 110),
        ("Delhi petrol at or above ₹120/l by Dec 2030", pe[:, (yrs == 2030) & (mon == 12)].ravel() >= 120),
        ("Delhi diesel at or above ₹105/l by Dec 2027", di[:, e27].ravel() >= 105),
        ("Delhi petrol below today's ₹102.12 in Dec 2027", pe[:, e27].ravel() < 102.12),
        ("USD/INR at or above 100 in Dec 2027", fx[:, e27].ravel() >= 100),
        ("USD/INR at or above 120 in Dec 2035", fx[:, (yrs == 2035) & (mon == 12)].ravel() >= 120),
        ("Landed LNG averages $15/MMBtu or more in 2027", lng[:, yrs == 2027].mean(1) >= 15),
        ("LPG cylinder at or above ₹1,000 in Dec 2027", lp[:, e27].ravel() >= 1000),
        ("LPG cylinder at or above ₹1,200 in Dec 2030", lp[:, (yrs == 2030) & (mon == 12)].ravel() >= 1200),
        ("Delhi CNG at or above ₹90/kg in Dec 2027", cg[:, e27].ravel() >= 90),
        ("Delhi CNG at or above ₹100/kg in Dec 2030", cg[:, (yrs == 2030) & (mon == 12)].ravel() >= 100),
        ("APM gas sits at its price cap in Dec 2027", apm[:, e27].ravel() >= cap27 - 1e-6),
    ]
    lines.append(dict(t="risk", s=i, names=[e[0] for e in ev], p=[float(e[1].mean()) for e in ev]))
out = os.path.join(HERE, "dashboard", "data.json")
with open(out, "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(l, separators=(",", ":"), ensure_ascii=False) for l in lines) + "\n]\n")
print("wrote", out, os.path.getsize(out) / 1e6, "MB", len(lines), "records")
