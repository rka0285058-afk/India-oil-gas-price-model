#!/usr/bin/env python3
"""
India Oil & Gas Price Model (IOGPM v1) - dynamic, geopolitics-aware, long-horizon (Dec-2026 .. Dec-2040)

ARCHITECTURE (monthly Monte Carlo, vectorised across paths)
  1. Geopolitical regime layer  : 4-state Markov chain (Normalisation / Simmering disruption /
                                  Acute escalation / Structural glut). Scenario presets re-weight the
                                  transition hazards and the initial regime probabilities.
  2. Brent                      : regime-switching mean-reverting log price (Schwartz one-factor style)
                                  with an entry jump into the escalation regime.
  3. USD/INR                    : inflation-gap trend + oil-beta (calibrated from RBI/PPAC data points)
  4. Indian Crude Basket (ICB)  : Brent x AR(1) basis  - Russian-crude discount overlay (PPAC/MoPNG basis)
  5. Gas                        : Henry Hub (OU), JKM = fundamentals + geopolitical premium (OU to regime mean),
                                  India landed LNG = oil-linked + HH-linked + spot portfolio,
                                  APM gas = clip(10% x lagged ICB, floor, cap) -- official Parikh-type formula
  6. Retail fuels (Delhi)       : cost-reflective price = (crude + crack + other + excise) x (1+VAT);
                                  under-recovery ledger; OMC/government reaction function (hikes, excise buffer,
                                  recoup-before-cut).
  7. Dynamic updating           : --obs observations.csv re-initialises state; Hamilton regime filter updates
                                  regime probabilities once >=12 monthly Brent observations are supplied.

Everything flagged ASSUMPTION is a judgement/placeholder to be replaced by a PPAC/MoPNG series when available.
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
L_BBL = 158.987  # litres per barrel

REG = ["Normalisation", "Simmering disruption", "Acute escalation", "Structural glut"]

# monthly regime transition matrix (rows = from). Mean durations ~ 24, 10, 5, 36 months.
BASE_P = np.array([
    [0.958, 0.025, 0.005, 0.012],
    [0.045, 0.900, 0.035, 0.020],
    [0.050, 0.150, 0.800, 0.000],
    [0.020, 0.010, 0.003, 0.967],
])

# ----------------------------------------------------------------------------------------------
# Official / market anchors used for calibration (source tags in comments)
# ----------------------------------------------------------------------------------------------
ANCHORS = dict(
    # PPAC (via Financial Express / ThePrint / The Wire reports of PPAC data)
    icb={"2025-10": 65.08, "2025-09": 69.61, "2026-03": 113.49, "2026-04": 114.48,
         "2026-06": 86.31, "2026-08": 90.19, "2026-09": 116.09},
    brent_month={"2025-04": 67.79, "2026-04": 120.55, "2026-10": 124.0},  # PPAC Apr; EIA STEO Oct-26
    # RBI reference rate / FBIL
    usdinr={"2025-04": 85.05, "2026-04": 95.24, "2026-05-low": 96.965, "2026-11-19": 93.35},
    # EIA STEO Nov-2026
    eia={"dec26": 99.0, "y2027": 86.0, "q4_2027": 77.0},
)

CFG = dict(
    n_paths=3000, seed=2026,
    start="2026-12-01", end="2040-12-01",
    # --- state at ~20-23 Nov 2026
    brent0=92.0,                      # front-month settle ~$92 (trade press, week of 16-23 Nov 2026)
    brent_hist0=[105.0, 124.0, 98.0],  # Sep(ASSUMPTION), Oct (EIA STEO), Nov est. -> 3-month LNG lag
    icb_prev0=95.0,                   # ASSUMPTION ~ Nov-2026 ICB
    fx0=93.35,                        # RBI reference rate 19-Nov-2026
    hh0=3.2, jkm0=30.0,               # Henry Hub Dec futures; JKM ~ $30 (Platts/JOGMEC, Oct-Nov 2026)
    petrol_rsp0=102.12, diesel_rsp0=95.20,   # Delhi RSP, unchanged since 1-Jun-2026
    petrol_crack0=18.0, diesel_crack0=50.0,  # ASSUMPTION (diesel export levy Rs14/l on 16-Nov implies elevated crack)
    # --- macro
    us_cpi=0.02, in_cpi=0.04, fx_trend=0.025, fx_vol=0.045,
    # --- Brent regimes (real 2026 $/bbl)
    m_brent=[72.0, 92.0, 125.0, 58.0], sig_brent=[0.24, 0.28, 0.30, 0.26], kappa=0.9,
    # --- gas
    kappa_hh=1.5, sig_hh=0.35,
    prem_mean=[1.0, 9.0, 24.0, 0.0], kappa_prem=2.0, sig_prem=3.0,
    w_oil=0.45, w_hh=0.15, w_spot=0.40,       # ASSUMPTION: India LNG portfolio weights (verify vs PPAC LNG data)
    oil_slope=0.125, oil_const=0.5,
    apm_cap0=7.00, apm_cap_step=0.25, apm_cap_max=8.50, apm_floor=4.0, apm_pct=0.10,
    # --- retail (Delhi)
    vat=dict(petrol=0.194, diesel=0.1675),
    other0=dict(petrol=7.4, diesel=5.7),      # ASSUMPTION: freight/marketing + dealer commission (Rs/l)
    excise0=dict(petrol=11.9, diesel=7.8),    # post 26-Mar-2026 cut (21.9-10, 17.8-10)  [verify on PPAC tariff table]
    excise_ref=dict(petrol=21.9, diesel=17.8),
    excise_floor=dict(petrol=5.0, diesel=0.0),
    crack_mean=dict(petrol=[14, 20, 35, 12], diesel=[20, 38, 80, 16]),
    crack_sig=dict(petrol=6.0, diesel=12.0), kappa_crack=2.5,
    hike_thr=dict(petrol=4.0, diesel=12.0),
    ledger0=dict(petrol=40.0, diesel=150.0),  # ASSUMPTION: cumulative under-recovery (Rs/l-months) to recoup
    # --- LPG (Delhi 14.2 kg domestic cylinder). Official: Rs942 since 7-Jun-2026 (Rs913 before; +Rs60 on 7-Mar-26); Saudi CP
    #     $543/t Feb-26, $775 Apr-26, $790 Jun-26; govt cites ~Rs703/cylinder pre-June-hike loss.
    lpg_rsp0=942.0, lpg_kg=14.2, gst_lpg=0.05,
    cp_const=250.0, cp_brent=4.6, cp_prem=2.0,     # ASSUMPTION: Saudi CP ($/t) = 250 + 4.6 x 3m-avg Brent + 2 x gas premium (fits Feb/Apr/Jun-26 CP within ~$40)
    lpg_other0=460.0,                               # Rs/cyl: residual (freight, bottling, distributor commission, duties) calibrated so cost ~Rs1,600 at CP $790 (govt: Rs913 + Rs703)
    lpg_hike_thr=250.0, lpg_step=30.0, lpg_ledger0=3000.0,   # ASSUMPTION: hike when under-recovery > Rs250; Rs30 steps (7-Mar +60, 6-Jun +29)
    # --- CNG (Delhi, IGL). Official: Rs86.98/kg (reported 11-Nov-2026); Delhi CNG VAT 5%.
    cng_rsp0=86.98, cng_vat=0.05, cng_mmbtu_kg=0.0475,
    cng_w_apm=0.45,                                 # ASSUMPTION: share of APM gas in Delhi CGD basket (rest regasified LNG)
    cng_other0=26.0,                                # Rs/kg: pipeline tariff, compression, margin, opex (ASSUMPTION, calibrated to Rs86.98)
    cng_hike_gap=2.5, cng_step=2.0, cng_cut_gap=5.0,
)

SCEN = {
    "Base (market-implied normalisation)": dict(
        init=[0.20, 0.60, 0.10, 0.10], mult={}, ru_share=0.25, ru_disc=1.5, ru_prem=0.0, w=0.30),
    "Diplomatic settlement - Hormuz reopens": dict(
        init=[0.55, 0.35, 0.02, 0.08], mult={(1, 0): 3.0, (1, 2): 0.4, (0, 1): 0.6},
        ru_share=0.25, ru_disc=2.5, ru_prem=0.0, w=0.20),
    "Prolonged standoff": dict(
        init=[0.05, 0.80, 0.10, 0.05], mult={(1, 0): 0.3, (1, 2): 1.5, (2, 1): 0.7},
        ru_share=0.30, ru_disc=1.0, ru_prem=0.0, w=0.20),
    "Escalation / renewed war": dict(
        init=[0.03, 0.37, 0.50, 0.10], mult={(2, 1): 0.5, (2, 0): 0.5, (1, 2): 2.0},
        ru_share=0.35, ru_disc=0.0, ru_prem=0.5, w=0.08),
    "Structural glut / demand peak": dict(
        init=[0.30, 0.40, 0.05, 0.25], mult={(0, 3): 4.0, (1, 3): 3.0, (3, 0): 0.5, (3, 1): 0.3, (3, 2): 0.3},
        ru_share=0.25, ru_disc=3.0, ru_prem=0.0, w=0.12),
    "Russia sanctions bite": dict(
        init=[0.20, 0.60, 0.10, 0.10], mult={(1, 2): 1.3, (0, 1): 1.3},
        ru_share=0.10, ru_disc=0.0, ru_prem=1.5, w=0.10),
}


def build_P(mult):
    P = BASE_P.copy()
    for (i, j), m in mult.items():
        P[i, j] *= m
    for i in range(4):
        off = P[i].copy()
        off[i] = 0
        if off.sum() > 0.6:
            off *= 0.6 / off.sum()
        P[i] = off
        P[i, i] = 1 - off.sum()
    return P


def stationary(P):
    w, v = np.linalg.eig(P.T)
    s = np.real(v[:, np.argmax(np.real(w))])
    return s / s.sum()


def calibrate_fx_beta(cfg):
    a = ANCHORS
    dfx1 = np.log(a["usdinr"]["2026-04"] / a["usdinr"]["2025-04"])
    db1 = np.log(a["brent_month"]["2026-04"] / a["brent_month"]["2025-04"])
    b1 = (dfx1 - cfg["fx_trend"]) / db1
    dfx2 = np.log(a["usdinr"]["2026-11-19"] / a["usdinr"]["2026-04"])
    db2 = np.log(cfg["brent0"] / a["brent_month"]["2026-04"])
    b2 = (dfx2 - cfg["fx_trend"] * 0.55) / db2
    return b1, b2, 0.5 * (b1 + b2)


def apm_cap(date, cfg):
    n = (date.year - 2027) + (1 if date.month >= 4 else 0) if date.year >= 2027 else 0
    return min(cfg["apm_cap_max"], cfg["apm_cap0"] + cfg["apm_cap_step"] * n)


def hh_target_real(y):
    return np.interp(y, [2026.9, 2027.5, 2032.0], [3.0, 2.9, 3.8])


def lng_struct_discount(y):
    return np.interp(y, [2027.5, 2031.0], [0.0, 2.0])  # new US/Qatar supply wave


def policy(ur, rsp, E, L, H, E_floor, E_ref, rng):
    """Monthly OMC/government reaction function (applied at month end)."""
    N = len(ur)
    L = np.maximum(0.0, L + ur)
    hike = (ur > H) & (rng.random(N) < np.clip(0.05 + 0.01 * ur, 0, 0.6))
    rsp = rsp + hike * 3.0
    spike = (ur > 2 * H + 10) & (E > E_floor) & (rng.random(N) < 0.4)   # fiscal buffer: cut excise in a spike
    E = E - spike * np.minimum(5.0, E - E_floor)
    ov = ur < -2.0
    restore = ov & (E < E_ref - 1e-9) & (rng.random(N) < 0.5)           # recoup excise first
    E = E + restore * np.minimum(2.5, E_ref - E)
    cut = ov & (E >= E_ref - 1e-9) & (L <= 1e-9) & (rng.random(N) < 0.4)  # then pass relief to pump
    rsp = rsp - cut * np.minimum(3.0, np.maximum(1.0, -ur / 2))
    return rsp, E, L


def simulate(sc, cfg, n, rng, dates):
    T, N, dt = len(dates), n, 1 / 12
    P = build_P(sc["mult"])
    cumP = np.cumsum(P, axis=1)
    reg = rng.choice(4, size=N, p=sc["init"])
    ka = np.exp(-cfg["kappa"] * dt)
    sdf = np.sqrt((1 - np.exp(-2 * cfg["kappa"] * dt)) / (2 * cfg["kappa"]))
    kh = np.exp(-cfg["kappa_hh"] * dt)
    sdh = np.sqrt((1 - np.exp(-2 * cfg["kappa_hh"] * dt)) / (2 * cfg["kappa_hh"]))
    kp = np.exp(-cfg["kappa_prem"] * dt)
    sdp = np.sqrt((1 - np.exp(-2 * cfg["kappa_prem"] * dt)) / (2 * cfg["kappa_prem"]))
    kc = np.exp(-cfg["kappa_crack"] * dt)
    sdc = np.sqrt((1 - np.exp(-2 * cfg["kappa_crack"] * dt)) / (2 * cfg["kappa_crack"]))

    lnB = np.full(N, np.log(cfg["brent0"]))
    lnFX = np.full(N, np.log(cfg["fx0"]))
    lnH = np.full(N, np.log(cfg["hh0"]))
    u_r = np.zeros(N)
    fund0 = 0.5 * (cfg["oil_slope"] * cfg["brent0"] + cfg["oil_const"]) + 0.5 * (1.15 * cfg["hh0"] + 3.5)
    PR = np.full(N, cfg["jkm0"] - fund0)
    hist = [np.full(N, b) for b in cfg["brent_hist0"]]
    icb_prev = np.full(N, cfg["icb_prev0"])
    cr = {f: np.full(N, cfg[f + "_crack0"]) for f in ("petrol", "diesel")}
    rsp = {"petrol": np.full(N, cfg["petrol_rsp0"]), "diesel": np.full(N, cfg["diesel_rsp0"])}
    E = {f: np.full(N, cfg["excise0"][f]) for f in rsp}
    Led = {f: np.full(N, cfg["ledger0"][f]) for f in rsp}

    names = ["brent", "icb", "fx", "petrol", "diesel", "apm", "lng", "jkm", "hh", "ur_petrol", "ur_diesel", "regime",
             "lpg", "cng", "ur_lpg"]
    rng2 = np.random.Generator(rng.bit_generator.jumped())   # separate stream: keeps petrol/diesel/gas draws unchanged
    lpg_p = np.full(N, cfg["lpg_rsp0"]); lpg_L = np.full(N, cfg["lpg_ledger0"])
    cng_p = np.full(N, cfg["cng_rsp0"])
    out = {k: np.zeros((N, T), dtype=np.float32) for k in names}

    for t, d in enumerate(dates):
        k = t + 1
        y = d.year + (d.month - 1) / 12
        infl_us = (1 + cfg["us_cpi"]) ** (k / 12)
        infl_in = (1 + cfg["in_cpi"]) ** (k / 12)
        # 1. regime
        u = rng.random(N)
        newreg = np.minimum((u[:, None] > cumP[reg]).sum(1), 3)
        entered = (newreg == 2) & (reg != 2)
        reg = newreg
        # 2. Brent
        mu = np.log(np.array(cfg["m_brent"])[reg] * infl_us)
        sig = np.array(cfg["sig_brent"])[reg]
        lnB_prev = lnB.copy()
        lnB = lnB * ka + mu * (1 - ka) + sig * sdf * rng.standard_normal(N)
        lnB = lnB + entered * rng.normal(0.10, 0.05, N)
        B = np.exp(lnB)
        # 3. FX
        lnFX = lnFX + cfg["fx_trend"] * dt + cfg["fx_beta"] * (lnB - lnB_prev) \
            + cfg["fx_vol"] * np.sqrt(dt) * rng.standard_normal(N)
        FX = np.exp(lnFX)
        # 4. ICB
        u_r = 0.6 * u_r + 0.03 * rng.standard_normal(N)
        ICB = B * (1 + u_r) - sc["ru_share"] * sc["ru_disc"] + sc["ru_prem"]
        # 5. gas
        lnH = lnH * kh + np.log(hh_target_real(y) * infl_us) * (1 - kh) + cfg["sig_hh"] * sdh * rng.standard_normal(N)
        HH = np.exp(lnH)
        fund = 0.5 * (cfg["oil_slope"] * B + cfg["oil_const"]) + 0.5 * (1.15 * HH + 3.5) - lng_struct_discount(y)
        PR = PR * kp + np.array(cfg["prem_mean"])[reg] * (1 - kp) + cfg["sig_prem"] * sdp * rng.standard_normal(N)
        PR = np.maximum(PR, -1.5)
        JKM = np.maximum(fund + PR, 0.6 * fund)
        hist = hist[1:] + [B]
        B3 = (hist[0] + hist[1] + hist[2]) / 3
        LNG = (cfg["w_oil"] * (cfg["oil_slope"] * B3 + cfg["oil_const"])
               + cfg["w_hh"] * (1.15 * HH + 4.25) + cfg["w_spot"] * (JKM - 0.4))
        APM = np.clip(cfg["apm_pct"] * icb_prev, cfg["apm_floor"], apm_cap(d, cfg))
        icb_prev = ICB
        # 6. retail fuels
        for f in ("petrol", "diesel"):
            cm = np.array(cfg["crack_mean"][f])[reg] * infl_us
            cr[f] = np.maximum(cr[f] * kc + cm * (1 - kc) + cfg["crack_sig"][f] * sdc * rng.standard_normal(N), 5.0)
            crude_l = ICB * FX / L_BBL
            crack_l = cr[f] * FX / L_BBL
            C = (crude_l + crack_l + cfg["other0"][f] * infl_in + E[f]) * (1 + cfg["vat"][f])
            ur = C - rsp[f]
            out[f][:, t] = rsp[f]
            out["ur_" + f][:, t] = ur
            rsp[f], E[f], Led[f] = policy(ur, rsp[f], E[f], Led[f], cfg["hike_thr"][f],
                                         cfg["excise_floor"][f], cfg["excise_ref"][f], rng)
        # 7. LPG (Saudi CP-linked, administered price) and CNG (gas-basket-linked, IGL)
        CP = cfg["cp_const"] + cfg["cp_brent"] * B3 + cfg["cp_prem"] * PR
        C_l = (CP * FX / 1000.0 * cfg["lpg_kg"] + cfg["lpg_other0"] * infl_in) * (1 + cfg["gst_lpg"])
        ur_l = C_l - lpg_p
        out["lpg"][:, t], out["ur_lpg"][:, t] = lpg_p, ur_l
        lpg_L = np.maximum(0.0, lpg_L + ur_l)
        hk = (ur_l > cfg["lpg_hike_thr"]) & (rng2.random(N) < np.clip(0.04 + 0.00015 * ur_l, 0, 0.3))
        ct = (ur_l < -50) & (lpg_L <= 1e-9) & (rng2.random(N) < 0.3)
        lpg_p = lpg_p + hk * cfg["lpg_step"] - ct * np.minimum(cfg["lpg_step"], -ur_l / 2)
        basket = cfg["cng_w_apm"] * APM + (1 - cfg["cng_w_apm"]) * LNG
        C_c = (basket * cfg["cng_mmbtu_kg"] * FX + cfg["cng_other0"] * infl_in) * (1 + cfg["cng_vat"])
        gap = C_c - cng_p
        out["cng"][:, t] = cng_p
        up = (gap > cfg["cng_hike_gap"]) & (rng2.random(N) < 0.45)
        dn = (gap < -cfg["cng_cut_gap"]) & (rng2.random(N) < 0.25)
        cng_p = cng_p + up * cfg["cng_step"] - dn * 1.0
        out["brent"][:, t], out["icb"][:, t], out["fx"][:, t] = B, ICB, FX
        out["apm"][:, t], out["lng"][:, t], out["jkm"][:, t], out["hh"][:, t] = APM, LNG, JKM, HH
        out["regime"][:, t] = reg
    return out


def wquant(x, w, qs):
    i = np.argsort(x)
    x, w = x[i], w[i]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cw, x)


def filter_regimes(brent_monthly, cfg):
    """Hamilton filter on observed monthly Brent levels (needs >=12 obs)."""
    a = np.array([0.25] * 4)
    m = np.array(cfg["m_brent"])
    s = np.array(cfg["sig_brent"]) / np.sqrt(2 * cfg["kappa"])
    for b in brent_monthly:
        a = a @ BASE_P
        like = np.exp(-0.5 * ((np.log(b) - np.log(m)) / s) ** 2) / s
        a = a * like
        a /= a.sum()
    return a


def load_obs(path, cfg):
    df = pd.read_csv(path)
    last = df.dropna(how="all", subset=[c for c in df.columns if c != "date"]).iloc[-1]
    mp = {"brent": "brent0", "usdinr": "fx0", "henry_hub": "hh0", "jkm": "jkm0",
          "petrol_delhi": "petrol_rsp0", "diesel_delhi": "diesel_rsp0", "icb": "icb_prev0"}
    for c, k in mp.items():
        if c in df.columns and pd.notna(last.get(c)):
            cfg[k] = float(last[c])
    br = df["brent"].dropna().values if "brent" in df.columns else []
    return filter_regimes(br, cfg) if len(br) >= 12 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=CFG["n_paths"])
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--obs", default=None, help="CSV of latest observations (see data/observations_template.csv)")
    ap.add_argument("--weights", default=None, help='override scenario weights e.g. "Prolonged standoff=0.4"')
    a = ap.parse_args()
    cfg = dict(CFG)
    cfg["n_paths"], cfg["seed"] = a.paths, a.seed
    b1, b2, bm = calibrate_fx_beta(cfg)
    cfg["fx_beta"] = bm
    filt = load_obs(a.obs, cfg) if a.obs else None
    if filt is not None:
        print("Regime filter posterior from observations:", dict(zip(REG, np.round(filt, 3))))
        for s in SCEN.values():
            s["init"] = list(filt)
    if a.weights:
        for kv in a.weights.split(","):
            k, v = kv.split("=")
            SCEN[k.strip()]["w"] = float(v)
    tot = sum(s["w"] for s in SCEN.values())
    for s in SCEN.values():
        s["w"] /= tot

    dates = pd.date_range(cfg["start"], cfg["end"], freq="MS")
    N = cfg["n_paths"]
    print(f"Horizon {dates[0]:%b-%Y} .. {dates[-1]:%b-%Y} ({len(dates)} months), {N} paths x {len(SCEN)} scenarios")
    print(f"FX oil-beta: window1 {b1:.3f}, window2 {b2:.3f} -> using {bm:.3f}")
    print("Stationary regime mix (base matrix):", dict(zip(REG, np.round(stationary(BASE_P), 3))))

    sims = {}
    for i, (name, sc) in enumerate(SCEN.items()):
        sims[name] = simulate(sc, cfg, N, np.random.default_rng(cfg["seed"] + i), dates)

    var_keys = ["brent", "icb", "fx", "petrol", "diesel", "apm", "lng", "jkm", "hh", "ur_petrol", "ur_diesel", "lpg", "cng", "ur_lpg"]
    W = np.concatenate([np.full(N, sc["w"] / N) for sc in SCEN.values()])
    stack = {v: np.concatenate([sims[n][v] for n in SCEN]) for v in var_keys}
    stack["regime"] = np.concatenate([sims[n]["regime"] for n in SCEN])

    # ---------- monthly fan (weighted + per scenario)
    qs = [0.10, 0.50, 0.90]
    rows = []
    for v in ["brent", "icb", "fx", "petrol", "diesel", "apm", "lng", "jkm", "hh", "lpg", "cng"]:
        for t, d in enumerate(dates):
            p = wquant(stack[v][:, t].astype(float), W, qs)
            rows.append(("Probability-weighted", v, d, p[0], p[1], p[2], float(np.average(stack[v][:, t], weights=W))))
        for n in SCEN:
            arr = sims[n][v]
            pp = np.percentile(arr, [10, 50, 90], axis=0)
            for t, d in enumerate(dates):
                rows.append((n, v, d, pp[0, t], pp[1, t], pp[2, t], float(arr[:, t].mean())))
    monthly = pd.DataFrame(rows, columns=["scenario", "variable", "date", "p10", "p50", "p90", "mean"])
    monthly.to_csv(os.path.join(OUT, "forecast_monthly.csv"), index=False)

    # ---------- annual averages (nominal and real 2026 prices)
    years = list(range(2027, 2041))
    rows = []
    for v in ["brent", "icb", "fx", "petrol", "diesel", "apm", "lng", "jkm", "hh", "lpg", "cng"]:
        for scen, arr_all, w in [("Probability-weighted", stack[v], W)] + [(n, sims[n][v], None) for n in SCEN]:
            for y in years:
                m = np.array([d.year == y for d in dates])
                ann = arr_all[:, m].astype(float).mean(1)
                ww = w if w is not None else np.full(len(ann), 1.0)
                p = wquant(ann, ww, qs)
                row = [scen, v, y, p[0], p[1], p[2], float(np.average(ann, weights=ww))]
                defl = ((1 + cfg["in_cpi"]) ** ((y - 2026) if v in ("petrol", "diesel", "lpg", "cng") else 0)) * \
                       ((1 + cfg["us_cpi"]) ** ((y - 2026) if v in ("brent", "icb", "lng", "jkm", "hh", "apm") else 0))
                row.append(p[1] / defl)
                rows.append(row)
    annual = pd.DataFrame(rows, columns=["scenario", "variable", "year", "p10", "p50", "p90", "mean", "p50_real_2026"])
    annual.to_csv(os.path.join(OUT, "forecast_annual.csv"), index=False)

    # ---------- regime probabilities through time (weighted)
    rp = []
    for t, d in enumerate(dates):
        for r in range(4):
            rp.append((d, REG[r], float(np.sum(W * (stack["regime"][:, t] == r)))))
    pd.DataFrame(rp, columns=["date", "regime", "probability"]).to_csv(os.path.join(OUT, "regime_probabilities.csv"), index=False)

    # ---------- risk probabilities (weighted)
    def pr(mask):
        return float(np.sum(W[mask]))
    yrs = np.array([d.year for d in dates])
    mon = np.array([d.month for d in dates])
    i27 = yrs == 2027
    end27 = (yrs == 2027) & (mon == 12)
    risks = [
        ("Brent >= $120 in at least one month of 2027", pr((stack["brent"][:, i27] >= 120).any(1))),
        ("Brent >= $150 in at least one month before 2031", pr((stack["brent"][:, yrs <= 2030] >= 150).any(1))),
        ("Brent <= $60 in at least one month of 2027-2030", pr((stack["brent"][:, (yrs >= 2027) & (yrs <= 2030)] <= 60).any(1))),
        ("Delhi petrol >= Rs110/l by Dec-2027", pr(stack["petrol"][:, end27].ravel() >= 110)),
        ("Delhi petrol >= Rs120/l by Dec-2030", pr(stack["petrol"][:, (yrs == 2030) & (mon == 12)].ravel() >= 120)),
        ("Delhi diesel >= Rs105/l by Dec-2027", pr(stack["diesel"][:, end27].ravel() >= 105)),
        ("Delhi petrol below today's Rs102.12 at Dec-2027", pr(stack["petrol"][:, end27].ravel() < 102.12)),
        ("USD/INR >= 100 at Dec-2027", pr(stack["fx"][:, end27].ravel() >= 100)),
        ("USD/INR >= 120 at Dec-2035", pr(stack["fx"][:, (yrs == 2035) & (mon == 12)].ravel() >= 120)),
        ("India landed LNG >= $15/MMBtu in 2027 average", pr(stack["lng"][:, i27].mean(1) >= 15)),
        ("Domestic LPG cylinder >= Rs1,000 at Dec-2027", pr(stack["lpg"][:, end27].ravel() >= 1000)),
        ("Domestic LPG cylinder >= Rs1,200 at Dec-2030", pr(stack["lpg"][:, (yrs == 2030) & (mon == 12)].ravel() >= 1200)),
        ("Delhi CNG >= Rs90/kg at Dec-2027", pr(stack["cng"][:, end27].ravel() >= 90)),
        ("Delhi CNG >= Rs100/kg at Dec-2030", pr(stack["cng"][:, (yrs == 2030) & (mon == 12)].ravel() >= 100)),
        ("APM gas at the policy cap in Dec-2027", pr(stack["apm"][:, end27].ravel() >= apm_cap(pd.Timestamp("2027-12-01"), cfg) - 1e-6)),
    ]
    pd.DataFrame(risks, columns=["event", "probability"]).to_csv(os.path.join(OUT, "risk_probabilities.csv"), index=False)

    # ---------- validation printout
    print("\n=== VALIDATION / CALIBRATION CHECKS ===")
    print("APM formula (10% x previous-month ICB, cap):")
    print(f"  Nov-2025: 10% x Oct-25 ICB 65.08 = {0.10 * 65.08:.2f}  vs Argus-reported $6.55")
    print(f"  Oct-2025: 10% x Sep-25 ICB 69.61 = {0.10 * 69.61:.2f} -> capped at 6.75 vs reported $6.75")
    print("Cost structure -> implied cracks (VAT/other costs are assumptions):")
    for lab, f, rsp0, ur0, icb, fx, e in [
        ("1-Apr-2026 petrol", "petrol", 94.77, 24.40, 114.0, 94.0, 11.9),
        ("1-Apr-2026 diesel", "diesel", 87.67, 104.99, 114.0, 94.0, 7.8),
        ("mid-Jun-2026 petrol", "petrol", 102.12, 3.0, 86.31, 94.5, 11.9),
        ("mid-Jun-2026 diesel", "diesel", 95.20, 27.0, 86.31, 94.5, 7.8)]:
        C = rsp0 + ur0
        inner = C / (1 + cfg["vat"][f]) - cfg["other0"][f] - e
        crude = icb * fx / L_BBL
        crack_l = inner - crude
        print(f"  {lab:20s} cost-reflective Rs{C:7.2f}/l -> implied crack ${crack_l * L_BBL / fx:.0f}/bbl")
    print("Model vs EIA Nov-2026 STEO (Brent):")
    base_n = list(SCEN)[0]
    for lab, cond, tgt in [("Dec-2026", (yrs == 2026), ANCHORS["eia"]["dec26"]),
                           ("2027 avg", (yrs == 2027), ANCHORS["eia"]["y2027"]),
                           ("Q4-2027 avg", (yrs == 2027) & (mon >= 10), ANCHORS["eia"]["q4_2027"])]:
        arr = stack["brent"][:, cond].astype(float).mean(1)
        print(f"  {lab:12s} EIA ${tgt:6.1f} | model P50 ${wquant(arr, W, [0.5])[0]:6.1f}  "
              f"(P10 ${wquant(arr, W, [0.1])[0]:.0f} - P90 ${wquant(arr, W, [0.9])[0]:.0f})")

    # ---------- static sensitivities
    print("\n=== STATIC SENSITIVITIES (steady state, pass-through to cost-reflective prices) ===")
    fx = cfg["fx0"]
    print(f"  +$10/bbl Brent -> Delhi petrol cost-reflective +Rs{10 * fx / L_BBL * (1 + cfg['vat']['petrol']):.1f}/l,"
          f" diesel +Rs{10 * fx / L_BBL * (1 + cfg['vat']['diesel']):.1f}/l")
    print(f"  +Rs1/USD     -> petrol cost-reflective +Rs{(90 + 18) / L_BBL * (1 + cfg['vat']['petrol']):.2f}/l at $90 ICB + $18 crack")
    print(f"  +$10/bbl Brent -> landed LNG +${cfg['w_oil'] * cfg['oil_slope'] * 10 + cfg['w_spot'] * 0.5 * cfg['oil_slope'] * 10:.2f}/MMBtu "
          f"(APM unchanged while capped: cap binds for ICB >= $70)")

    # ---------- compact headline table
    print("\n=== HEADLINE (probability-weighted P50, nominal; P10-P90 in brackets) ===")
    show = annual[(annual.scenario == "Probability-weighted") & (annual.year.isin([2027, 2028, 2030, 2035, 2040]))]
    for v, lab, fmt in [("brent", "Brent $/bbl", "{:.0f}"), ("icb", "ICB $/bbl", "{:.0f}"), ("fx", "USD/INR", "{:.1f}"),
                        ("petrol", "Delhi petrol Rs/l", "{:.1f}"), ("diesel", "Delhi diesel Rs/l", "{:.1f}"),
                        ("apm", "APM gas $/MMBtu", "{:.2f}"), ("lng", "India LNG landed $/MMBtu", "{:.1f}"),
                        ("lpg", "LPG 14.2kg Delhi Rs/cyl", "{:.0f}"), ("cng", "Delhi CNG Rs/kg", "{:.1f}")]:
        parts = []
        for y in [2027, 2028, 2030, 2035, 2040]:
            r = show[(show.variable == v) & (show.year == y)].iloc[0]
            parts.append(f"{y}: " + fmt.format(r.p50) + " [" + fmt.format(r.p10) + "-" + fmt.format(r.p90) + "]")
        print(f"  {lab:26s} " + " | ".join(parts))
    print("\n=== RISK PROBABILITIES (weighted) ===")
    for e, p in risks:
        print(f"  {p * 100:5.1f}%  {e}")
    # initial-state diagnostics
    fx0 = cfg["fx0"]
    for f in ("petrol", "diesel"):
        crude = cfg["icb_prev0"] * fx0 / L_BBL
        C = (crude + cfg[f + "_crack0"] * fx0 / L_BBL + cfg["other0"][f] + cfg["excise0"][f]) * (1 + cfg["vat"][f])
        print(f"Model-implied current under-recovery, Delhi {f}: Rs{C - cfg[f + '_rsp0']:.1f}/l (not an official figure)")


if __name__ == "__main__":
    main()
