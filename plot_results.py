#!/usr/bin/env python3
"""Charts for IOGPM outputs (reads outputs/*.csv). Palette: validated reference categorical order (light mode)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

m = pd.read_csv(os.path.join(OUT, "forecast_monthly.csv"), parse_dates=["date"])
W = "Probability-weighted"


def style(ax, title, ylab):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, length=0)


def fan(ax, var, color, label=None, scen=W):
    d = m[(m.scenario == scen) & (m.variable == var)]
    ax.fill_between(d.date, d.p10, d.p90, color=color, alpha=0.16, lw=0)
    ax.plot(d.date, d.p50, color=color, lw=2, label=label)
    return d


# ---------------- Figure 1: probability-weighted outlook ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
ax = axs[0, 0]
d = fan(ax, "brent", CAT[0])
ax.scatter([pd.Timestamp("2026-12-01"), pd.Timestamp("2027-10-01")], [99, 77], color=CAT[1], s=40, zorder=5)
ax.text(pd.Timestamp("2031-06-01"), 62, "orange dots: EIA STEO Nov-26 forecast\n\\$99 (Dec-26) and \\$77 (Q4-27)",
        fontsize=8, color=INK2)
style(ax, "Brent crude  -  median and P10-P90 band", "US$/bbl (nominal)")

ax = axs[0, 1]
fan(ax, "fx", CAT[0])
style(ax, "USD/INR  -  median and P10-P90 band", "INR per US$")

ax = axs[1, 0]
dp = fan(ax, "petrol", CAT[0])
dd = fan(ax, "diesel", CAT[1])
ax.text(dp.date.iloc[-1], dp.p50.iloc[-1] - 5, " Petrol", color=INK2, fontsize=9, va="center")
ax.text(dd.date.iloc[-1], dd.p50.iloc[-1] + 4, " Diesel", color=INK2, fontsize=9, va="center")
ax.axhline(102.12, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.text(pd.Timestamp("2033-06-01"), 84, "dashed line: Delhi petrol today, Rs102.12", fontsize=8, color=MUTED,
        bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
style(ax, "Delhi pump prices  -  nominal, median and P10-P90", "Rs per litre")

ax = axs[1, 1]
for var, c, lab in [("lng", CAT[0], "India landed LNG"), ("apm", CAT[1], "APM domestic gas"), ("jkm", CAT[2], "JKM spot (Asia)")]:
    d = m[(m.scenario == W) & (m.variable == var)]
    ax.plot(d.date, d.p50, color=c, lw=2)
    off = {"lng": 0.9, "jkm": -0.9, "apm": 0.0}[var]
    ax.text(d.date.iloc[-1], d.p50.iloc[-1] + off, " " + lab, color=INK2, fontsize=9, va="center")
style(ax, "Gas prices  -  probability-weighted medians", "US$/MMBtu (nominal)")
for a in axs.ravel():
    a.set_xlim(pd.Timestamp("2026-12-01"), pd.Timestamp("2042-06-01"))
fig.suptitle("India oil & gas price outlook to 2040  -  probability-weighted across six geopolitical scenarios",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.text(0.01, 0.005, "Model state as of ~20-23 Nov 2026. Bands = 10th-90th percentile of 3,000 paths per scenario. "
         "Illustrative scenario model, not a point forecast.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(os.path.join(OUT, "fig1_outlook.png"), dpi=150)

# ---------------- Figure 2: scenarios ----------------
scen = [s for s in m.scenario.unique() if s != W]
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, var, ttl, yl in [(axs[0], "brent", "Brent crude  -  median by scenario", "US$/bbl (nominal)"),
                         (axs[1], "petrol", "Delhi petrol  -  median by scenario", "Rs per litre"),
                         (axs[2], "lng", "India landed LNG  -  median by scenario", "US$/MMBtu")]:
    for i, s in enumerate(scen):
        d = m[(m.scenario == s) & (m.variable == var)]
        ax.plot(d.date, d.p50, color=CAT[i], lw=1.8, label=s)
    style(ax, ttl, yl)
    ax.set_xlim(pd.Timestamp("2026-12-01"), pd.Timestamp("2040-12-01"))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Scenario medians  -  how the geopolitical path changes India's price outlook", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig2_scenarios.png"), dpi=150)
print("saved")
