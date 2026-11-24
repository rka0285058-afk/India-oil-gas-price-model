# India Oil & Gas Price Model (IOGPM v1)

Dynamic, scenario-driven Monte Carlo model of India's crude, gas and pump prices, Dec-2026 to Dec-2040.
State as of ~20-23 Nov 2026. Run: `python3 india_oil_gas_price_model.py` then `python3 plot_results.py`.

## Structure
1. **Geopolitical regimes** (Markov chain): Normalisation, Simmering disruption (today), Acute escalation, Structural glut.
2. **Brent**: regime-switching mean-reverting log price (real targets $72 / $92 / $125 / $58).
3. **USD/INR**: 2.5%/yr trend + oil beta 0.14 (calibrated from RBI rates vs PPAC/EIA Brent: Apr-25 to Apr-26 gives 0.153, Apr-26 to Nov-26 gives 0.125).
4. **Indian Crude Basket**: Brent x AR(1) basis - Russian-discount overlay.
5. **Gas**: Henry Hub, JKM (fundamentals + geopolitical premium), India LNG portfolio (45% oil-linked, 15% HH-linked, 40% spot - ASSUMPTION), APM = clip(10% x prior-month ICB, $4 floor, cap $7.00 rising $0.25/yr to $8.50).
6. **Pump prices (Delhi)**: cost-reflective price = (crude + crack + other + excise) x (1+VAT). Reaction function: hikes when under-recovery persists, excise cut in spikes, excise restored first on over-recovery, pump cuts only after past losses are recouped.
7. **Dynamic updating**: `--obs data/observations.csv` resets the state; with >=12 monthly Brent rows a Hamilton filter updates regime probabilities.

## Scenarios and weights (subjective priors, change with `--weights`)
Base 30%, Diplomatic settlement 20%, Prolonged standoff 20%, Structural glut 12%, Russia sanctions bite 10%, Escalation 8%.

## Data used
- PPAC (via press reports of PPAC releases): ICB monthly (Oct-25 $65.08, Mar-26 $113.49, Apr-26 $114.48, Jun-26 $86.31, Aug-26 $90.19, Sep-26 $116.09); Apr-26 Brent $120.55; Delhi RSP and ministry under-recovery figures (1-Apr-26: petrol Rs24.40, diesel Rs104.99); APM cap $7.00 (31-Mar-26 notification); ceiling gas $8.90 (Apr-Sep 26).
- MoF Notification 11/2026-CE (26-Mar-26): excise cut of Rs10/l.
- RBI/FBIL reference rates: Rs93.35 (19-Nov-26), record low Rs96.965 (May-26).
- EIA STEO Nov-2026: Brent ~$99-104 Nov/Dec, $86 avg 2027, $77 Q4-27; Henry Hub $2.87 winter.
- IEA WEO 2026 (India crude output -20% by 2035, demand +1.3 mb/d by 2035); OPEC WOO 2026 (India +8.1 mb/d by 2050).
- Market data (not official): Brent settle ~$92 (trade press), JKM ~$30 (Platts/JOGMEC).

## Assumptions to replace with official series
Freight + dealer commission (Rs7.4 petrol / Rs5.7 diesel), Delhi VAT (19.4% / 16.75%), post-cut excise levels (Rs11.9 / Rs7.8), starting diesel crack ($50) and petrol crack ($18), cumulative OMC loss ledger, LNG portfolio weights, APM cap path, scenario weights. Pull these from PPAC price build-up, tariff and LNG-import tables.

## Known limits
No out-of-sample backtest (only ~8 months of crisis data). Model Q4-27 Brent ($86) sits above EIA ($77): the model has no inventory-rebuild glide path. Seasonality, LPG/CNG/PNG/ATF, state-level VAT, ceiling-price gas and volume (supply-shortage) effects are not modelled.

## LPG and CNG (added 24-Nov-2026)
- **LPG (Delhi 14.2 kg domestic)**: Saudi CP ($/t) = 250 + 4.6 x 3m-avg Brent + 2 x gas premium (ASSUMPTION, fits Feb/Apr/Jun-26 CP of $543/$775/$790 within ~$40). Cost-reflective price = (CP x FX x 14.2 + Rs460 residual) x 1.05 GST. Price starts at Rs942 (since 7-Jun-26; Rs913 before; +Rs60 on 7-Mar-26), moves in Rs30 hikes when under-recovery > Rs250, cuts only after the ledger is recouped. Rs460 is a residual calibrated to the government's ~Rs703/cylinder pre-June loss; replace with PPAC's price build-up.
- **CNG (Delhi, IGL)**: gas basket = 45% APM (ASSUMPTION) + 55% landed LNG; x 0.0475 MMBtu/kg x FX + Rs26/kg other costs (ASSUMPTION), + 5% VAT. Starts Rs86.98/kg (reported 11-Nov-26); Rs2 hikes when cost exceeds price by Rs2.5, Rs1 cuts when price exceeds cost by Rs5.
- Uses a separate random stream, so existing Brent, FX, petrol, diesel and gas results are unchanged.

## Repository layout
- `india_oil_gas_price_model.py` - the model (`python3 india_oil_gas_price_model.py`)
- `plot_results.py` - static charts into `outputs/`
- `export_dashboard_data.py` - exports quantile grids and sample paths to `dashboard/data.json`
- `dashboard/index.html` + `data.json` - interactive dashboard (serve the folder, e.g. `python3 -m http.server`, and open `index.html`)
- `data/observations_template.csv` - template for the `--obs` dynamic update
- `outputs/` - monthly and annual forecasts, regime and risk probabilities, charts

Requires Python 3 with numpy, pandas and matplotlib. Illustrative scenario model, not a point forecast or investment advice.
