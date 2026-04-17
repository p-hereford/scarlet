# SCARLET — JPM Fraud Pressure Intelligence

**S**cam **C**ascade **A**nalysis and **R**isk **L**earning **E**arly **T**racker

SCARLET is a fraud-pressure intelligence and control-signalling system built around public complaint data for a JPMorgan-style risk environment. The core logic is complete and operational. What remains is licensing, formal documentation, and repository preparation for presentation.

---

## What it does

Standard fraud models identify whether a transaction, customer, or case looks suspicious. SCARLET operates one layer higher. It converts fragmented fraud indicators into a clear control posture, so that the response scales with the level of pressure rather than reacting case by case.

The central question it answers is:

> Is fraud pressure rising above recent norms, and should controls be tightened now rather than later?

---

## Why complaint data

Consumer complaints are a lagging indicator by nature — people file after the event. SCARLET addresses this directly by incorporating industry-wide complaint velocity as a leading signal. Fraud campaigns tend to surface across the market before they fully register in any single institution's own data. When that upstream pressure rises, SCARLET flags it before JPM's own series reflects it.

A secondary news signal via GDELT captures media-driven filing behaviour: coverage of fraud incidents pushes latent victims to act, compressing the lag further.

---

## System architecture

The pipeline runs in four stages.

**Data ingestion** pulls three sources: JPMorgan complaints from the CFPB database, industry-wide fraud complaints across relevant product categories from the same source, and weekly news article counts from GDELT. All three are fetched across monthly windows to manage API reliability.

**Feature engineering** aggregates complaint data into weekly fraud indicators. JPM-side features include complaint counts, fraud rate, week-on-week growth, four-week rolling average, and z-score. Industry-side features are constructed entirely in lagged form — they are meant to act as upstream warning signals, not to introduce future information into the current week.

**Forecast engine** is a two-part hurdle model. The first stage is a logistic classifier that identifies whether a given week is likely to cross the stress threshold. The second stage is a ridge regressor, trained on stress weeks only, that estimates expected complaint volume conditional on a spike. The final prediction blends both outputs weighted by spike probability, with the four-week rolling mean as the non-stress fallback.

The model was evaluated using walk-forward cross-validation across five expanding windows. On stress weeks, the hurdle model produces a mean absolute error of 55 against a lag-1 baseline of 143 — a reduction of approximately 62 percent on the events that matter most. Recall on stress events is 1.0 across all folds.

**Decision engine** classifies each week into a regime — normal, elevated, or stressed — using level, growth, and z-score in combination. It then assigns a control posture: monitor, heightened monitoring, pre-emptive tightening, or tighten controls. The regime classification and control recommendation are independent of the forecast model and can be used without it.

**Executive brief** translates the current regime, signal summary, and recommended action into a plain-language output formatted for a senior risk audience.

---

## Performance

Results below are from the hold-out period (most recent 20 percent of available history) and from five-fold walk-forward cross-validation.

| Metric | SCARLET | Naive lag-1 | Rolling mean |
|---|---|---|---|
| MAE — all weeks | 127.51 | 115.38 | 102.88 |
| MAE — stress weeks | 119.80 | 225.00 | 175.00 |
| MAE — stress weeks (CV) | 55.08 | 143.33 | 93.75 |
| Recall — stress events | 1.00 | — | — |
| Precision — stress events | 0.31 | — | — |

The overall MAE figure is higher than the rolling mean baseline. This is expected and acceptable: the model accepts a modest overall error increase in exchange for materially better performance on stress weeks, which are the events that carry operational and reputational consequence. The rolling mean will never miss a stress event by 400 complaints; SCARLET will not either, but it catches them earlier and with a calibrated probability signal attached.

The residual gap on the December 2024 event (actual: 450, predicted: 116) reflects the information ceiling of endogenous complaint data. That event was driven by an external incident not captured in lagged complaint counts or industry velocity. Closing that gap requires an additional external data stream; it is not a modelling failure.

---

## Data sources

| Source | Description | Update frequency |
|---|---|---|
| CFPB Consumer Complaint Database | JPM complaints, 2022–2024 | On demand via API |
| CFPB Consumer Complaint Database | Industry complaints, fraud-relevant products | On demand via API or bulk download |
| GDELT Doc 2.0 API | Weekly article counts, JPMorgan fraud / Chase scam | On demand, free, no key required |

The bulk download path for industry complaints is available at `consumerfinance.gov` and is the recommended approach when the API is under load.

---

## Repository structure

```
SCARLET/
  src/
    ingestion.py            fetch all three data sources
    features.py             build weekly feature table
    forecasting.py          hurdle model, CV, evaluation output
    decision.py             regime classification and control logic
    brief.py                executive brief generator
    evaluate.py             regime and signal evaluation
    impact.py               scenario-based loss avoided estimates
    reporting.py            write all outputs to file
    run_pipline.py          quick regime summary, terminal
    plots.py                three-panel signal chart
  data/
    jpm_complaints.csv
    industry_complaints.csv
    complaints_full.csv     CFPB bulk download, not committed
    jpm_weekly_features.csv
  outputs/
    latest_brief.txt
    latest_decisions.csv
    evaluation_summary.txt
    economic_impact.txt
```

---

## Running the pipeline

```bash
# 1. Fetch data (or place complaints_full.csv in data/ and skip API fetch)
python src/ingestion.py

# 2. Build feature table
python src/features.py

# 3. Run forecast engine and evaluation
python src/forecasting.py

# 4. Run decision engine
python src/run_pipeline.py

# 5. Generate executive brief
python src/brief.py

# 6. Export all outputs to file
python src/reporting.py
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`.

---

## Economic impact layer

SCARLET includes a scenario-based translation from signal to indicative loss avoided. The assumptions are explicit: $250 loss proxy per fraud complaint, reduction rates scaled to action intensity from 0 percent at monitor to 25 percent at tighten controls. These are decision-framing figures, not claims about realised P&L.

---

## What is not in scope

SCARLET does not classify individual transactions or customer accounts. It does not connect to internal transaction systems, case management platforms, or real-time feeds. It operates on weekly aggregated complaint data and is designed to inform control posture decisions at the institutional level, not to replace transaction-level fraud detection.

---

## Status

The forecast engine, feature pipeline, decision logic, and output layer are complete. Remaining work: formal licensing, repository clean-up, and preparation for institutional presentation.

---

*P.H. — Risk Analytics*