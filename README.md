# Blood Bank Inventory Balancer

An AI-assisted inventory product for blood banks: an ML demand forecast feeds an inventory
decision engine, tested in a simulation. Built as part of my AI product management portfolio
in healthcare operations.

> **Status:** in progress (started 28 Oct 2026). All data is synthetic, calibrated to published
> Indian blood bank studies. No real patient or hospital data is used.

## The problem

Platelets last only **5 days**. Blood banks plan platelet stock with one fixed number all year, so:

- In normal months, bags sit too long and **expire** (waste).
- In the dengue season (Aug–Oct), demand roughly doubles and stock **runs out** (shortage).

India discarded 17.4 lakh blood components in 2025, mainly through expiry, and platelets are
the most affected ([OC Academy](https://www.ocacademy.in/blogs/optimizing-blood-inventory-management-indian-healthcare)).
Platelet shortages can delay chemotherapy for cancer patients
([The Week, Oct 2026](https://www.theweek.in/news/health/2026/10/12/cancer-patients-need-platelets-every-few-days-how-delayed-transfusions-can-disrupt-treatment.html)).

## Users

| User | Need |
| --- | --- |
| Blood bank officer | How many platelets to prepare today; which units are about to expire |
| City / network coordinator | Which bank has spare units when another is short |
| Hospital operations head | Waste and shortage trends over time |

## Approach: three layers

| Layer | Question | Technique |
| --- | --- | --- |
| Prediction | How much demand next 14 days? | Time-series forecasting (ML) |
| Decision | How much to prepare, keep or move? | Inventory rules, capacity planning, transfers |
| Product | Will staff trust and act on it? | Dashboard, alerts, human-approved agent actions |

## Results so far

_To be filled in after running the Policy Lab._

| Rule | Unmet demand % | Waste % |
| --- | --- | --- |
| Current fixed rule | | |
| My recommended rule | | |

## Files

| File | What it is |
| --- | --- |
| `platelet_policy_lab.ipynb` | Notebook: test stock rules and compare waste vs shortage |
| `bb_sim.py` | Simulation engine for next year's platelets (used by the notebook) |
| `generate_bloodbank_data.py` | Generates last year's synthetic data (the three CSVs) |
| `bb_daily.csv` | Daily stock, demand, issues and discards by bank, component and blood group |
| `bb_units.csv` | Life of every unit: collected, expired or issued, and why discarded |
| `bb_donations.csv` | Daily donations per bank, including camp days |
| `DATA_GUIDE.md` | Column definitions, built-in patterns and simplifications |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook platelet_policy_lab.ipynb
```

To regenerate the data: `python generate_bloodbank_data.py`

## Roadmap

- [x] Synthetic data generator and simulation engine
- [ ] Policy Lab results and recommended rule
- [ ] Forecasting comparison: monthly average vs SARIMA/Prophet vs gradient boosting
- [ ] One-page product brief
- [ ] Streamlit dashboard
- [ ] Rule-based alerts (low stock, near expiry, blood received)
- [ ] Transfer-coordination agent with human approval

## Limitations

Synthetic data and simplified compatibility rules; results show how the product logic behaves,
not real-world impact. A real evaluation would need a pilot with one blood bank.
