"""
Platelet policy lab: simulation engine.

Simulates platelets at the same 4 blood banks as generate_bloodbank_data.py, using the same
rules, but for the NEXT year (1 Oct 2026 - 30 Sep 2027) with fresh randomness. That gives you
a fair test: learn from last year's data (bb_daily.csv), test your rule on a year it has
never seen.

You control three levers:

    platelet_target(bank, today, recent_demand) -> int
        How many platelet units the bank aims to hold. Each day, the bank prepares
        platelets from donations to top stock up to this target (if capacity allows).
        recent_demand = list of the bank's daily platelet demand so far (most recent last).

    capacity_plan(bank, today) -> int
        Extra platelet units/day the bank can prepare (e.g., donor drives, extra shifts).
        Default: 0.

    transfers=True
        Each morning, units on their last usable day that the holding bank is unlikely to
        use go to the bank furthest below its target.

Usage:
    from bb_sim import simulate
    result = simulate(platelet_target=my_rule)
    result["summary"]   # one-row-per-bank table of waste and shortages
"""
import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd

SHELF_DAYS = 5
GROUPS = {"O+": 0.35, "B+": 0.31, "A+": 0.21, "AB+": 0.07,
          "O-": 0.02, "B-": 0.02, "A-": 0.015, "AB-": 0.005}
G_NAMES, G_PROBS = list(GROUPS), list(GROUPS.values())

BANKS = {
    "Bank A (tertiary hospital)": {"supply": 1.10, "demand": 1.15, "plt_bias": 1.5, "plt_cap": 30},
    "Bank B (standalone centre)": {"supply": 0.60, "demand": 0.45, "plt_bias": 0.8, "plt_cap": 30},
    "Bank C (district hospital)": {"supply": 0.55, "demand": 0.55, "plt_bias": 1.0, "plt_cap": 12},
    "Bank D (private hospital)": {"supply": 0.75, "demand": 0.70, "plt_bias": 1.1, "plt_cap": 20},
}
BASE_DONATIONS = 30
BASE_PLT_DEMAND = 9
TTI_REACTIVE, PROCESSING_LOSS = 0.015, 0.01
RETURN_IN_TIME, RETURN_LATE = 0.04, 0.015

# A bank's "usual" platelet demand per day (what the current fixed rule plans around).
USUAL_DEMAND = {b: BASE_PLT_DEMAND * c["demand"] * c["plt_bias"] for b, c in BANKS.items()}


def current_rule(bank, today, recent_demand):
    """What banks do today: keep 3 days of USUAL demand, all year round."""
    return round(USUAL_DEMAND[bank] * 3)


def no_extra_capacity(bank, today):
    return 0


def _donation_factor(d):
    f = 0.75 if d.month in (4, 5, 6) else 1.0
    return f * (0.6 if d.weekday() == 6 else 1.0)


def _dengue_factor(d):   # hidden "truth" — your forecast has to learn it from history
    doy = d.timetuple().tm_yday
    return 1 + 1.3 * np.exp(-((doy - 265) / 35) ** 2)


def _take(stock_b, grp):
    rh = grp[-1]
    order = [grp] + [g for g in G_NAMES if g.endswith(rh) and g != grp] \
                  + [g for g in G_NAMES if not g.endswith(rh)]
    for g in order:
        if stock_b[g]:
            return g, stock_b[g].pop(0)
    return None, None


def simulate(platelet_target=current_rule, capacity_plan=no_extra_capacity, transfers=False,
             start=date(2026, 10, 1), days=365, seed=2027, warmup=30):
    # Separate random streams so every rule faces the same patients and donors.
    rng_dem, rng_don, rng_proc = (np.random.default_rng([seed, k]) for k in range(3))
    stock = {b: {g: [] for g in G_NAMES} for b in BANKS}   # sorted list of expiry ordinals
    pending = []                                            # (bank, group, expiry) ready tomorrow
    history = {b: [] for b in BANKS}                        # daily platelet demand
    rows = []
    n_transferred = 0

    sim_start = start - timedelta(days=warmup)
    for i in range(days + warmup):
        d = sim_start + timedelta(days=i)
        today = d.toordinal()
        rec = {b: dict(produced=0, demand=0, issued=0, unmet=0, expired=0,
                       discarded_other=0, transferred_in=0, transferred_out=0) for b in BANKS}

        # 1. discard expired units
        for b in BANKS:
            for g in G_NAMES:
                lst = stock[b][g]
                while lst and lst[0] <= today:
                    lst.pop(0)
                    rec[b]["expired"] += 1

        # 2. release yesterday's tested units
        for b, g, exp in pending:
            bisect.insort(stock[b][g], exp)
        pending = []

        targets = {b: max(0, int(platelet_target(b, d, list(history[b])))) for b in BANKS}

        # 3. optional transfers: units on their LAST usable day, which the bank holding them
        #    is unlikely to use today, go to the bank furthest below its target.
        if transfers:
            level = {b: sum(len(v) for v in stock[b].values()) for b in BANKS}
            need = {b: (np.mean(history[b][-7:]) if history[b] else USUAL_DEMAND[b]) for b in BANKS}
            gap = {b: targets[b] - level[b] for b in BANKS}
            last_day = today + 1
            for gv in BANKS:
                spare = int(level[gv] - need[gv])          # units beyond today's likely use
                for g in G_NAMES:
                    while spare > 0 and stock[gv][g] and stock[gv][g][0] == last_day:
                        takers = [t for t in BANKS if t != gv and gap[t] > 0]
                        if not takers:
                            break
                        t = max(takers, key=lambda x: gap[x])
                        bisect.insort(stock[t][g], stock[gv][g].pop(0))
                        gap[t] -= 1; spare -= 1
                        rec[t]["transferred_in"] += 1; rec[gv]["transferred_out"] += 1
                        n_transferred += d >= start

        for b, cfg in BANKS.items():
            # 4. demand, oldest unit first
            wk = 0.8 if d.weekday() >= 5 else 1.0
            n = rng_dem.poisson(BASE_PLT_DEMAND * cfg["demand"] * cfg["plt_bias"] * wk * _dengue_factor(d))
            history[b].append(n)
            for grp in rng_dem.choice(G_NAMES, size=n, p=G_PROBS):
                rec[b]["demand"] += 1
                g, exp = _take(stock[b], grp)
                if g is None:
                    rec[b]["unmet"] += 1
                    continue
                rec[b]["issued"] += 1
                x = rng_proc.random()
                if x < RETURN_IN_TIME:
                    bisect.insort(stock[b][g], exp)
                elif x < RETURN_IN_TIME + RETURN_LATE:
                    rec[b]["discarded_other"] += 1

            # 5. donations -> prepare platelets up to target, within capacity
            camp = rng_don.random() < 0.05
            n_don = rng_don.poisson(BASE_DONATIONS * cfg["supply"] * _donation_factor(d))
            if camp:
                n_don += rng_don.poisson(90 * cfg["supply"])
            level = sum(len(v) for v in stock[b].values())
            cap = cfg["plt_cap"] + max(0, int(capacity_plan(b, d)))
            to_make = int(min(cap, max(0, targets[b] - level)))
            for grp in rng_don.choice(G_NAMES, size=n_don, p=G_PROBS):
                if to_make <= 0:
                    break
                if rng_proc.random() >= 0.85:
                    continue
                to_make -= 1
                if rng_proc.random() < TTI_REACTIVE + PROCESSING_LOSS:
                    rec[b]["discarded_other"] += 1
                    rec[b]["produced"] += 1
                    continue
                rec[b]["produced"] += 1
                pending.append((b, grp, today + SHELF_DAYS))

        if d >= start:
            for b in BANKS:
                rows.append({"bank": b, "date": d, "target": targets[b],
                             "closing_stock": sum(len(v) for v in stock[b].values()), **rec[b]})

    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])
    return {"daily": daily, "summary": summarise(daily), "monthly": monthly(daily),
            "transfers": n_transferred}


def summarise(daily):
    s = daily.groupby("bank")[["demand", "unmet", "produced", "expired", "discarded_other",
                               "transferred_in", "transferred_out"]].sum()
    tot = s.sum().rename("ALL BANKS")
    s = pd.concat([s, tot.to_frame().T])
    s["unmet_%"] = (s["unmet"] / s["demand"] * 100).round(1)
    s["waste_%"] = ((s["expired"] + s["discarded_other"]) / s["produced"] * 100).round(1)
    return s[["demand", "unmet", "unmet_%", "produced", "expired", "waste_%",
              "transferred_in", "transferred_out"]]


def monthly(daily):
    m = daily.groupby(daily["date"].dt.to_period("M"))[["demand", "unmet", "expired"]].sum()
    m.index = m.index.astype(str)
    return m


def scorecard(name, result):
    """One-line result for comparing rules side by side."""
    a = result["summary"].loc["ALL BANKS"]
    return {"rule": name, "unmet_%": a["unmet_%"], "waste_%": a["waste_%"],
            "units_unmet": int(a["unmet"]), "units_expired": int(a["expired"]),
            "units_produced": int(a["produced"]), "transfers": result["transfers"]}
