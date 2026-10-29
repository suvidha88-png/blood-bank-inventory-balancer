"""
Synthetic blood bank inventory data for the Blood Bank Inventory Balancer project.

Simulates 4 blood banks in one city over 365 days (Oct 2025 - Sep 2026), unit by unit:

    whole-blood donation (daily walk-ins + occasional large camps)
      -> testing (reactive / damaged units discarded)
      -> component separation: PRBC, FFP, random-donor platelets (capacity-limited), cryo
      -> storage until expiry (PRBC 35 d, platelets 5 d, FFP and cryo 365 d)
      -> issue against daily demand, oldest unit first (FIFO)
      -> some issued units come back unused (in time -> restocked; late -> discarded)
      -> expired units discarded

Built-in patterns the product should be able to find:
    * Platelets have the highest discard rate (5-day shelf life).
    * Platelet demand rises in the dengue season (Aug-Nov).
    * Large donation camps create PRBC/platelet gluts followed by expiry.
    * Bank A (tertiary, oncology-heavy) runs short of platelets while Bank B has surplus,
      so inter-bank transfers could cut both shortages and waste.

All data is synthetic; bank names and parameters are illustrative, loosely calibrated to
published Indian blood bank discard studies. Blood group shares are approximate.

Outputs:
    bb_units.csv      one row per component unit with its outcome
    bb_daily.csv      bank x date x component x blood group: stock, demand, issues, discards
    bb_donations.csv  bank x date: donations collected, camp flag, units failing testing
"""
import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)

START = date(2025, 10, 1)
DAYS = 365
WARMUP = 30  # simulated but not reported, so the year starts with realistic stock

SHELF_DAYS = {"PRBC": 35, "Platelets": 5, "FFP": 365, "Cryo": 365}
NEAR_EXPIRY_DAYS = {"PRBC": 7, "Platelets": 2, "FFP": 30, "Cryo": 30}
GROUPS = {"O+": 0.35, "B+": 0.31, "A+": 0.21, "AB+": 0.07,
          "O-": 0.02, "B-": 0.02, "A-": 0.015, "AB-": 0.005}
G_NAMES, G_PROBS = list(GROUPS), list(GROUPS.values())

# supply = donation scale, demand = demand scale, plt_bias = extra platelet demand
BANKS = {
    "Bank A (tertiary hospital)": {"supply": 1.10, "demand": 1.15, "plt_bias": 1.5, "plt_cap": 30},
    "Bank B (standalone centre)": {"supply": 0.60, "demand": 0.45, "plt_bias": 0.8, "plt_cap": 30},
    "Bank C (district hospital)": {"supply": 0.55, "demand": 0.55, "plt_bias": 1.0, "plt_cap": 12},
    "Bank D (private hospital)": {"supply": 0.75, "demand": 0.70, "plt_bias": 1.1, "plt_cap": 20},
}

BASE_DONATIONS = 30          # whole-blood units/day at supply scale 1
BASE_DEMAND = {"PRBC": 34, "Platelets": 9, "FFP": 8, "Cryo": 0.8}
# Banks prepare platelets to keep ~N days of their *usual* demand in stock. The target
# ignores the dengue season on purpose: that blind spot is what a forecast would fix.
PLT_TARGET_DAYS = 3.0
TTI_REACTIVE = 0.015         # donation discarded after testing
PROCESSING_LOSS = 0.01       # per component: leak, clot, low volume
RETURN_IN_TIME = 0.04        # issued, unused, back within 30 min -> restocked
RETURN_LATE = 0.015          # issued, unused, back late -> discarded


def donation_factor(d):
    f = 1.0
    if d.month in (4, 5, 6):          # summer dip (heat, exams, holidays)
        f *= 0.75
    if d.weekday() == 6:
        f *= 0.6
    return f


def dengue_factor(d):
    doy = d.timetuple().tm_yday
    return 1 + 1.3 * np.exp(-((doy - 265) / 35) ** 2)   # peak late September


units = []                      # list of dicts; index = unit id
stock = {}                      # (bank, comp, group) -> sorted [(expiry_ordinal, uid)]
daily, donations = [], []
pending = []                    # units collected yesterday, released after testing


def new_unit(bank, comp, grp, collected):
    uid = len(units)
    units.append({"unit_id": f"U{uid:06d}", "bank": bank, "component": comp,
                  "blood_group": grp, "collection_date": collected,
                  "expiry_date": collected + timedelta(days=SHELF_DAYS[comp]),
                  "outcome": "In stock", "outcome_date": None, "discard_reason": ""})
    return uid


def close(uid, outcome, d, reason=""):
    units[uid].update(outcome=outcome, outcome_date=d, discard_reason=reason)


def take_unit(bank, comp, grp):
    """FIFO issue with simplified compatibility fallbacks:
    PRBC -> group O of the same Rh; FFP -> AB of the same Rh;
    platelets -> any ABO group of the same Rh, then any group (common practice when short).
    """
    options = [grp]
    rh = grp[-1]
    if comp == "PRBC" and not grp.startswith("O"):
        options.append("O" + rh)
    elif comp == "FFP" and not grp.startswith("AB"):
        options.append("AB" + rh)
    elif comp == "Platelets":
        same_rh = [g for g in G_NAMES if g.endswith(rh) and g != grp]
        other_rh = [g for g in G_NAMES if not g.endswith(rh)]
        options += same_rh + other_rh
    for g in options:
        lst = stock.get((bank, comp, g))
        if lst:
            return lst.pop(0)[1]
    return None


sim_start = START - timedelta(days=WARMUP)
for day_i in range(DAYS + WARMUP):
    d = sim_start + timedelta(days=day_i)
    report = d >= START
    rec = {}

    def r(bank, comp, grp):
        key = (bank, comp, grp)
        if key not in rec:
            rec[key] = dict(received=0, demand=0, issued=0, unmet=0, expired=0,
                            returned_restocked=0, discarded_return=0, discarded_testing=0)
        return rec[key]

    # 1. morning: discard expired units
    for (bank, comp, grp), lst in stock.items():
        while lst and lst[0][0] <= d.toordinal():
            _, uid = lst.pop(0)
            close(uid, "Discarded", d, "Expired")
            r(bank, comp, grp)["expired"] += 1

    # 2. release yesterday's tested units into stock
    for uid in pending:
        u = units[uid]
        key = (u["bank"], u["component"], u["blood_group"])
        bisect.insort(stock.setdefault(key, []), (u["expiry_date"].toordinal(), uid))
        r(*key)["received"] += 1
    pending = []

    opening = {k: len(v) for k, v in stock.items()}

    for bank, cfg in BANKS.items():
        # 3. demand and FIFO issue
        wk = 0.8 if d.weekday() >= 5 else 1.0
        for comp, base in BASE_DEMAND.items():
            lam = base * cfg["demand"] * wk
            if comp == "Platelets":
                lam *= cfg["plt_bias"] * dengue_factor(d)
            n = rng.poisson(lam)
            for grp in rng.choice(G_NAMES, size=n, p=G_PROBS):
                rr = r(bank, comp, grp)
                rr["demand"] += 1
                uid = take_unit(bank, comp, grp)
                if uid is None:
                    rr["unmet"] += 1
                    continue
                rr["issued"] += 1
                ug = units[uid]["blood_group"]
                x = rng.random()
                if x < RETURN_IN_TIME and comp in ("PRBC", "Platelets"):
                    bisect.insort(stock[(bank, comp, ug)],
                                  (units[uid]["expiry_date"].toordinal(), uid))
                    r(bank, comp, ug)["returned_restocked"] += 1
                elif x < RETURN_IN_TIME + RETURN_LATE and comp in ("PRBC", "Platelets"):
                    close(uid, "Discarded", d, "Returned outside cold chain")
                    r(bank, comp, ug)["discarded_return"] += 1
                else:
                    close(uid, "Issued", d)

        # 4. today's donations -> testing -> separation (released tomorrow)
        camp = rng.random() < 0.05
        n_don = rng.poisson(BASE_DONATIONS * cfg["supply"] * donation_factor(d))
        if camp:
            n_don += rng.poisson(90 * cfg["supply"])
        plt_target = BASE_DEMAND["Platelets"] * cfg["demand"] * cfg["plt_bias"] * PLT_TARGET_DAYS
        plt_stock = sum(len(stock.get((bank, "Platelets", g), [])) for g in G_NAMES)
        # prepare only up to target, limited by separation capacity in the time window
        plt_left = int(min(cfg["plt_cap"], max(0, round(plt_target - plt_stock))))
        n_tti = 0
        for grp in rng.choice(G_NAMES, size=n_don, p=G_PROBS):
            comps = ["PRBC", "FFP"]
            if plt_left > 0 and rng.random() < 0.85:
                comps.append("Platelets")
                plt_left -= 1
            if rng.random() < 0.04:
                comps.append("Cryo")
            reactive = rng.random() < TTI_REACTIVE
            n_tti += reactive
            for comp in comps:
                uid = new_unit(bank, comp, grp, d)
                if reactive:
                    close(uid, "Discarded", d, "Reactive on testing")
                    r(bank, comp, grp)["discarded_testing"] += 1
                elif rng.random() < PROCESSING_LOSS:
                    close(uid, "Discarded", d, "Processing loss")
                    r(bank, comp, grp)["discarded_testing"] += 1
                else:
                    pending.append(uid)
        if report:
            donations.append({"bank": bank, "date": d, "whole_blood_collected": n_don,
                              "camp_day": camp, "donations_reactive": n_tti})

    if not report:
        continue
    for bank in BANKS:
        for comp in SHELF_DAYS:
            for grp in G_NAMES:
                key = (bank, comp, grp)
                lst = stock.get(key, [])
                rr = rec.get(key) or r(*key)
                limit = (d + timedelta(days=NEAR_EXPIRY_DAYS[comp])).toordinal()
                daily.append({
                    "bank": bank, "date": d, "component": comp, "blood_group": grp,
                    "opening_stock": opening.get(key, 0),
                    "received": rr["received"], "demand": rr["demand"],
                    "issued": rr["issued"], "unmet_demand": rr["unmet"],
                    "returned_restocked": rr["returned_restocked"],
                    "discarded_expired": rr["expired"],
                    "discarded_return": rr["discarded_return"],
                    "discarded_testing": rr["discarded_testing"],
                    "closing_stock": len(lst),
                    "near_expiry_units": sum(1 for e, _ in lst if e <= limit),
                })

u = pd.DataFrame(units)
u = u[u["collection_date"] >= START - timedelta(days=WARMUP)]
u.to_csv("bb_units.csv", index=False)
pd.DataFrame(daily).to_csv("bb_daily.csv", index=False)
pd.DataFrame(donations).to_csv("bb_donations.csv", index=False)
print(f"Wrote {len(u)} units, {len(daily)} daily rows, {len(donations)} donation rows")
