"""JevBench-style hard-reasoning synthetic data (plan §2: ~10k policies).

Programmatic decision items with ground truth by construction: policy
compliance, exact probability, date arithmetic, limitation periods, and
fact-presence (anti-hallucination). All entities are invented here; nothing
is copied from JevBench (Jev is closed: no Jev outputs ever in training).

Families (share of n):
  policy_choice  40%  fictional policy doc, decisive clause + distractors
  probability    25%  exact hypergeometric/binomial -> score bands or noul
  temporal       20%  date arithmetic -> noul (overdue) or choice (deadline)
  legal          10%  limitation periods incl. tolling -> 3-way choice
  fact            5%  noul: does the doc state X (planted vs absent)

Length discipline: the serve budget keeps ~320 state tokens. Most items
target 60-160 words; policy_long items run 200-300 words with the decisive
clause in the first ~120 words (robustness to trailing noise, still
learnable). Deterministic in (n, seed); ids unique; hash-freezable.
"""

from __future__ import annotations

import datetime as _dt
import random
from fractions import Fraction
from math import comb

from .data import DecisionExample

SOURCE = "synthetic_hard"

# ---------------------------------------------------------------------------
# shared fictional inventory (invented here, seeded picks)


_ORGS = [
    ("Harborlight Freight", "TRADE COMPLIANCE OFFICE"),
    ("Northgate Mutual", "CLAIMS INTAKE"),
    ("Cinderbrook Foods", "SUPPLIER QUALITY"),
    ("Aldermere Clinic", "PATIENT BILLING"),
    ("Foxglove Software", "SUPPORT OPERATIONS"),
    ("Dunmore Logistics", "FLEET DISPATCH"),
    ("Kestrel Air Cargo", "LOAD CONTROL"),
    ("Millbrook Legal Aid", "INTAKE SCREENING"),
    ("Stonebridge Bank", "PAYMENTS REVIEW"),
    ("Wrenfield Telecom", "SERVICE DESK"),
]

_FIRST = ["A.", "J.", "M.", "R.", "S.", "T.", "K.", "D."]
_LAST = ["Abernathy", "Calloway", "Drummond", "Ellery", "Fairbanks", "Godwin", "Holloway", "Inkwell"]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _date(rng: random.Random, y0: int = 2024, y1: int = 2026) -> _dt.date:
    y = rng.randint(y0, y1)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return _dt.date(y, m, d)


def _add_years(d: _dt.date, years: int) -> _dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


# ---------------------------------------------------------------------------
# family 1: policy_choice


_POLICY_RULES = ("license", "refund", "retention", "routing", "discount")


def _policy_item(rng: random.Random, i: int, long: bool) -> DecisionExample:
    org, dept = rng.choice(_ORGS)
    rule = rng.choice(_POLICY_RULES)
    head = f"{org.upper()} - {dept}\n"
    if rule == "license":
        threshold = rng.choice([5000, 10000, 25000])
        value = rng.choice([threshold - rng.randint(500, 2000), threshold + rng.randint(500, 5000)])
        exception = rng.random() < 0.3
        clauses = [
            f"SEC-1: Outbound shipments with declared value over ${threshold:,} require an export license.",
            f"SEC-2: Shipments at or under ${threshold:,} clear without a license.",
        ]
        if exception:
            clauses.append("SEC-3: Medical relief consignments are license-exempt with form MED-9 attached.")
        clauses.append(
            f"SEC-4: License applications take {rng.randint(3, 10)} business days to process."
        )
        shipment_value = value
        med_relief = exception and rng.random() < 0.5
        scenario = (
            f"Shipment S-{rng.randint(1000, 9999)} declares ${shipment_value:,}. "
            + ("It is a medical relief consignment with form MED-9 attached. " if med_relief else "It carries general merchandise. ")
            + "What is required before release?"
        )
        if med_relief and exception:
            gold = "license_exception"
        elif shipment_value > threshold:
            gold = "license_required"
        else:
            gold = "no_license_required"
        options = {
            "license_required": "must obtain an export license",
            "no_license_required": "clears without a license",
            "license_exception": "exempt under the relief rule",
        }
        instructions = "What is required before the shipment is released?"
    elif rule == "refund":
        window = rng.choice([14, 30, 60])
        delivered = _date(rng)
        delta = rng.choice([window - rng.randint(1, 10), window + rng.randint(1, 20)])
        requested = delivered + _dt.timedelta(days=delta)
        fraud = rng.random() < 0.2
        clauses = [
            f"POL-1: Refunds are granted for requests within {window} days of delivery.",
            f"POL-2: Requests after day {window} are denied as expired.",
        ]
        if fraud:
            clauses.append("POL-3: Orders flagged by risk review go to fraud review regardless of timing.")
        flagged = fraud and rng.random() < 0.5
        scenario = (
            f"Order {rng.randint(10000, 99999)} delivered {delivered.isoformat()}; "
            f"refund requested {requested.isoformat()}. "
            + ("Risk review flagged this order. " if flagged else "No risk flags. ")
            + "How should the request be handled?"
        )
        if flagged and fraud:
            gold = "fraud_review"
        elif delta <= window:
            gold = "refund_eligible"
        else:
            gold = "refund_expired"
        options = {
            "refund_eligible": "grant the refund",
            "refund_expired": "deny as out of window",
            "fraud_review": "route to fraud review",
        }
        instructions = "How should the refund request be handled?"
    elif rule == "retention":
        years = rng.choice([3, 5, 7])
        age = rng.choice([years - rng.randint(1, 2), years + rng.randint(0, 3)])
        ref = _dt.date(2026, 9, 1)
        created = _dt.date(ref.year - age, rng.randint(1, 12), rng.randint(1, 28))
        hold = rng.random() < 0.25
        clauses = [
            f"RET-1: Retain customer records for {years} years from creation.",
            f"RET-2: Records older than {years} years are purged on schedule.",
        ]
        if hold:
            clauses.append("RET-3: Records under legal hold are kept until the hold lifts.")
        on_hold = hold and rng.random() < 0.5
        scenario = (
            f"Record R-{rng.randint(1000, 9999)} created {created.isoformat()}; "
            f"review date {ref.isoformat()}. "
            + ("A legal hold notice is active. " if on_hold else "No legal hold. ")
            + "What should happen to the record?"
        )
        if on_hold and hold:
            gold = "legal_hold"
        elif age > years:
            gold = "purge"
        else:
            gold = "retain"
        options = {
            "retain": "keep the record",
            "purge": "purge on schedule",
            "legal_hold": "keep until the hold lifts",
        }
        instructions = "What should happen to the record?"
    elif rule == "routing":
        teams = {
            "payments": "invoices, refunds, billing disputes",
            "outages": "down systems, failed jobs, error spikes",
            "access": "password resets, new accounts, permissions",
        }
        topic = rng.choice(list(teams))
        sev = rng.randint(1, 5)
        keyword = {
            "payments": rng.choice(["duplicate charge", "missing invoice", "refund status"]),
            "outages": rng.choice(["server is down", "job failed overnight", "error rate spiking"]),
            "access": rng.choice(["locked out of account", "new hire setup", "permission change"]),
        }[topic]
        clauses = [
            "RTE-1: Billing and payment issues go to the payments team.",
            "RTE-2: Down systems and failed jobs go to the outages team.",
            "RTE-3: Account and permission requests go to the access team.",
            f"RTE-4: Severity {sev} of 5 logged for this ticket.",
        ]
        scenario = f"Ticket {rng.randint(100, 999)}: customer reports '{keyword}'. Which team owns it?"
        gold = topic
        options = {k: f"route to {k} ({v})" for k, v in teams.items()}
        instructions = "Which team should own this ticket?"
    else:  # discount
        cap = rng.choice([10, 15, 20])
        pct = rng.choice([cap - rng.randint(1, 5), cap + rng.randint(0, 10)])
        clauses = [
            f"FIN-1: Sales staff may discount up to {cap}% without approval.",
            f"FIN-2: Discounts of {cap}% or more need a manager signature.",
        ]
        scenario = (
            f"Quote Q-{rng.randint(1000, 9999)} applies a {pct}% discount. "
            "Can it be sent as-is?"
        )
        gold = "manager_approval" if pct >= cap else "auto_approved"
        options = {
            "auto_approved": "send without approval",
            "manager_approval": "needs a manager signature",
        }
        instructions = "Can the quote be sent as-is?"
    # filler clauses for length (distractors + boilerplate, never decisive)
    fillers = [
        "ADM-1: All decisions are logged with timestamp and reviewer id.",
        "ADM-2: Appeals must be filed within 10 business days.",
        "ADM-3: This policy is reviewed annually each January.",
        "ADM-4: Questions go to the policy desk, ext. 4410.",
        "ADM-5: Translations are for convenience; the English text governs.",
        "ADM-6: Breach of this policy may trigger retraining.",
        "ADM-7: Records requests go through the front desk form F-12.",
        "ADM-8: After-hours issues call the duty line, ext. 4499.",
        "ADM-9: Policy exceptions need two signatures and a case number.",
        "ADM-10: Archived versions are kept for reference only.",
        "ADM-11: Staff must acknowledge updates within five business days.",
        "ADM-12: Fees, if any, are listed in schedule S of the appendix.",
        "ADM-13: Weekend coverage follows the on-call rotation posted monthly.",
        "ADM-14: Complaints follow the three-step grievance process.",
    ]
    n_fill = rng.randint(8, 14) if long else rng.randint(0, 2)
    picked = rng.sample(fillers, n_fill) if n_fill else []
    if long:
        # decisive clauses first, filler after (learnable under the budget)
        body = clauses + picked
    else:
        body = clauses + picked
        rng.shuffle(body)
    state = head + "\n".join(body) + "\nCASE: " + scenario
    return DecisionExample(
        id=f"synthhard-policy-{i:05d}",
        source=SOURCE,
        qtype="choice",
        state=state,
        instructions=instructions,
        answer=gold,
        criteria=options,
        meta={"family": "policy", "rule": rule, "long": long},
    )


# ---------------------------------------------------------------------------
# family 2: probability (exact combinatorics)


_SCORE_LEVELS = ["very unlikely", "unlikely", "even chance", "likely", "very likely"]


def _prob_band(p: Fraction) -> int:
    if p < 0:
        return 0
    band = int(p * 5)
    return min(4, band)


def _probability_item(rng: random.Random, i: int) -> DecisionExample:
    org, dept = rng.choice(_ORGS)
    head = f"{org.upper()} - {dept}\n"
    flavor = rng.random()
    if flavor < 0.6:
        # hypergeometric: lot N with K defectives, draw n, P(all pass)
        total = rng.randint(8, 30)
        bad = rng.randint(1, max(1, total // 4))
        draw = rng.randint(2, min(6, total - bad))
        good = total - bad
        p = Fraction(comb(good, draw), comb(total, draw))
        state = (
            f"{head}Inspection report: lot L-{rng.randint(100, 999)} holds {total} units, "
            f"{bad} failed inspection. Dispatch draws {draw} units for shipping. "
            f"What is the chance all {draw} drawn units pass?"
        )
        detail = f"lot {total}, {bad} bad, draw {draw}"
    else:
        # binomial: per-job fault rate 1/d, n jobs, P(no faults)
        denom = rng.choice([4, 5, 8, 10])
        jobs = rng.randint(2, 8)
        p = Fraction(denom - 1, denom) ** jobs
        state = (
            f"{head}Reliability log: machine faults about 1 in {denom} jobs. "
            f"Next batch runs {jobs} jobs. "
            f"What is the chance the batch runs with zero faults?"
        )
        detail = f"1-in-{denom}, {jobs} jobs"
    if rng.random() < 0.5:
        return DecisionExample(
            id=f"synthhard-prob-{i:05d}",
            source=SOURCE,
            qtype="score",
            state=state,
            instructions="Rate the chance the good outcome happens.",
            answer=_prob_band(p),
            criteria=_SCORE_LEVELS,
            meta={"family": "probability", "kind": "score", "detail": detail, "p": str(p)},
        )
    thresh = rng.choice([25, 50, 75])
    gold = p * 100 >= thresh
    return DecisionExample(
        id=f"synthhard-prob-{i:05d}",
        source=SOURCE,
        qtype="noul",
        state=state + f" Flag if the chance is at least {thresh}%.",
        instructions=f"Is the chance of the good outcome at least {thresh}%?",
        answer=gold,
        criteria={"false": f"chance below {thresh}%", "true": f"chance at least {thresh}%"},
        meta={"family": "probability", "kind": "noul", "detail": detail, "p": str(p)},
    )


# ---------------------------------------------------------------------------
# family 3: temporal (exact date arithmetic)


def _temporal_item(rng: random.Random, i: int) -> DecisionExample:
    org, dept = rng.choice(_ORGS)
    head = f"{org.upper()} - {dept}\n"
    if rng.random() < 0.6:
        # overdue?noul
        due = _date(rng)
        grace = rng.choice([0, 7, 14, 30])
        delta = rng.choice([grace - rng.randint(1, 12), grace + rng.randint(0, 20)])
        today = due + _dt.timedelta(days=delta)
        gold = delta > grace
        state = (
            f"{head}Invoice I-{rng.randint(1000, 9999)} due {due.isoformat()} "
            f"with a {grace}-day grace period. Today is {today.isoformat()}. "
            f"Is the invoice overdue?"
        )
        return DecisionExample(
            id=f"synthhard-temporal-{i:05d}",
            source=SOURCE,
            qtype="noul",
            state=state,
            instructions="Is the invoice overdue as of today?",
            answer=gold,
            criteria={"false": "within grace", "true": "past grace"},
            meta={"family": "temporal", "kind": "overdue"},
        )
    # which deadline applies (choice, 3 options, partitioned by category)
    category = rng.choice(["standard", "priority", "bulk"])
    placed = _date(rng)
    spans = {"standard": 30, "priority": 7, "bulk": 45}
    due = placed + _dt.timedelta(days=spans[category])
    options = {
        "deadline_30d": "standard 30-day deadline",
        "deadline_7d": "priority 7-day deadline",
        "deadline_45d": "bulk 45-day deadline",
    }
    gold = {"standard": "deadline_30d", "priority": "deadline_7d", "bulk": "deadline_45d"}[category]
    state = (
        f"{head}Order O-{rng.randint(1000, 9999)} ({category} handling) placed "
        f"{placed.isoformat()}. Standard spans 30 days, priority 7 days, bulk 45 days. "
        f"Which deadline applies to this order?"
    )
    assert due == placed + _dt.timedelta(days=spans[category])
    return DecisionExample(
        id=f"synthhard-temporal-{i:05d}",
        source=SOURCE,
        qtype="choice",
        state=state,
        instructions="Which deadline applies to this order?",
        answer=gold,
        criteria=options,
        meta={"family": "temporal", "kind": "deadline"},
    )


# ---------------------------------------------------------------------------
# family 4: legal screening (limitation periods, exact date math)


_CLAIMS = {
    "contract": (6, "six-year contract limit"),
    "injury": (3, "three-year injury limit"),
    "property": (4, "four-year property limit"),
}


def _legal_item(rng: random.Random, i: int) -> DecisionExample:
    org, dept = rng.choice(_ORGS)
    head = f"{org.upper()} - {dept}\n"
    claim = rng.choice(list(_CLAIMS))
    limit, desc = _CLAIMS[claim]
    event = _date(rng, 2018, 2023)
    over = rng.choice([limit - rng.randint(1, 2), limit + rng.randint(0, 3)])
    filed = _add_years(event, over)
    tolling = rng.random() < 0.3
    toll_years = 0
    if tolling:
        toll_years = rng.randint(1, 3)
        note = (
            f"The claimant was a minor until {_add_years(event, toll_years).isoformat()}, "
            f"tolling the limit by {toll_years} years. "
        )
    else:
        note = "No tolling facts. "
    options = {
        "within_limitation": "filed in time",
        "time_barred": "filed too late",
        "tolling_applies": "timely only because of tolling",
    }
    if tolling and over > limit and over <= limit + toll_years:
        gold = "tolling_applies"
    elif over > limit:
        gold = "time_barred"
    else:
        gold = "within_limitation"
    state = (
        f"{head}Intake screening: {claim} claim, incident {event.isoformat()}, "
        f"filed {filed.isoformat()}. {desc}. " + note
        + "How should intake classify the filing?"
    )
    return DecisionExample(
        id=f"synthhard-legal-{i:05d}",
        source=SOURCE,
        qtype="choice",
        state=state,
        instructions="How should intake classify the filing?",
        answer=gold,
        criteria=options,
        meta={"family": "legal", "claim": claim},
    )


# ---------------------------------------------------------------------------
# family 5: fact presence (anti-hallucination noul)


def _fact_item(rng: random.Random, i: int) -> DecisionExample:
    org, dept = rng.choice(_ORGS)
    head = f"{org.upper()} - {dept}\n"
    topics = {
        "maintenance window": ["Sunday 02:00-04:00", "Saturday 22:00-23:30", "Wednesday 01:00-02:00"],
        "support line": ["ext. 4410", "ext. 4425", "ext. 4401"],
        "refund cap": ["$500 per order", "$200 per order", "$1000 per order"],
        "retention span": ["3 years", "5 years", "7 years"],
    }
    topic = rng.choice(list(topics))
    values = topics[topic]
    stated = rng.choice(values)
    asked = stated if rng.random() < 0.5 else rng.choice([v for v in values if v != stated])
    gold = asked == stated
    person = _person(rng)
    state = (
        f"{head}Notice {rng.randint(100, 999)}: please be advised the {topic} is "
        f"{stated}. Contact {person} with questions. "
        f"Does the notice state the {topic} is {asked}?"
    )
    return DecisionExample(
        id=f"synthhard-fact-{i:05d}",
        source=SOURCE,
        qtype="noul",
        state=state,
        instructions="Does the notice state exactly that fact?",
        answer=gold,
        criteria={"false": "not stated", "true": "stated as-is"},
        meta={"family": "fact"},
    )


# ---------------------------------------------------------------------------
# entry point


def synthetic_hard_examples(n: int = 10000, seed: int = 7) -> list[DecisionExample]:
    """Full hard-reasoning slice: deterministic in (n, seed), ids unique."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    builders = (
        ["policy"] * 40 + ["probability"] * 25 + ["temporal"] * 20
        + ["legal"] * 10 + ["fact"] * 5
    )
    long_frac = 0.3
    examples: list[DecisionExample] = []
    for i in range(n):
        fam = builders[i % len(builders)]
        if fam == "policy":
            # reshuffle the policy rule order per item for variety (still seeded)
            examples.append(_policy_item(rng, i, long=rng.random() < long_frac))
        elif fam == "probability":
            examples.append(_probability_item(rng, i))
        elif fam == "temporal":
            examples.append(_temporal_item(rng, i))
        elif fam == "legal":
            examples.append(_legal_item(rng, i))
        else:
            examples.append(_fact_item(rng, i))
    ids = [ex.id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate synthetic_hard ids")
    return examples