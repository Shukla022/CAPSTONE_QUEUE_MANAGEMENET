"""Scheduling policies.

Every policy exposes ``choose(candidates, now, ctx) -> index`` that selects the
next patient to serve from the list of currently-waiting (already-arrived)
candidates. This is the sequencing decision the simulator asks for whenever a
doctor becomes free.

Policies
--------
* FCFSPolicy          -- first-come-first-served (the classic baseline)
* StaticPriorityPolicy-- emergencies first, then FCFS (Abusair et al. style)
* MultiObjectivePolicy-- our weighted scheduler balancing waiting, fairness,
                         utilisation (shortest-processing-time) and emergency.

A ``candidate`` is a dict: {id, ready, service, priority}.
``ctx`` carries {emergency_level, weights, avg_service}.
"""
from __future__ import annotations

from typing import List, Dict


class Policy:
    name = "policy"

    def choose(self, candidates: List[Dict], now: float, ctx: Dict) -> int:
        raise NotImplementedError


class FCFSPolicy(Policy):
    name = "FCFS"

    def choose(self, candidates, now, ctx) -> int:
        # earliest arrival, tie-break by id for determinism
        return min(range(len(candidates)),
                   key=lambda i: (candidates[i]["ready"], candidates[i]["id"]))


class StaticPriorityPolicy(Policy):
    name = "Static-Priority"

    def choose(self, candidates, now, ctx) -> int:
        lvl = ctx["emergency_level"]
        # emergencies (priority >= lvl) first, then earliest arrival
        return min(
            range(len(candidates)),
            key=lambda i: (
                0 if candidates[i]["priority"] >= lvl else 1,
                candidates[i]["ready"],
                candidates[i]["id"],
            ),
        )


class MultiObjectivePolicy(Policy):
    name = "Multi-Objective"
    EMERGENCY_SCALE = 100.0  # makes the weighted emergency term dominate when present

    def choose(self, candidates, now, ctx) -> int:
        w = ctx["weights"]
        lvl = ctx["emergency_level"]
        avg_s = max(ctx.get("avg_service", 1.0), 1e-6)
        w_waitfair = w["waiting"] + w["fairness"]
        w_util = w["utilization"]
        w_emer = w["emergency"]

        best_i, best_score = 0, -1e18
        for i, c in enumerate(candidates):
            wait_so_far = max(now - c["ready"], 0.0)
            emergency_flag = 1.0 if c["priority"] >= lvl else 0.0
            # older waiters -> higher score (waiting + fairness)
            wait_term = wait_so_far / (avg_s + 1.0)
            # shorter jobs -> higher score (throughput / utilisation, SPT rule)
            short_term = avg_s / (c["service"] + 1.0)
            score = (w_waitfair * wait_term
                     + w_util * short_term
                     + w_emer * self.EMERGENCY_SCALE * emergency_flag)
            if score > best_score:
                best_score, best_i = score, i
        return best_i


POLICIES = {
    "FCFS": FCFSPolicy,
    "Static-Priority": StaticPriorityPolicy,
    "Multi-Objective": MultiObjectivePolicy,
}


def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Choose from {list(POLICIES)}.")
    return POLICIES[name]()
