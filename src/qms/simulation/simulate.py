"""Multi-server discrete-event simulation of a single queue (department/day).

Given the patients who actually show up (each with an arrival/ready time, a
service duration, and a priority), and a number of parallel doctors, the
simulator plays out service under a chosen sequencing policy and returns the
realised waiting times plus utilisation inputs.

Because only the *ordering* differs between policies (arrival and service times
are held fixed from the data), policy comparisons are strictly apples-to-apples.
"""
from __future__ import annotations

from typing import List, Dict
import numpy as np

from ..optimization.scheduler import Policy


def simulate(patients: List[Dict], n_servers: int, policy: Policy,
             emergency_level: int = 1, weights: Dict | None = None) -> Dict:
    """Simulate service of ``patients`` across ``n_servers`` doctors.

    Each patient dict needs: id, ready (minutes), service (minutes), priority.
    """
    n_servers = max(int(n_servers), 1)
    weights = weights or {"waiting": 0.4, "utilization": 0.2,
                          "fairness": 0.2, "emergency": 0.2}
    if not patients:
        return {"waits": [], "priorities": [], "busy_time": 0.0,
                "n_servers": n_servers, "makespan": 0.0, "start": 0.0, "order": []}

    remaining = [dict(p) for p in patients]
    start = min(p["ready"] for p in remaining)
    free = [start] * n_servers
    avg_service = float(np.mean([p["service"] for p in remaining]))
    ctx = {"emergency_level": emergency_level, "weights": weights,
           "avg_service": avg_service}

    waits, priorities, order = [], [], []
    while remaining:
        si = int(np.argmin(free))
        f = free[si]
        ready_cands = [p for p in remaining if p["ready"] <= f]
        if not ready_cands:                       # doctor idle: jump to next arrival
            f = min(p["ready"] for p in remaining)
            free[si] = f
            ready_cands = [p for p in remaining if p["ready"] <= f]

        idx = policy.choose(ready_cands, f, ctx)
        chosen = ready_cands[idx]

        wait = max(f - chosen["ready"], 0.0)
        free[si] = f + chosen["service"]

        waits.append(wait)
        priorities.append(chosen["priority"])
        order.append(chosen["id"])
        remaining.remove(chosen)

    makespan = max(free)
    busy_time = float(sum(p["service"] for p in patients))
    return {
        "waits": waits,
        "priorities": priorities,
        "busy_time": busy_time,
        "n_servers": n_servers,
        "makespan": makespan,
        "start": start,
        "order": order,
    }


def minutes_of_day(ts) -> float:
    """Convert a pandas Timestamp to minutes since midnight (for ready times)."""
    if ts is None or (hasattr(ts, "__len__") and len(ts) == 0):
        return 0.0
    return float(ts.hour * 60 + ts.minute + ts.second / 60.0)
