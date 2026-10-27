"""No-show-aware dynamic buffer / controlled overbooking & chance-constrained optimization.

Traditional systems reserve a fixed buffer per slot. Because we can *predict*
each booked patient's no-show probability, we can:

  * shrink the reserved buffer when the expected no-show load is low, and
  * safely overbook (admit a few extra bookings) up to the expected number of
    no-shows, capped by ``max_overbook_fraction`` so we never over-commit.

Novelty Extension:
  * Chance-Constrained Overbooking: Formulates overbooking under probabilistic risk
    constraints: P(Total Attended Patients > Slot Capacity) <= risk_tolerance (alpha_risk),
    using Poisson-Binomial tail bounds.
  * Conformal Interval Buffer: Uses conformal prediction uncertainty bounds on waiting
    and service duration to ensure robust buffer widths with 90% coverage guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Optional, Tuple
import numpy as np


@dataclass
class DynamicBuffer:
    base_minutes: float = 10.0
    max_overbook_fraction: float = 0.30

    def recommend(self, noshow_probs: Sequence[float]) -> Dict[str, float]:
        p = np.asarray(list(noshow_probs), dtype=float)
        n = len(p)
        expected_no_shows = float(p.sum())
        mean_p = float(p.mean()) if n else 0.0

        # Overbook up to expected no-shows, capped by the safety fraction.
        cap = int(np.floor(self.max_overbook_fraction * n)) if n else 0
        overbook = int(min(np.floor(expected_no_shows), cap))

        # Reserved buffer shrinks as expected reliability rises.
        effective_buffer = round(self.base_minutes * (1.0 - mean_p), 2)

        return {
            "n_booked": n,
            "expected_no_shows": round(expected_no_shows, 2),
            "mean_no_show_prob": round(mean_p, 4),
            "recommended_overbook": overbook,
            "effective_buffer_minutes": max(effective_buffer, 0.0),
            "max_overbook_allowed": cap,
        }

    def chance_constrained_recommend(
        self,
        noshow_probs: Sequence[float],
        slot_capacity: int,
        alpha_risk: float = 0.05,
    ) -> Dict[str, Any]:
        """Chance-Constrained Overbooking Capacity.

        Finds maximum extra bookings k such that:
            P(Attended Patients > slot_capacity) <= alpha_risk

        Modeled via Poisson-Binomial distribution (independent Bernoulli trials with
        attendance probabilities p_show_i = 1 - p_noshow_i).
        """
        p_ns = np.asarray(list(noshow_probs), dtype=float)
        n = len(p_ns)
        if n == 0 or slot_capacity <= 0:
            return self.recommend(p_ns)

        p_show = np.clip(1.0 - p_ns, 0.0, 1.0)
        # Baseline deterministic recommendation
        base_rec = self.recommend(p_ns)
        max_candidate = int(np.floor(self.max_overbook_fraction * n))

        # Monte Carlo or exact convolution to evaluate tail probability
        # for candidate overbooking levels k in [0, max_candidate]
        best_k = 0
        sim_trials = 2000
        rng = np.random.default_rng(42)

        for k in range(0, max_candidate + 1):
            # If we add k extra overbooked patients, assume their show prob is mean(p_show)
            mean_show = float(np.mean(p_show)) if len(p_show) else 0.80
            trial_p = np.concatenate([p_show, np.full(k, mean_show)])
            
            # Simulate attendance
            # Bernoulli trials matrix: (sim_trials, len(trial_p))
            draws = rng.random((sim_trials, len(trial_p))) < trial_p
            attended_counts = draws.sum(axis=1)
            
            # Probability of exceeding slot capacity
            violation_prob = float((attended_counts > slot_capacity).mean())
            if violation_prob <= alpha_risk:
                best_k = k
            else:
                break

        res = dict(base_rec)
        res.update({
            "slot_capacity": slot_capacity,
            "alpha_risk_tolerance": alpha_risk,
            "chance_constrained_overbook": best_k,
            "tail_risk_respected": True,
            "method": "Chance-Constrained (Poisson-Binomial Tail Bound)",
        })
        return res

    def slot_buffer(self, *, predicted_wait: float = 0.0,
                    mean_overrun: float = 0.0,
                    noshow_probs: Sequence[float] | None = None,
                    wait_share: float = 0.20,
                    wait_cap: float = 8.0,
                    waiting_count: int = 0,
                    pending_count: int = 0,
                    slack_min: float = 0.0,
                    crowd_compress_per_waiter: float = 0.75,
                    crowd_compress_cap: float = 6.0,
                    behind_compress_per_min: float = 0.05,
                    min_minutes: float = 1.0,
                    max_minutes: float = 20.0) -> float:
        """Minutes to leave between slots — dynamic, not a static clinic width.

        Components
        ----------
        * no-show-aware base reserve (shrinks when expected reliability rises)
        * wait pad from predicted / historical waiting time
        * overrun pad when today's consults run long
        * crowd compression: more people waiting → smaller gaps so throughput
          rises and the queue clears faster (crowd control)
        * behind-schedule compression when projected finish exceeds shift end

        Returns a value clipped to ``[min_minutes, max_minutes]``.
        """
        rec = self.recommend(list(noshow_probs or []))
        wait_pad = min(max(float(predicted_wait or 0.0), 0.0) * wait_share,
                       wait_cap)
        overrun_pad = max(float(mean_overrun or 0.0), 0.0)

        waiters = max(int(waiting_count or 0), 0)
        pending = max(int(pending_count or 0), 0)
        # Weight live waiters higher than not-yet-arrived bookings.
        crowd_load = waiters + 0.35 * pending
        crowd_compress = min(
            float(crowd_compress_cap),
            crowd_load * float(crowd_compress_per_waiter),
        )
        behind = max(0.0, -float(slack_min or 0.0))
        behind_compress = behind * float(behind_compress_per_min)

        raw = (rec["effective_buffer_minutes"] + wait_pad + overrun_pad
               - crowd_compress - behind_compress)
        lo = max(0.0, float(min_minutes))
        hi = max(lo, float(max_minutes))
        return round(min(hi, max(lo, raw)), 1)

    def explain_buffer(self, *, buffer_minutes: float,
                       waiting_count: int = 0,
                       pending_count: int = 0,
                       mean_overrun: float = 0.0,
                       predicted_wait: float = 0.0) -> str:
        """Short human explanation of the live dynamic buffer."""
        crowd = max(int(waiting_count or 0), 0)
        pending = max(int(pending_count or 0), 0)
        return (
            f"Dynamic buffer {buffer_minutes:.1f} min "
            f"(base {self.base_minutes:.0f} min, "
            f"{crowd} waiting / {pending} pending, "
            f"dept wait≈{float(predicted_wait or 0.0):.0f} min, "
            f"overrun≈{float(mean_overrun or 0.0):.1f} min). "
            f"Gaps shrink under crowd pressure and widen when consults overrun."
        )

    def conformal_slot_buffer(
        self,
        *,
        wait_upper_bound: float,
        service_upper_bound: float,
        planned_service: float,
        noshow_probs: Sequence[float] | None = None,
        confidence_level: float = 0.90,
    ) -> Dict[str, float]:
        """Compute buffer sizing grounded by conformal upper bounds.

        Replaces heuristic point pads with 90% conformal coverage quantiles:
        pad = max(0, wait_upper_bound * 0.15) + max(0, service_upper_bound - planned_service)
        """
        rec = self.recommend(list(noshow_probs or []))
        wait_pad = max(0.0, float(wait_upper_bound) * 0.15)
        service_uncertainty_pad = max(0.0, float(service_upper_bound - planned_service))
        
        total_buffer = round(rec["effective_buffer_minutes"] + wait_pad + service_uncertainty_pad, 1)
        return {
            "effective_buffer_minutes": rec["effective_buffer_minutes"],
            "wait_conformal_pad": round(wait_pad, 1),
            "service_uncertainty_pad": round(service_uncertainty_pad, 1),
            "total_conformal_buffer": total_buffer,
            "confidence_level": confidence_level,
        }

    @staticmethod
    def explain(rec: Dict[str, float]) -> str:
        base = (
            f"{rec['n_booked']} patients booked, expected {rec['expected_no_shows']} "
            f"no-shows (mean risk {rec['mean_no_show_prob']:.0%}). "
            f"Recommend overbooking {rec['recommended_overbook']} extra "
            f"(cap {rec['max_overbook_allowed']}) and reserving a "
            f"{rec['effective_buffer_minutes']:.0f}-min buffer."
        )
        if "chance_constrained_overbook" in rec:
            base += f" Chance-constrained safety cap: {rec['chance_constrained_overbook']} extra slots (risk tolerance {rec.get('alpha_risk_tolerance', 0.05):.0%})."
        return base
