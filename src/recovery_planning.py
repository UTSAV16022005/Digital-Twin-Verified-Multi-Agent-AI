"""Constrained recovery-planning agent for Digital Twin outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from src.digital_twin import DigitalTwin, TransitionOutcome, TwinState


@dataclass(frozen=True)
class PlanningWeights:
    """Weights for the transparent candidate-ranking objective."""

    failure_risk: float = 0.55
    model_uncertainty: float = 0.15
    mission_cost: float = 0.20
    projected_drift: float = 0.10


@dataclass
class RecoveryPlan:
    """Auditable output of the Recovery Planning Agent."""

    selected_action: str | None
    decision_status: str
    rationale: str
    ranked_candidates: list[dict]
    rejected_candidates: list[dict]

    def as_dict(self) -> dict:
        return {
            "selected_action": self.selected_action,
            "decision_status": self.decision_status,
            "rationale": self.rationale,
            "ranked_candidates": self.ranked_candidates,
            "rejected_candidates": self.rejected_candidates,
        }


class RecoveryPlanningAgent:
    """Rank Digital Twin outcomes after hard safety filtering.

    The planner does not invent physical actions and does not override safety
    constraints. It chooses only from the action space registered by the Twin.
    """

    def __init__(
        self,
        weights: PlanningWeights | None = None,
        risk_budget: float = 0.25,
    ) -> None:
        self.weights = weights or PlanningWeights()
        self.risk_budget = risk_budget

    def score(self, outcome: TransitionOutcome) -> float:
        """Calculate the lower-is-better planning objective."""

        weights = self.weights
        return float(
            weights.failure_risk * outcome.predicted_failure_risk
            + weights.model_uncertainty * outcome.model_uncertainty
            + weights.mission_cost * outcome.mission_cost
            + weights.projected_drift * outcome.projected_drift
        )

    def plan(
        self,
        twin: DigitalTwin,
        state: TwinState,
        horizon: int = 5,
        actions: list[str] | None = None,
    ) -> RecoveryPlan:
        """Simulate, filter, score, and rank candidate recovery actions."""

        outcomes = twin.what_if(state, actions=actions, horizon=horizon)
        ranked: list[dict] = []
        rejected: list[dict] = []
        for outcome in outcomes:
            record = {
                "action": outcome.action,
                "objective_score": self.score(outcome),
                "predicted_failure_risk": outcome.predicted_failure_risk,
                "model_uncertainty": outcome.model_uncertainty,
                "projected_drift": outcome.projected_drift,
                "mission_cost": outcome.mission_cost,
                "predicted_rul": outcome.predicted_state.predicted_rul,
                "safe_under_constraints": outcome.safe_under_constraints,
                "violations": outcome.violations,
            }
            if not outcome.safe_under_constraints:
                rejected.append(record)
            elif outcome.predicted_failure_risk > self.risk_budget:
                record["violations"] = [
                    f"predicted_failure_risk_above_budget_{self.risk_budget:.3f}"
                ]
                rejected.append(record)
            else:
                ranked.append(record)

        ranked.sort(key=lambda record: record["objective_score"])
        if ranked:
            selected = ranked[0]
            action = selected["action"]
            rationale = (
                f"Selected {action} because it passed the hard constraints and had the "
                f"lowest weighted objective ({selected['objective_score']:.4f}) within "
                f"the risk budget {self.risk_budget:.3f}."
            )
            status = "recommendation"
        else:
            action = None
            rationale = (
                "No candidate action passed both the Digital Twin constraints and the "
                f"failure-risk budget {self.risk_budget:.3f}. Escalate to the safety layer."
            )
            status = "no_safe_action"

        return RecoveryPlan(
            selected_action=action,
            decision_status=status,
            rationale=rationale,
            ranked_candidates=ranked,
            rejected_candidates=rejected,
        )

