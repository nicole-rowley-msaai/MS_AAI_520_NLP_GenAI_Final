"""LangGraph wiring: plan -> route -> specialists -> synthesize ->
evaluate <-> optimize -> reflect -> remember.

run_agent(ticker) returns (report_markdown, trace). Every node logs to
the RunTrace so the notebook can print each step.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents import evaluator as ev
from src.agents.evaluation_lead import QualityPolicy
from src.agents.evaluation_pipeline import (
    NumericPolicy,
    ReviewedEvaluationSession,
    numeric_feedback,
)
from src.agents.numeric_validation import NumericAudit, audit_numeric_claims
from src.agents.planner import make_plan
from src.agents.router import route_step
from src.agents.schemas import (
    Draft,
    Evaluation,
    Finding,
    Reflection,
    ResearchPlan,
    RouteDecision,
)
from src.agents.specialists import run_specialist
from src.agents.synthesizer import render, synthesize
from src.memory import store
from src.tools import safe_call
from src.tools.market import get_key_stats
from src.trace import RunTrace


class AgentState(TypedDict, total=False):
    ticker: str
    sector: str | None
    use_memory: bool
    trace: RunTrace
    notes: list[dict]
    plan: ResearchPlan
    routes: list[RouteDecision]
    findings: list[Finding]
    draft: Draft
    evaluation: Evaluation
    history: list[dict]  # one entry per evaluator iteration
    iteration: int
    reflection: Reflection
    new_notes: list[dict]
    report: str
    reviewed_evaluation: bool
    review_session: ReviewedEvaluationSession
    numeric_audit: NumericAudit
    accepted: bool
    stop_reason: str
    selected_iteration: int


# Nodes ----------------------------------------------------------------------
def node_plan(state: AgentState) -> AgentState:
    t = state["trace"]
    stats = safe_call("get_key_stats", get_key_stats, state["ticker"])
    sector = stats.data.get("sector") if stats.ok else None
    notes = (
        store.relevant_notes(state["ticker"], sector, store.load_notes())
        if state.get("use_memory", True)
        else []
    )
    t.log(
        "memory", "read", {"use_memory": state.get("use_memory", True)}, notes
    )
    plan = t.timed(
        "planner", "plan", make_plan, state["ticker"], sector, notes
    )
    return {"sector": sector, "notes": notes, "plan": plan}


def node_route(state: AgentState) -> AgentState:
    t = state["trace"]
    routes = [
        t.timed("router", "decision", route_step, step)
        for step in state["plan"].steps
    ]
    return {"routes": routes}


def node_specialists(state: AgentState) -> AgentState:
    """Run one specialist per route, in parallel.

    The specialists are independent (each has its own steps and tools),
    and together they are more than half of a run's wall time, so they
    run in threads. The trace is append-only, so concurrent logging is
    safe; each tool call records its route so the log can be regrouped.
    """
    t = state["trace"]
    by_route: dict[str, list] = {}
    for step, decision in zip(state["plan"].steps, state["routes"]):
        by_route.setdefault(decision.route, []).append(step)

    def run(item):
        route, steps = item
        return t.timed(
            "specialist",
            route,
            run_specialist,
            route,
            state["ticker"],
            steps,
            t,
        )

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(by_route)) as pool:
        findings = list(pool.map(run, by_route.items()))
    t.log(
        "specialists",
        "wall",
        {"routes": list(by_route)},
        {"parallel": len(by_route)},
        time.perf_counter() - start,
    )
    return {"findings": findings}


def node_synthesize(state: AgentState) -> AgentState:
    t = state["trace"]
    draft = t.timed(
        "synthesizer", "draft", synthesize, state["ticker"], state["findings"]
    )
    updates: AgentState = {"draft": draft, "iteration": 0, "history": []}
    if state.get("reviewed_evaluation", False):
        # Use nonblocking rules for claims the narrow parser cannot recognize.
        # Explicit contradictions still prevent acceptance.
        updates["review_session"] = ReviewedEvaluationSession(
            quality_policy=QualityPolicy(),
            numeric_policy=NumericPolicy(
                require_recognized_claims=False,
                reject_unverified=False,
            ),
        )
    return updates


def node_evaluate(state: AgentState) -> AgentState:
    t = state["trace"]
    evaluation = t.timed(
        "evaluator", "scores", ev.evaluate, state["draft"], state["findings"]
    )
    row = {
        "iteration": state["iteration"],
        "average": evaluation.average,
        "scores": {c: getattr(evaluation, c) for c in ev.CRITERIA},
        "critiques": evaluation.critiques,
    }
    updates: AgentState = {"evaluation": evaluation}
    if state.get("reviewed_evaluation", False):
        audit = audit_numeric_claims(state["draft"], state["findings"])
        candidate = state["review_session"].record(
            state["draft"], evaluation, audit
        )
        row.update(
            {
                "accepted": candidate.passed,
                "failures": list(candidate.failures),
                "numeric_checked": len(audit.checks),
                "numeric_contradicted": len(audit.contradictions),
                "numeric_unverified": len(audit.unverified),
            }
        )
        updates["numeric_audit"] = audit
        updates["review_session"] = state["review_session"]
        t.log(
            "evaluator", "numeric_audit",
            {"iteration": state["iteration"]}, audit.rows(),
        )
    updates["history"] = state["history"] + [row]
    return updates


def should_refine(state: AgentState) -> str:
    # Every draft gets at least MIN_ITERATIONS rounds of critique and
    # revision; after that the loop stops as soon as the score passes.
    if state.get("reviewed_evaluation", False):
        session = state["review_session"]
        refine = session.should_refine()
        any_passed = any(candidate.passed for candidate in session.candidates)
        outcome = "refine" if refine else "pass" if any_passed else "exhausted"
        state["trace"].log(
            "evaluator", "gate",
            {
                "average": state["evaluation"].average,
                "iteration": state["iteration"],
                "failures": list(session.candidates[-1].failures),
            },
            outcome,
        )
        return "refine" if refine else "finish"

    done_minimum = state["iteration"] >= ev.MIN_ITERATIONS
    passed = done_minimum and (
        state["evaluation"].average >= ev.PASS_THRESHOLD
    )
    exhausted = state["iteration"] + 1 >= ev.MAX_ITERATIONS
    state["trace"].log(
        "evaluator",
        "gate",
        {
            "average": state["evaluation"].average,
            "iteration": state["iteration"],
        },
        "pass" if passed else "exhausted" if exhausted else "refine",
    )
    return "finish" if passed or exhausted else "refine"


def node_optimize(state: AgentState) -> AgentState:
    t = state["trace"]
    feedback = state["evaluation"]
    if state.get("reviewed_evaluation", False):
        feedback = feedback.model_copy(deep=True)
        feedback.critiques.extend(
            numeric_feedback(
                state["numeric_audit"],
                state["review_session"].numeric_policy,
            )
        )
    draft = t.timed(
        "optimizer",
        "revision",
        ev.optimize,
        state["draft"],
        feedback,
        state["findings"],
    )
    if draft.ticker != state["draft"].ticker:
        raise ValueError("optimizer changed the stock ticker")
    return {"draft": draft, "iteration": state["iteration"] + 1}


def node_reflect(state: AgentState) -> AgentState:
    t = state["trace"]
    updates: AgentState = {}
    selected_draft = state["draft"]
    if state.get("reviewed_evaluation", False):
        outcome = state["review_session"].finish()
        selected_draft = outcome.draft
        updates.update(
            {
                "draft": selected_draft,
                "evaluation": outcome.evaluation,
                "numeric_audit": outcome.audit,
                "accepted": outcome.accepted,
                "stop_reason": outcome.stop_reason,
                "selected_iteration": outcome.selected_iteration,
                "history": [
                    {
                        **row,
                        "selected": (
                            row["iteration"] == outcome.selected_iteration
                        ),
                    }
                    for row in state["history"]
                ],
            }
        )
        t.log(
            "evaluator", "selection",
            {"evaluated": len(outcome.history)},
            {
                "selected_iteration": outcome.selected_iteration,
                "accepted": outcome.accepted,
                "stop_reason": outcome.stop_reason,
                "average": outcome.evaluation.average,
                "numeric_checked": len(outcome.audit.checks),
                "numeric_unverified": len(outcome.audit.unverified),
                "numeric_contradicted": len(outcome.audit.contradictions),
            },
        )
    tool_log = [e.output for e in t.select("tool", "call")]
    reflection = t.timed(
        "reflection",
        "self_review",
        ev.reflect,
        selected_draft,
        state["findings"],
        tool_log,
        stored_lessons=[n["lesson"] for n in state.get("notes", [])],
    )
    new_notes = []
    if state.get("use_memory", True):
        new_notes = store.add_lessons(
            state["ticker"], state.get("sector"), reflection.lessons
        )
    t.log("memory", "write", reflection.lessons, new_notes)
    report = render(selected_draft)
    if state.get("reviewed_evaluation", False):
        if not updates["accepted"]:
            report = (
                "> **Evaluation not accepted:** The available drafts did not "
                "meet all quality and numerical checks. Manual review is "
                "required.\n\n" + report
            )
        elif not updates["numeric_audit"].checks:
            report = (
                "> **Numeric coverage warning:** No financial claims matched "
                "the limited numeric parser. This is not evidence of numeric "
                "accuracy.\n\n" + report
            )
    updates.update(
        {
            "reflection": reflection,
            "new_notes": new_notes,
            "report": report,
        }
    )
    return updates


# Graph ----------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("route", node_route)
    g.add_node("specialists", node_specialists)
    g.add_node("synthesize", node_synthesize)
    g.add_node("evaluate", node_evaluate)
    g.add_node("optimize", node_optimize)
    g.add_node("reflect", node_reflect)

    g.set_entry_point("plan")
    g.add_edge("plan", "route")
    g.add_edge("route", "specialists")
    g.add_edge("specialists", "synthesize")
    g.add_edge("synthesize", "evaluate")
    g.add_conditional_edges(
        "evaluate", should_refine, {"refine": "optimize", "finish": "reflect"}
    )
    g.add_edge("optimize", "evaluate")
    g.add_edge("reflect", END)
    return g.compile()


GRAPH = build_graph()


def run_agent(
    ticker: str,
    use_memory: bool = True,
    save: bool = True,
    reviewed_evaluation: bool = False,
) -> tuple[str, RunTrace, AgentState]:
    """Research one ticker; opt into the enhanced evaluator when requested.

    Keep the original behavior by default for reproducible comparisons.
    """
    trace = RunTrace(ticker=ticker.upper())
    final = GRAPH.invoke(
        {
            "ticker": ticker.upper(),
            "use_memory": use_memory,
            "trace": trace,
            "reviewed_evaluation": reviewed_evaluation,
        }
    )
    if save:
        trace.save()
    return final["report"], trace, final
