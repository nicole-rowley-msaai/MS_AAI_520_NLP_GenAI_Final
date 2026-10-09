"""LangGraph wiring: plan -> route -> specialists -> synthesize ->
evaluate <-> optimize -> reflect -> remember.

run_agent(ticker) returns (report_markdown, trace). Every node logs to
the RunTrace so the notebook can print each step.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents import evaluator as ev
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
    t = state["trace"]
    by_route: dict[str, list] = {}
    for step, decision in zip(state["plan"].steps, state["routes"]):
        by_route.setdefault(decision.route, []).append(step)
    findings = []
    for route, steps in by_route.items():
        finding = t.timed(
            "specialist",
            route,
            run_specialist,
            route,
            state["ticker"],
            steps,
            t,
        )
        findings.append(finding)
    return {"findings": findings}


def node_synthesize(state: AgentState) -> AgentState:
    t = state["trace"]
    draft = t.timed(
        "synthesizer", "draft", synthesize, state["ticker"], state["findings"]
    )
    return {"draft": draft, "iteration": 0, "history": []}


def node_evaluate(state: AgentState) -> AgentState:
    t = state["trace"]
    evaluation = t.timed(
        "evaluator", "scores", ev.evaluate, state["draft"], state["findings"]
    )
    history = state["history"] + [
        {
            "iteration": state["iteration"],
            "average": evaluation.average,
            "scores": {c: getattr(evaluation, c) for c in ev.CRITERIA},
            "critiques": evaluation.critiques,
        }
    ]
    return {"evaluation": evaluation, "history": history}


def should_refine(state: AgentState) -> str:
    passed = state["evaluation"].average >= ev.PASS_THRESHOLD
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
    draft = t.timed(
        "optimizer",
        "revision",
        ev.optimize,
        state["draft"],
        state["evaluation"],
        state["findings"],
    )
    return {"draft": draft, "iteration": state["iteration"] + 1}


def node_reflect(state: AgentState) -> AgentState:
    t = state["trace"]
    tool_log = [e.output for e in t.select("tool", "call")]
    reflection = t.timed(
        "reflection",
        "self_review",
        ev.reflect,
        state["draft"],
        state["findings"],
        tool_log,
    )
    new_notes = []
    if state.get("use_memory", True):
        new_notes = store.add_lessons(
            state["ticker"], state.get("sector"), reflection.lessons
        )
    t.log("memory", "write", reflection.lessons, new_notes)
    return {
        "reflection": reflection,
        "new_notes": new_notes,
        "report": render(state["draft"]),
    }


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
    ticker: str, use_memory: bool = True, save: bool = True
) -> tuple[str, RunTrace, AgentState]:
    """Research one ticker. Returns (report markdown, trace, final state)."""
    trace = RunTrace(ticker=ticker.upper())
    final = GRAPH.invoke(
        {
            "ticker": ticker.upper(),
            "use_memory": use_memory,
            "trace": trace,
        }
    )
    if save:
        trace.save()
    return final["report"], trace, final
