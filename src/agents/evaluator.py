"""Evaluator-optimizer loop and the post-run reflection.

evaluate()  scores a draft 1-5 on five criteria and lists critiques.
optimize()  rewrites the draft using those critiques.
reflect()   critiques the final report and the process, and writes lessons.

The evaluator uses a different model family from the writer so the draft
is not graded by the model that wrote it.
"""

from src.agents.schemas import CRITERIA, Draft, Evaluation, Finding, Reflection
from src.agents.synthesizer import findings_text, render
from src.config import MODELS
from src.llm import structured

PASS_THRESHOLD = 4.0
MAX_ITERATIONS = 3

EVAL_SYSTEM = (
    "You grade equity research briefs. Score 1-5 on each criterion:\n"
    "- completeness: covers fundamentals, news, market, macro, and risks\n"
    "- evidence: claims are tied to the findings provided\n"
    "- accuracy: every figure matches the findings exactly\n"
    "- balance: bull and bear cases are both substantive\n"
    "- clarity: a portfolio manager can act on it in two minutes\n"
    "List specific, fixable critiques. Be strict about figures that do not "
    "appear in the findings."
)

OPTIMIZE_SYSTEM = (
    "Revise the research brief to address every critique. Keep what was "
    "correct. Do not add figures that are not in the findings. Return the "
    "complete revised brief."
)

REFLECT_SYSTEM = (
    "You are the agent reviewing its own work after the run. First judge "
    "the output: name the weakest section and any claims not backed by the "
    "findings. Then judge the process: what data was missing, which tools "
    "failed, what to change next time. Finally write 3-5 short lessons that "
    "would help a future run on this company or its sector, phrased as "
    "reusable rules (e.g. 'For banks, pull net interest margin')."
)


def evaluate(draft: Draft, findings: list[Finding],
             model: str = MODELS["evaluator"]) -> Evaluation:
    user = f"FINDINGS\n{findings_text(findings)}\n\nBRIEF\n{render(draft)}"
    return structured(model, EVAL_SYSTEM, user, Evaluation)


def optimize(draft: Draft, evaluation: Evaluation, findings: list[Finding],
             model: str = MODELS["writer"]) -> Draft:
    user = (
        f"FINDINGS\n{findings_text(findings)}\n\n"
        f"CURRENT BRIEF\n{render(draft)}\n\n"
        f"SCORES: { {c: getattr(evaluation, c) for c in CRITERIA} }\n"
        f"CRITIQUES:\n" + "\n".join(f"- {c}" for c in evaluation.critiques)
    )
    revised = structured(model, OPTIMIZE_SYSTEM, user, Draft)
    revised.ticker = draft.ticker
    return revised


def reflect(draft: Draft, findings: list[Finding], tool_log: list[dict],
            model: str = MODELS["writer"]) -> Reflection:
    failed = [t for t in tool_log if not t.get("ok")]
    user = (
        f"FINAL BRIEF\n{render(draft)}\n\n"
        f"FINDINGS\n{findings_text(findings)}\n\n"
        f"TOOL CALLS: {len(tool_log)} total, {len(failed)} failed\n"
        f"FAILED: {failed}"
    )
    return structured(model, REFLECT_SYSTEM, user, Reflection)
