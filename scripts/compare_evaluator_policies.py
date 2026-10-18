"""Paired *policy decision* replay of the original and enhanced EquityAgent gates.

Uses existing LLM scores rather than grading different drafts with random LLM calls.
The reviewed run created revisions with enhanced feedback; this analysis does not
claim that the original optimizer would have produced identical subsequent drafts.

Offline:
    python scripts/compare_evaluator_policies.py

Capture another paired input set from one live enhanced run (incurs API costs):
    python scripts/compare_evaluator_policies.py --live NVDA
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data' / 'evaluation' / 'nvda_oct18_logged.json'
DEFAULT_OUTPUT = ROOT / 'data' / 'evaluation' / 'nvda_oct18_gate_comparison.json'

# Values confirmed in the project's src/agents/evaluator.py and
# src/agents/evaluation_lead.py as of 2026-10-18.
ORIGINAL_PASS_AVERAGE = 4.0
ORIGINAL_MIN_REVISIONS = 1
ORIGINAL_MAX_EVALUATIONS = 3
ENHANCED_MIN_AVERAGE = 4.0
ENHANCED_MIN_ACCURACY = 4
ENHANCED_MIN_EVIDENCE = 4
ENHANCED_MIN_REVISIONS = 1
ENHANCED_MAX_EVALUATIONS = 3


def _check_record(row: dict[str, Any], number: int) -> None:
    if row.get('iteration') != number:
        raise ValueError('Records must begin at iteration 0 and be consecutive')
    scores = row.get('scores', {})
    for name in ('completeness', 'evidence', 'accuracy', 'balance', 'clarity'):
        if not isinstance(scores.get(name), int) or not 1 <= scores[name] <= 5:
            raise ValueError(f'Invalid or missing score: {name}')
    computed = sum(scores.values()) / 5
    if abs(computed - float(row['average'])) > 0.01:
        raise ValueError('Reported average disagrees with rubric scores')
    if not isinstance(row.get('numeric_contradicted'), int) or row['numeric_contradicted'] < 0:
        raise ValueError('numeric_contradicted must be a nonnegative integer')
    if not isinstance(row.get('numeric_checked'), int) or row['numeric_checked'] < 0:
        raise ValueError('numeric_checked must be a nonnegative integer')
    if row['numeric_contradicted'] > row['numeric_checked']:
        raise ValueError('Contradictions cannot exceed checked claims')


def compare_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay policy decisions on exactly the same recorded scorer outputs.

    Does not re-run either LLM or the optimizer. Only the acceptance logic is
    compared. Numeric results are the narrow enhanced parser's outputs.
    """
    if not history:
        raise ValueError('At least one recorded evaluation is required')
    if len(history) > max(ORIGINAL_MAX_EVALUATIONS, ENHANCED_MAX_EVALUATIONS):
        raise ValueError('History exceeds evaluation budget')
    for n, row in enumerate(history):
        _check_record(row, n)

    original_stopped = False
    reviewed_stopped = False
    enhanced_accepted = []
    result_rows = []
    original_stop = None
    reviewed_stop = None
    for n, row in enumerate(history):
        scores = row['scores']
        original_eligible = row['average'] >= ORIGINAL_PASS_AVERAGE
        failures = []
        if row['average'] < ENHANCED_MIN_AVERAGE:
            failures.append('average_below_threshold')
        if scores['accuracy'] < ENHANCED_MIN_ACCURACY:
            failures.append('accuracy_below_threshold')
        if scores['evidence'] < ENHANCED_MIN_EVIDENCE:
            failures.append('evidence_below_threshold')
        if row['numeric_contradicted']:
            failures.append('numeric_contradictions')
        # The live policy uses require_recognized_claims=False,
        # reject_unverified=False. Neither missing checks nor unverified
        # recognized claims blocks acceptance.
        reviewed_eligible = not failures
        if reviewed_eligible:
            enhanced_accepted.append(n)

        original_action = 'after_stop'
        if not original_stopped:
            original_can_stop = n >= ORIGINAL_MIN_REVISIONS
            if original_can_stop and original_eligible:
                original_action = 'accept'
                original_stopped = True
                original_stop = n
            elif n + 1 >= ORIGINAL_MAX_EVALUATIONS:
                original_action = 'exhausted'
                original_stopped = True
                original_stop = n
            else:
                original_action = 'revise'

        enhanced_action = 'after_stop'
        if not reviewed_stopped:
            enhanced_can_stop = n >= ENHANCED_MIN_REVISIONS
            if enhanced_can_stop and enhanced_accepted:
                enhanced_action = 'accept'
                reviewed_stopped = True
                reviewed_stop = n
            elif n + 1 >= ENHANCED_MAX_EVALUATIONS:
                enhanced_action = 'exhausted'
                reviewed_stopped = True
                reviewed_stop = n
            else:
                enhanced_action = 'revise'

        result_rows.append({
            'iteration': n,
            'average': row['average'],
            'accuracy': scores['accuracy'],
            'evidence': scores['evidence'],
            'numeric_checked': row['numeric_checked'],
            'numeric_contradicted': row['numeric_contradicted'],
            'numeric_unverified': row.get('numeric_unverified'),
            'original_threshold_eligible': original_eligible,
            'enhanced_eligible': reviewed_eligible,
            'enhanced_failures': failures,
            'original_action': original_action,
            'enhanced_action': enhanced_action,
            'legacy_unsupported_figures_count': row.get('legacy_unsupported_figures_count'),
        })
    eligible_until_stop = [n for n in enhanced_accepted if reviewed_stop is None or n <= reviewed_stop]
    chosen_enhanced = max(
        eligible_until_stop,
        key=lambda n: (
            history[n]['average'],
            history[n]['scores']['accuracy'],
            history[n]['scores']['evidence'],
        ),
    ) if eligible_until_stop else None
    return {
        'comparison_type': 'matched_score_policy_replay_not_independent_agent_ab_test',
        'original_stop_iteration': original_stop,
        'enhanced_stop_iteration': reviewed_stop,
        'enhanced_selected_accepted_iteration': chosen_enhanced,
        'rows': result_rows,
        'interpretation': (
            'Both policies reuse the same LLM ratings from one enhanced-agent run. '
            'They do not prove different final report quality; the enhanced optimizer '
            'generated the recorded revisions. Unrecognized numeric claims were not audited.'
        ),
    }


def capture_live(ticker: str, destination: Path) -> Path:
    """Collect one live enhanced run, including frozen drafts and findings.

    API/market calls can cost money. Store data only; never save environment keys.
    """
    sys.path.insert(0, str(ROOT))
    from src.graph import run_agent
    from src.evaluation import unsupported_figures

    _, _, state = run_agent(
        ticker.upper(), use_memory=False, save=False, reviewed_evaluation=True
    )
    history = state['history']
    by_iteration = {candidate.iteration: candidate for candidate in state['review_session'].candidates}
    rows = []
    for row in history:
        item = by_iteration[row['iteration']]
        rows.append({
            'iteration': row['iteration'],
            'average': row['average'],
            'scores': row['scores'],
            'numeric_checked': len(item.audit.checks),
            'numeric_contradicted': len(item.audit.contradictions),
            'numeric_unverified': len(item.audit.unverified),
            'legacy_unsupported_figures_count': len(
                unsupported_figures(item.draft, state['findings'])
            ),
            'legacy_unsupported_figures': unsupported_figures(item.draft, state['findings']),
            'draft': item.draft.model_dump(mode='json'),
            'evaluation': item.evaluation.model_dump(mode='json'),
            'numeric_audit': item.audit.rows(),
        })
    document = {
        'ticker': ticker.upper(),
        'source': 'one live enhanced LangGraph run; shared drafts and evaluator outputs',
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'findings': [f.model_dump(mode='json') for f in state['findings']],
        'history': rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding='utf-8')
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Use an existing saved run, no API requests')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--live', metavar='TICKER',
                        help='Optional: run one new enhanced agent and save frozen inputs (paid APIs)')
    args = parser.parse_args()

    if args.live:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.input = ROOT / 'data' / 'evaluation' / f'{args.live.upper()}_{timestamp}_frozen.json'
        capture_live(args.live, args.input)
        args.output = args.input.with_name(args.input.stem + '_comparison.json')

    run = json.loads(args.input.read_text(encoding='utf-8'))
    compared = compare_history(run['history'])
    document = {
        'ticker': run.get('ticker', 'unknown'),
        'source': run.get('source', 'unknown'),
        'comparison': compared,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Source:', run.get('source'))
    print(f"{'Iter':<5} {'Avg':<6} {'Accuracy':<9} {'Original':<11} {'Enhanced':<11} {'Failures'}")
    for row in compared['rows']:
        print(f"{row['iteration']:<5} {row['average']:<6} {row['accuracy']:<9} "
              f"{row['original_action']:<11} {row['enhanced_action']:<11} "
              f"{','.join(row['enhanced_failures']) or 'none'}")
    print('Original stop iteration:', compared['original_stop_iteration'])
    print('Enhanced stop iteration:', compared['enhanced_stop_iteration'])
    print('Enhanced selected accepted iteration:', compared['enhanced_selected_accepted_iteration'])
    print('Saved comparison:', args.output)
    print('NOTE: Paired decision replay, not a separate original-agent execution.')


if __name__ == '__main__':
    main()
