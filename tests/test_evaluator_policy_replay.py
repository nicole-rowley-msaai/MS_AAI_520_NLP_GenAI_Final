"""Offline acceptance gate comparison regression tests (no API calls)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_evaluator_policies.py'
spec = importlib.util.spec_from_file_location('paired_comparison', SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def row(i, accuracy, evidence=5, completeness=5, balance=5, clarity=5,
        contradictions=0, checked=2):
    scores = dict(accuracy=accuracy, evidence=evidence, completeness=completeness,
                  balance=balance, clarity=clarity)
    return {'iteration': i, 'scores': scores,
            'average': sum(scores.values()) / 5,
            'numeric_checked': checked,
            'numeric_contradicted': contradictions}


def test_logged_nvda_policies_both_end_on_revision_one():
    import json
    source = Path(__file__).resolve().parents[1] / 'data/evaluation/nvda_oct18_logged.json'
    history = json.loads(source.read_text())['history']
    out = comparison.compare_history(history)
    assert out['original_stop_iteration'] == 1
    assert out['enhanced_stop_iteration'] == 1
    assert out['enhanced_selected_accepted_iteration'] == 1
    assert out['rows'][0]['enhanced_failures'] == ['accuracy_below_threshold']


def test_original_can_accept_low_accuracy_after_min_revision():
    history = [row(0, 3), row(1, 3, completeness=5, evidence=5, balance=5, clarity=5), row(2, 5)]
    out = comparison.compare_history(history)
    assert out['rows'][1]['original_action'] == 'accept'
    assert out['rows'][1]['enhanced_action'] == 'revise'
    assert out['enhanced_stop_iteration'] == 2


def test_contradiction_blocks_enhanced_but_not_original():
    history = [row(0, 3), row(1, 5, contradictions=1), row(2, 5)]
    out = comparison.compare_history(history)
    assert out['rows'][1]['original_action'] == 'accept'
    assert out['rows'][1]['enhanced_action'] == 'revise'
    assert out['rows'][1]['enhanced_failures'] == ['numeric_contradictions']


def test_unverified_or_missing_recognition_does_not_block():
    history = [row(0, 5, checked=0), row(1, 5, checked=0)]
    out = comparison.compare_history(history)
    assert out['rows'][1]['enhanced_action'] == 'accept'


def test_policies_require_one_revision_even_perfect_initial_draft():
    out = comparison.compare_history([row(0, 5), row(1, 5)])
    assert out['rows'][0]['original_action'] == 'revise'
    assert out['rows'][0]['enhanced_action'] == 'revise'


def test_bad_average_rejected():
    history = [row(0, 5), row(1, 5)]
    history[1]['average'] = 1.0
    with pytest.raises(ValueError):
        comparison.compare_history(history)


def test_no_history_raises():
    with pytest.raises(ValueError):
        comparison.compare_history([])


def test_best_accepted_earlier_draft_survives_later_bad_revision():
    history = [row(0, 5), row(1, 3)]
    out = comparison.compare_history(history)
    assert out['rows'][1]['enhanced_action'] == 'accept'
    assert out['enhanced_selected_accepted_iteration'] == 0
