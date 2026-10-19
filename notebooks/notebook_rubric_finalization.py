"""Final editorial layer for the EquityAgent AAI-520 notebook.

Makes grading requirements explicit and explains outputs without changing
computational code or fabricating results. Designed to run AFTER the notebook
integration and publication transforms. Calling twice is idempotent.
"""
from __future__ import annotations

import hashlib


def _text(cell: dict) -> str:
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else source


def _set(cell: dict, content: str) -> None:
    cell['source'] = content


def _markdown(content: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': content,
            'id': hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]}


def _locate(cells: list[dict], needle: str) -> int:
    for i, cell in enumerate(cells):
        if needle in _text(cell):
            return i
    raise ValueError(f'Expected notebook heading not found: {needle}')


RUBRIC_TABLE = '''### Assessment requirements and implementation map

The notebook explicitly demonstrates the **four agent functions** and the
**three workflow patterns** required by the assignment.

| Grading area | Required capability | Execution evidence |
|---|---|---|
| Agent Functions | Plans research steps for a stock symbol | **B2**: research plan and task sequence |
| Agent Functions | Uses tools dynamically through APIs, datasets, and retrieval | **B3**: tool calls, inputs, and results |
| Agent Functions | Self-reflects on the quality of its output | **B9**: reflection, unsupported claims, lessons |
| Agent Functions | Learns across runs using persistent notes | **B10**: lessons saved and retrieved by a subsequent plan |
| Workflow Patterns | **Prompt Chaining:** Ingest → Preprocess → Classify → Extract → Summarize | **B5**: intermediate outputs for each news processing stage |
| Workflow Patterns | **Routing:** Direct research tasks to appropriate specialists | **B4**: earnings, market, and news routing decisions |
| Workflow Patterns | **Evaluator–Optimizer:** Generate → Evaluate → Refine | **B7**: scored draft iterations, feedback, numerical audit, and final selection |

The writer and optimizer use **Claude Sonnet 5** (`claude-sonnet-5`), while
**GPT-6 Sol** (`gpt-6-sol`) grades their drafts against the rubric. These
roles separate report generation from LLM-based evaluation; an additional
numeric audit checks only the claims it can recognize.
'''

A4_NOTE = '''### Interpretation of A4: financial sentiment classification

On the 300-headline held-out test set, the **few-shot LLM majority vote**
recorded macro-F1 **0.848**, compared with **0.840** for zero-shot voting,
**0.771** for TF–IDF logistic regression, and **0.575** for the keyword
baseline. The small difference between the two LLM prompting conditions
should not be treated as statistically established. Accuracy also depends on
label consensus: few-shot accuracy was **96.0%** for unanimous headlines,
versus **44.7%** for the lowest agreement band (50–65%). This underscores
the difficulty of inherently ambiguous financial headlines and the limits
of evaluating on a public benchmark.
'''

B7_NOTE = '''### Interpretation of B7: evaluation and iterative revision

The first NVDA draft scored **4.0/5**, but its **accuracy and evidence scores
were each 3/5**, below the explicit quality thresholds. The next draft scored
**4.6/5**, yet a recognized numerical contradiction prevented acceptance.
Following another revision, the accepted draft scored **4.0/5** with no
contradictions among **three recognized numerical checks**. Two of those
checks remained **unverified**, so zero detected contradictions must not be
interpreted as comprehensive financial fact checking. The final selected
draft scored lower on the LLM rubric than the preceding candidate because
the policy gives verified contradictions precedence over the average score.
'''

C1_NOTE = '''### Interpretation of C1: comparative research and memory results

In the separate comparison runs for **NVDA, JPM, and XOM**, the no-tools
baselines scored **1.8, 2.0, and 1.6**, respectively; the research agent's
final drafts each scored **4.4/5**. This is evidence of better *rubric scores*
when specialists have access to retrieved findings, not independent proof of
financial accuracy. The tool success rate reported for these runs was
**100%**. The XOM evaluation selected iteration **0** despite evaluating two
drafts, illustrating the ability to retain an earlier preferred candidate.

In the memory ablation, the memory-off plan had **7 steps** and read
**0 lessons**; the memory-on plan had **8 steps** and read **8 lessons**.
Their final scores were **4.4** and **4.2**, respectively. This demonstrates
that stored lessons influenced planning, but does not establish a quality
improvement. The numeric parser examined only **4, 1, and 0** claims in the
three accepted research briefs, respectively, so financial coverage remains
limited. The exact-string unsupported-figure measure can also count benign
rounding or unit differences.
'''

C2_NOTE = '''### Interpretation of C2: matched evaluator policy comparison

The comparison uses the **same observed NVDA draft sequence and LLM scores**,
not two independently generated reports. Under the baseline stopping policy,
iteration **1** would be accepted with a **4.6/5** average. The enhanced
policy instead rejected iteration 1 because of a recognized numeric
contradiction and accepted iteration **2** at **4.0/5** after that detected
contradiction was removed. This illustrates how the two acceptance rules
differ; it does **not** demonstrate that one produces generally superior
investment recommendations. The enhanced policy's choice should also be
interpreted alongside the two unverified figures in its final numeric audit.
'''

B7_REVISED = '''## B7. Evaluator–Optimizer: generate, evaluate, and refine

The research agent synthesizes an initial brief, scores five criteria from
1 to 5 (**completeness, evidence, accuracy, balance, and clarity**), then
uses specific evaluation feedback to revise the brief. Claude Sonnet 5 writes
and optimizes the drafts; GPT-6 Sol evaluates them. These are distinct model
roles, rather than two independent scoring models.

The **baseline acceptance policy** requires an average of at least 4.0 after
the minimum revision count. The **enhanced quality policy** additionally
requires accuracy and evidence scores of at least 4/5, blocks acceptance on
recognized numerical contradictions, records unresolved figures, and can
select the highest scoring acceptable revision.

| Acceptance rule | Baseline | Enhanced policy |
|---|---|---|
| Minimum average | 4.0 | 4.0 |
| Minimum accuracy | No separate floor | 4 / 5 |
| Minimum evidence | No separate floor | 4 / 5 |
| Minimum revision rounds | 1 | 1 |
| Maximum evaluated drafts | 3 | 3 |
| Contradictions found by limited numeric parser | Not a separate gate | Block acceptance |
| Final draft choice | Last evaluated revision | Best accepted revision |

Unrecognized or unsupported claims are **unverified**, not proven correct.
The parser covers only a narrow subset of financial prose; discrepancies
require human review. Exhausting the evaluation budget does not constitute
acceptance.
'''

C2_REVISED = '''## C2. Paired Evaluator–Optimizer acceptance analysis

This is a controlled **decision-policy ablation** using the exact same
recorded NVDA drafts and LLM rubric scores. Both acceptance rules rely on
the same GPT-6 Sol evaluator, while Claude Sonnet 5 is responsible for draft
creation and revision. The exercise compares stopping decisions; it is not
a second independent live run or an experiment with a newly trained judge.

The baseline policy requires at least one revision and average score 4.0 or
higher. The enhanced policy adds separate evidence and accuracy thresholds,
a contradiction gate for recognizable financial claims, and retention of an
acceptable earlier draft when a subsequent revision is worse. Changes in
acceptance logic alone should not be interpreted as validated gains in
financial correctness or investment performance.
'''

REFERENCES = '''### References

- Brown, T. B., et al. (2020). *Language models are few-shot learners.*
  Advances in Neural Information Processing Systems, 33, 1877–1901.
  https://arxiv.org/abs/2005.14165
- Jurafsky, D., & Martin, J. H. (n.d.). *Speech and language processing*
  (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
- Lewis, P., et al. (2020). *Retrieval-augmented generation for
  knowledge-intensive NLP tasks.* Advances in Neural Information Processing
  Systems, 33. https://arxiv.org/abs/2005.11401
- Madaan, A., et al. (2023). *Self-refine: Iterative refinement with
  self-feedback.* Advances in Neural Information Processing Systems, 36.
  https://arxiv.org/abs/2303.17651
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
  Good debt or bad debt: Detecting semantic orientations in economic texts.
  *Journal of the Association for Information Science and Technology, 65*(4),
  782–796. https://doi.org/10.1002/asi.23062
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023).
  *Reflexion: Language agents with verbal reinforcement learning.* Advances
  in Neural Information Processing Systems, 36.
  https://arxiv.org/abs/2303.11366
- Wang, X., et al. (2023). *Self-consistency improves chain of thought
  reasoning in language models.* International Conference on Learning
  Representations. https://arxiv.org/abs/2203.11171
- Yao, S., et al. (2023). *ReAct: Synergizing reasoning and acting in language
  models.* International Conference on Learning Representations.
  https://arxiv.org/abs/2210.03629
'''


def finalize_for_rubric(notebook: dict) -> dict:
    cells = notebook['cells']
    # Retain the provenance and rubric evidence already present in the notebook.
    overview = cells[_locate(cells, '## Research questions and experimental design')]
    text = _text(overview)
    if '### Assessment requirements and implementation map' not in text:
        _set(overview, text.rstrip() + '\n\n' + RUBRIC_TABLE + '\n')

    i = _locate(cells, '## B7. Evidence-based evaluation and revision') if any(
        '## B7. Evidence-based evaluation and revision' in _text(c) for c in cells
    ) else _locate(cells, '## B7. Evaluator–Optimizer:')
    _set(cells[i], B7_REVISED)

    c2 = cells[_locate(cells, '## C2. Paired')]
    _set(c2, C2_REVISED)

    replay = cells[_locate(cells, '# Replay Nicole\'s original stopping rule')]
    _set(replay, _text(replay).replace(
        "# Replay Nicole's original stopping rule on the SAME recorded rubric scores.\n"
        "# This never calls an LLM and does not manufacture a second model output.",
        "# Compare baseline and enhanced decisions on the same scored drafts.\n"
        "# This is a policy replay, not another independent model evaluation."
    ))

    for anchor, heading, content in (
        ('## A5. Interpretation of sentiment results', '### Interpretation of A4:', A4_NOTE),
        ('## B8. Final report', '### Interpretation of B7:', B7_NOTE),
        ('## C2. Paired', '### Interpretation of C1:', C1_NOTE),
        ('## C3. Limitations and threats to validity', '### Interpretation of C2:', C2_NOTE),
    ):
        if not any(heading in _text(c) for c in cells):
            cells.insert(_locate(cells, anchor), _markdown(content))

    ending = cells[_locate(cells, '## Discussion and conclusion')]
    text = _text(ending)
    old_start = text.find('The Evaluation Lead contribution is a')
    old_end = text.find('### Reproducibility and implementation')
    if old_start != -1 and old_end != -1:
        text = (text[:old_start]
                + 'The system combines a common five-criterion rubric with explicit\n'
                + 'quality thresholds, selective financial-claim checks, revision\n'
                + 'feedback, and selection of the strongest qualifying draft.\n'
                + 'These controls improve auditability; broader claims of financial\n'
                + 'accuracy require independent validation over more companies.\n\n'
                + text[old_end:])
    old_disclosure = text.find('**Development disclosure:**')
    if old_disclosure != -1:
        next_ref = text.find('### References', old_disclosure)
        if next_ref != -1:
            text = (text[:old_disclosure]
                    + '**AI assistance disclosure:** ChatGPT assisted with code\n'
                    + 'development, debugging, test design, and technical writing,\n'
                    + 'particularly for the evaluation framework. The team reviewed\n'
                    + 'and tested the submitted implementation and reports the\n'
                    + 'measured notebook results separately from illustrative tests.\n\n'
                    + text[next_ref:])
    idx = text.find('### References')
    if idx != -1:
        text = text[:idx] + REFERENCES
    else:
        text += '\n\n' + REFERENCES
    _set(ending, text)
    return notebook
