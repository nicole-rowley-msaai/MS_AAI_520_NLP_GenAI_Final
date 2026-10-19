# EquityAgent final submission handoff

## Deliverables

- `notebooks/final_project.ipynb`: **executed** 70-cell notebook. All 39 original code cells and their captured outputs were preserved from the October 18–19 Colab execution. Four Markdown interpretation cells and an explicit rubric map were added. No API calls were made during this editorial update.
- `EquityAgent_Final_Executed.html`: exported from the executed notebook. This is an allowable submission format under the rubric.
- `Results/table1_integrated_live_run.csv`: the three-company Table 1 values transcribed directly from the notebook's actual displayed CSV output, with no invented results.
- `notebooks/build_final.py` and `notebooks/notebook_rubric_finalization.py`: reproducible generation of the edited notebook **without outputs**. The helper is invoked after existing integration and publication passes.

## Important: do not accidentally remove execution outputs

`python notebooks/build_final.py` always produces a fresh **unexecuted** notebook. Do not run the builder after installing the executed notebook unless you intend to rerun it in Colab. The saved HTML is unaffected. You can restore the executed notebook from this ZIP if necessary.

## Install safely

From the repository root on `integration_pawan_ashley`:

```bash
# Confirm branch and save your local copy before replacement
git branch --show-current
cp notebooks/final_project.ipynb ../final_project_before_rubric_finalization.ipynb
# Move ZIP beside the repository, then:
unzip -o ../EquityAgent_submission_review_package.zip -d .
```

Only stage project paths. The ZIP also contains this handoff, a root HTML, and a results CSV; it does not contain credentials. Exclude `.DS_Store`, `.vscode`, `.env`, API credentials, etc.

## Validate without spending API credits

```bash
python -m pytest -q
python - <<'PY'
import json
from pathlib import Path
nb=json.loads(Path('notebooks/final_project.ipynb').read_text())
print('cells',len(nb['cells']))
print('executed code',sum(c.get('execution_count') is not None for c in nb['cells']))
print('errors',sum(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])))
PY
```

Expected: `70` total cells, `39` executed code cells, `0` errors. Previously, 155 project tests passed on the local integration branch. This external editorial pass does not alter executed application code; the tests must still be rerun on your Mac.

## Git author identity

Before your *next* commit, set a GitHub-verified email **you control**:

```bash
git config user.name 'Pawan Bishwokarma'
git config user.email 'YOUR_VERIFIED_GITHUB_EMAIL'
git config --get user.email
```

This fixes **future** attribution, not old commits. Do not run `git filter-branch` or force push near the deadline without a backup and explicit agreement from Nicole and Ashley. The notebook identifies all three team member roles, but that does not substitute for old GitHub commit attribution. Fixing historical commits remains an optional coordinated follow-up.

## Review and publishing

1. Check saved HTML in your browser: section titles, tables, figures, code, and references.
2. Ensure the team's supplementary report Table 1 agrees with `Results/table1_integrated_live_run.csv`.
3. Create a PR from `integration_pawan_ashley` into `main` for Nicole's approval; **do not merge without approval**.
4. If merge succeeds, then update the notebook's checkout branch from `integration_pawan_ashley` to `main` **before a future re-execution**. Do not change the source branch in an already executed document unless the printed checkout log and code actually match.
5. Submit the executed HTML (or PDF exported from it) and optional supplemental report before 11:59 PM October 19. Only one teammate submits.

## Interpretation notes

The main NVDA quality-gate demo accepted draft 2 at 4.0 after rejecting draft 1 at 4.6 for a recognized numeric contradiction. The final draft still contains two **unverified** recognized financial claims. The policy replay is based on common rubric scores, not independent model evaluations. C1's three-ticker comparison is a separate set of live runs and should not be confused with B7's NVDA run.

## Remaining lower priority suggestions

- Complete APA author lists and styling in the references.
- PEP 8 line-length corrections throughout previously existing source scripts.
- Add a brief README pointer to the evaluation modules and notebook 04.
- Coordinate historical Git authorship correction with the team if the assessment requires it.
