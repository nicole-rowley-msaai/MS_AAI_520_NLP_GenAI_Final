"""Build notebooks/final_project.ipynb from notebooks 01 and 02.

One setup cell, then Part A (data and NLP, from 01), Part B (agent run,
from 02) and Part C (evaluation, from 02), under the three headings the
assignment asks for. Run from the repo root:

    python notebooks/build_final.py

The result has no outputs; run it top to bottom in Colab and export.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
NB1 = json.loads((HERE / "01_data_and_nlp_starter.ipynb").read_text())
NB2 = json.loads((HERE / "02_agent_run.ipynb").read_text())

REPO_URL = "https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def cells(nb: dict, start: int, stop: int) -> list[dict]:
    """Copy cells [start, stop) with outputs cleared and sources joined."""
    out = []
    for c in nb["cells"][start:stop]:
        c = json.loads(json.dumps(c))
        c["source"] = "".join(c["source"])
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        out.append(c)
    return out


def renumber(section: list[dict], part: str) -> list[dict]:
    """Turn '## 3. Title' into '## B2. Title', counting from 1 per part."""
    n = 0
    for c in section:
        if c["cell_type"] != "markdown":
            continue

        def sub(m):
            nonlocal n
            n += 1
            return f"## {part}{n}. "

        c["source"] = re.sub(r"^## \d+\. ", sub, c["source"], flags=re.M)
    return section


TITLE = f"""# EquityAgent: An Autonomous Agentic Workflow for Investment Research

**GitHub repository:** {REPO_URL} (branch `main`)

MS in Applied AI, University of San Diego. AAI-520: NLP and Generative AI,
final project.

**Team:** Data lead, NLP lead, Agent lead (Nicole Rowley), Evaluation lead.

EquityAgent researches a stock ticker end to end: it plans the research,
calls market, macro, news and SEC-filing tools, runs a five-step prompt
chain over the news, routes each task to a specialist, writes a brief,
grades and revises it, reflects on the run, and stores lessons that change
the next run's plan. Demo tickers: NVDA, JPM, XOM.

| Part | Contents | Assignment heading |
|---|---|---|
| A | Data check, Kaggle news EDA, classifier evaluation | Agent Functions and Capabilities |
| B | One agent run with every step printed | Agent Design and Workflows |
| C | Baseline comparison, memory ablation, limitations | Evaluation and Iteration |

**Models.** Planner, router and news chain: `gpt-6-luna`. Writer and
optimizer: `claude-sonnet-5`. Evaluator: `gpt-6-sol` (a different family
from the writer, so the draft is not graded by the model that wrote it).
Embeddings: `all-MiniLM-L6-v2`.
"""

SETUP = """# --- Colab setup -----------------------------------------------------
import os
import sys

from google.colab import drive, userdata

os.chdir("/content")
drive.mount("/content/drive")

REPO = "/content/drive/MyDrive/equity-agent"
REPO_URL = (
    f"https://{userdata.get('GITHUB_TOKEN')}@github.com/"
    "nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final.git"
)
if os.path.isdir(os.path.join(REPO, ".git")):
    !git -C {REPO} pull -q
else:
    !git clone -q --branch main {REPO_URL} {REPO}
%cd {REPO}
sys.path.insert(0, REPO)

SKIP = r"google-adk\\|dependency resolver"
!pip install -q -r requirements.txt 2>&1 | grep -v "{SKIP}"

SECRET_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FRED_API_KEY",
    "NEWSAPI_KEY",
    "EDGAR_USER_AGENT",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
]
loaded = []
for key in SECRET_KEYS:
    try:
        os.environ[key] = userdata.get(key)
        loaded.append(key)
    except Exception:
        print(f"Missing secret: {key}")
print("Loaded:", loaded)
"""

IMPORTS = """import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="huggingface_hub")

from src import display as show  # noqa: E402
from src.config import TICKERS  # noqa: E402
from src.evaluation import (  # noqa: E402
    baseline_brief,
    quick_smoke,
    score_against_findings,
    tool_success_rate,
    unsupported_figures,
)
from src.graph import run_agent  # noqa: E402
from src.memory import store  # noqa: E402
from src.tools import safe_call  # noqa: E402
from src.tools.retrieval import index_filings  # noqa: E402

pd.set_option("display.max_colwidth", 120)
"""

PART_A = """# Part A. Data and NLP

## Agent Functions and Capabilities

The agent's tools and the news prompt chain are built and tested before
the agent uses them. This part checks each data source (yfinance, FRED,
NewsAPI, SEC EDGAR), explores the Kaggle financial-news dataset used to
benchmark the sentiment classifier, and compares the LLM classifier with
a keyword baseline and a supervised TF-IDF baseline, with zero-shot and
few-shot prompts, broken down by annotator agreement. Every tool call
goes through `safe_call`, so a failed API shows as `ok=False` instead of
stopping the notebook.
"""

PART_B = """# Part B. Agent run

## Agent Design and Workflows

The agent is a LangGraph state graph (`src/graph.py`):

```
ticker -> plan -> route -> specialists -> synthesize -> evaluate --pass--> reflect -> memory
                                                            |                 ^
                                                            +--refine--> optimize
```

| Requirement | Where | Printed below |
|---|---|---|
| Plans research steps | `agents/planner.py` | B2 |
| Uses tools dynamically | `agents/toolkit.py`, `tools/` | B3 |
| Routing | `agents/router.py` | B4 |
| Prompt chaining | `chains/news_chain.py` | B5 |
| Evaluator-optimizer | `agents/evaluator.py` | B7 |
| Self-reflection | `agents/evaluator.py: reflect` | B9 |
| Learns across runs | `memory/store.py` | B10 |

This part runs the agent once on NVDA and prints the output of every step
from the run trace, then runs it a second time to show memory in use.
Memory is cleared first so B2 is a true first run.
"""

RESET_MEMORY = """# Start from empty memory so the first run below is a true first run.
store.NOTES_PATH.unlink(missing_ok=True)
print("Lessons in memory:", len(store.load_notes()))
"""

NEWS_NOTE = """The digest's `net_sentiment` is not the model's opinion: the chain counts
the per-article labels from the classify step and applies majority rule
with a 15-point margin (`sentiment_counts` is printed with the summary).
Empty `figures` lists in the extract step are expected for headline-only
items; the extractor only reports numbers that appear in the text.
"""

PART_C = """# Part C. Evaluation

## Evaluation and Iteration

Part B showed one evaluator-optimizer loop. This part measures the agent
against alternatives, all scored with the same rubric and the same
findings so the comparison is fair.

The numeric spot-check (`unsupported_*`) counts figures in a brief that
appear in no finding. A figure the writer labels as calculated (for
example "a calculated ~65% increase") is exempt: the evaluator asks for
that label, and such figures are arithmetic on the findings rather than
claims needing their own source. The baseline never sees the findings,
so its evidence and accuracy scores are structurally low; the comparison
measures whether tools and the loop produce a grounded brief.
"""

SUMMARY = """## Summary

- **Baseline vs agent.** The single-prompt baseline has no tools, so its
  figures cannot be checked against findings; the agent's can, and the
  `unsupported_*` columns above count figures that appear in no finding.
- **Evaluator-optimizer.** Every draft gets at least one revision; the
  B7 critiques and the B8 report show what changed.
- **Memory.** The score is similar with memory on and off; the benefit
  shows in the plan (extra filing searches and news cross-checks that
  answer the previous run's lessons), not in the rubric average.
- **Robustness.** Each tool call is wrapped so a failed API degrades the
  brief instead of ending the run, and a model that keeps returning
  truncated JSON is retried on the other provider.
- **Speed.** The three specialists run in parallel; the timing table
  shows their summed time (`specialist`) against their wall time
  (`specialists`).
- **Known limits.** The reflection step inherits the model's priors and
  has flagged correct current data (a 10-year yield above 5%, an analyst
  target well above the price) as suspicious; segment-level revenue and
  peer comparables are the data gaps it raises on every run.
"""

# 01: data check, Kaggle EDA, classifier evaluation (after its setup)
part_a = renumber(cells(NB1, 4, 23), "A")  # 01's chain demo = B5

# 02: smoke test through memory (after its setup); the smoke-test heading
# in 02 carries the assignment heading, which Part A now owns.
part_b = cells(NB2, 4, 28)
part_b[0]["source"] = "## 1. Setup check: every tool returns data\n"
part_b = renumber(part_b, "B")
# 02 put the "Evaluation and Iteration" heading above section 7; Part C
# owns it here.
for c in part_b:
    if c["cell_type"] == "markdown":
        c["source"] = c["source"].replace(
            "## Evaluation and Iteration\n\n", ""
        )

# 02: section 11 and limitations
part_c = cells(NB2, 28, len(NB2["cells"]))
part_c[0]["source"] = re.sub(
    r"^## 11\. Evaluation\n+", "", part_c[0]["source"]
)
part_c[0]["source"] = "## 1. Three comparisons\n\n" + part_c[0]["source"]
for c in part_c:
    if c["cell_type"] == "markdown" and c["source"].startswith(
        "## Limitations"
    ):
        c["source"] = c["source"].replace(
            "## Limitations", "## 2. Limitations"
        )
part_c = renumber(part_c, "C")

# Insert the memory reset before the first run and the sentiment note
# after the news-chain heading.
first_run = next(
    i for i, c in enumerate(part_b) if "run_agent(TICKER)" in c["source"]
)
part_b.insert(first_run, code(RESET_MEMORY))
news = next(
    i for i, c in enumerate(part_b) if "News prompt chain" in c["source"]
)
part_b.insert(news + 1, md(NEWS_NOTE))

final = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        md(TITLE),
        md("## Setup"),
        code(SETUP),
        code(IMPORTS),
        md(PART_A),
        *part_a,
        md(PART_B),
        *part_b,
        md(PART_C),
        *part_c,
        md(SUMMARY),
    ],
}

out = HERE / "final_project.ipynb"
out.write_text(json.dumps(final, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} with {len(final['cells'])} cells")
