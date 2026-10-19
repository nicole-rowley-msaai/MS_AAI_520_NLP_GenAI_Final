# NLP track: news prompt chain and sentiment classifier

Owner: NLP Lead. Covers the Kaggle news EDA and cleaning, the five-step
news prompt chain (Ingest → Preprocess → Classify → Extract → Summarize),
and the classifier evaluation in Part A of the final notebook.

In-text citations follow APA 7 and refer to the AAI-520 module readings
unless marked otherwise; full references are at the end.

## Why these changes

The October 14 run of the final notebook showed three problems in the NLP
track:

- **The classifier over-polarizes.** On the 300-headline Kaggle test set,
  neutral recall was 0.72 while negative recall was 1.00 (precision 0.71)
  and positive precision was 0.69: the model pushed neutral headlines into
  positive or negative. The live NVDA chain then counted 61 positive,
  15 neutral, and 10 negative items.
- **The only baseline was a keyword regex**, so the 0.80 accuracy of the
  LLM had no strong reference point.
- **The live chain had gaps**: a package release from pypi.org passed
  preprocessing as news, the ticker was matched as a substring, the
  extract step returned phrases such as "dozens of AAA games" as figures,
  unlabeled articles were silently counted as neutral, and the batches
  ran one after another.

## Changes

| Change | File | Rationale and source |
|---|---|---|
| Sentiment definition restates the PhraseBank annotation guideline: judge only what the text states, from an investor's view of the stock price | `src/chains/news_chain.py` (`SENTIMENT_RULE`) | The gold labels were made with this instruction (Malo et al., 2014); a prompt that matches the label definition targets the over-polarized neutral class |
| Optional few-shot demonstrations in `classify(examples=...)`, two per label, from the train split and unanimous sentences | `news_chain.py`, `kaggle.fewshot_examples` | In-context learning without weight updates (Brown et al., 2020, Section 2); examples drawn from the training set, never the test set (Brown et al., 2020, Section 2.4); examples help when zero-shot instructions fall short (Stöffelbauer, 2023) |
| Majority vote across the three LLM runs | `kaggle.majority_vote` | Self-consistency: keep the answer most samples agree on (Wang et al., 2023); reuses existing runs, no extra API calls |
| TF-IDF + logistic regression baseline on the train split | `kaggle.tfidf_baseline` | Prompting and training on labeled data are the two ways to apply a pretrained model (Jurafsky & Martin, 2026, Section 9.4); this is a cheap stand-in for the second (Pedregosa et al., 2011) |
| Accuracy by annotator-agreement band | `kaggle.load_agreement`, `score_by_agreement` | PhraseBank groups its 4,840 sentences by how strongly 5 to 8 annotators agreed (Malo et al., 2014) |
| Restored the PhraseBank release files | `data/kaggle/FinancialPhraseBank/` | They were on `data_nlp` but removed from `main` in the Agent v6 commit |
| Dedupe on a normalized key; drop sentences that appear with conflicting labels | `kaggle.normalize_text`, `clean_kaggle`, `cleaning_report` | Text normalization and case folding for matching (Jurafsky & Martin, 2026, Sections 2.4 and 2.6); the classifier still sees cased text, because case carries signal for sentiment (Section 2.6) |
| `train_test()` split; test rows identical to `held_out_sample()` | `kaggle.py` | Keeps the test set fixed while making the train split available |
| Company names matched as whole words with aliases (JPMorgan, JP Morgan, J.P. Morgan); ticker matched in upper case only | `news_chain.mention_pattern` | One normal form for several written forms, and case separating meanings (US vs. us) (Jurafsky & Martin, 2026, Section 2.6) |
| Package and code-host feeds dropped (pypi.org, github.com, ...); per-rule drop counts logged | `news_chain.preprocess` | The NVDA run kept "nvidia-nvcomp-cu13 6.0.0.12" from pypi.org as news |
| Extract step keeps only figures that contain a digit | `src/chains/schemas.py` (`Extraction`) | A figure is a number token such as a price or percentage (Jurafsky & Martin, 2026, Section 2.5.1) |
| Coverage logged per classify call; ids the model skips are listed, not counted as neutral; invented ids ignored | `news_chain.coverage`, `sentiment_counts` | Net sentiment is a count, so a missing label should not become a neutral vote |
| Classify and extract batches run concurrently (4 workers) | `news_chain._run_batches` | The news specialist was the slowest part of a run (about 11 sequential LLM calls for NVDA) |
| NewsAPI `sortBy="relevancy"`; ingest logs the earliest and latest publish time | `src/tools/news.py`, `news_chain.date_span` | With `publishedAt`, the 100-article page fills with the newest items; NVDA's first five fell within 40 minutes |
| Notebook 01: new section on classifier diagnostics; removed the `git config` lines that made every Colab commit appear as one teammate | `notebooks/01_data_and_nlp_starter.ipynb` | Collaboration is measured through GitHub |
| Final notebook builder includes the new section, clones `main` instead of `agent`, and fixes the `SyntaxWarning` in the setup cell | `notebooks/build_final.py` | `main` is where the merged work now lives |

## Results: full run of October 18, 2026

The final notebook was run end to end in Colab on branch `nlp-track-ashley`
with the agent and evaluator code unchanged. The executed copy is
`notebooks/final_project_run_2026-10-18_nlp-track.ipynb`; it ran with no
errors and every tool call succeeded.

**Cleaning.** 8 exact duplicates, 11 duplicates after normalization, and
2 sentences that appear with two different labels. Cleaning keeps 4,833
rows (the previous exact-match rule kept 4,838). The stratified test set
has 300 rows: 178 neutral, 84 positive, 38 negative.

**Classifiers on the test set (notebook sections A3 and A4).**

| Classifier | Accuracy (mean of 3) | Macro-F1 (mean of 3) | Accuracy (vote) | Macro-F1 (vote) |
|---|---|---|---|---|
| Keyword regex | 0.687 | 0.575 | | |
| TF-IDF + logistic regression | 0.793 | 0.771 | | |
| LLM zero-shot | 0.834 | 0.838 | 0.840 | 0.843 |
| LLM few-shot | 0.827 | 0.833 | 0.840 | 0.850 |

The zero-shot LLM leads the supervised baseline by about four points of
accuracy and seven of macro-F1. Neutral recall rose from 0.72 (October 14
run) to 0.85, negative precision from 0.71 to 0.90, and positive precision
from 0.69 to 0.75. That gain combines the guideline-based sentiment
definition with the cleaning, which changed the test sample slightly.
Few-shot demonstrations did not help; the majority vote helped slightly.

**Accuracy by annotator agreement.**

| Agreement band | Test sentences | TF-IDF + LR | LLM zero-shot (vote) | LLM few-shot (vote) |
|---|---|---|---|---|
| 100% | 149 | 0.899 | 0.946 | 0.960 |
| 75-99% | 71 | 0.775 | 0.887 | 0.859 |
| 66-74% | 42 | 0.762 | 0.714 | 0.762 |
| 50-65% | 38 | 0.447 | 0.474 | 0.421 |

Every classifier falls to roughly 45% where only 50-65% of annotators
agreed, so that band is a ceiling set by the labels.

**Impact on the agent (October 14 on `main` against October 18).**

| Measure | Oct 14 | Oct 18 |
|---|---|---|
| NVDA news window | newest items; first five within 40 minutes | Sep 18 to Oct 16 |
| NVDA labels, positive / neutral / negative | 61 / 15 / 10, net positive | 19 / 60 / 13, net neutral |
| News items left unlabeled | not tracked | 0 of 95 |
| NVDA evaluator loop | 3.6 to 4.6 | 4.2 to 4.6 |
| Agent final score, NVDA / JPM / XOM | 4.4 / 4.4 / 4.6 | 4.4 / 4.4 / 4.4 |
| Single-prompt baseline, NVDA / JPM / XOM | 2.2 / 1.8 / 2.0 | 2.0 / 1.8 / 1.8 |
| Unsupported figures in agent briefs | 1 / 5 / 3 | 11 / 3 / 13 |
| Memory off to on | 4.4 to 4.4 | 4.2 to 4.4 |
| Specialists, summed / wall time | 201 s / 170 s | 90 s / 57 s |

- Agent scores moved by at most 0.2, within the evaluator's run-to-run
  variation: the NLP changes alter the findings without measurably
  changing how briefs are judged.
- The NVDA sentiment flip has two causes this run cannot separate: the
  new sentiment definition and a four-week news window.
- The unsupported-figure count is an exact string match. Traceable cases
  in the B8 brief are rounding ($215.9B for $215.94B), units ($150B for
  "$150 billion"), or unlabeled calculations (CPI up 3.72% year over
  year). A figure-by-figure check of the C1 briefs is still open.
- Wall time fell by about two thirds with a similar number of headlines,
  consistent with concurrent classify and extract batches.

## Limitations

- The PhraseBank has been public since 2014, so the LLM may have seen the
  test sentences during pre-training; test-set contamination can inflate
  benchmark scores (Brown et al., 2020, Section 4).
- PhraseBank items are full sentences from company news, many about
  Finnish firms (for example Finnair and Cramo in the few-shot sample);
  live headlines are shorter and noisier, so benchmark accuracy is likely
  an upper estimate for the live chain.
- Fine-tuning an encoder for sequence classification (Jurafsky & Martin,
  2026, Section 9.4.1), as in the Module 3 assignment, would be the
  stronger supervised comparison. It was out of scope for the deadline,
  and transfer learning still needs a substantial amount of labeled data
  (Tunstall et al., 2022, Chapter 1).

## References

Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are
few-shot learners. *Advances in Neural Information Processing Systems,
33*, 1877–1901.

Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing*
(3rd ed. draft; Chapter 2, Regular expressions, tokenization, edit
distance; Chapter 9, Masked language models).
https://web.stanford.edu/~jurafsky/slp3/

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
Good debt or bad debt: Detecting semantic orientations in economic texts.
*Journal of the Association for Information Science and Technology,
65*(4), 782–796. https://doi.org/10.1002/asi.23062

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn:
Machine learning in Python. *Journal of Machine Learning Research, 12*,
2825–2830. (Not a module reading.)

Stöffelbauer, A. (2023, October 24). How large language models work:
From zero to ChatGPT. *Data Science at Microsoft* (Medium).

Tunstall, L., von Werra, L., & Wolf, T. (2022). *Natural language
processing with transformers* (Rev. ed.). O'Reilly Media.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S.,
Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of
thought reasoning in language models. *International Conference on
Learning Representations (ICLR)*. (Not a module reading; cited in the
Module 5 assignment.)
