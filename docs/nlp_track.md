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

## Offline results (no API calls)

Run on the PhraseBank 50%-agreement file, which holds the same 4,846
sentences as the Kaggle CSV.

**Cleaning.** 8 exact duplicates, 11 duplicates after normalization, and
2 sentences that appear with two different labels. Cleaning keeps 4,833
rows (the previous exact-match rule kept 4,838). The stratified test set
has 300 rows: 178 neutral, 84 positive, 38 negative. Because five more
rows are removed, the test set differs slightly from the October 14 run,
so every classifier must be rerun on it.

**Baselines on the test set.**

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| Keyword regex | 0.687 | 0.575 |
| TF-IDF + logistic regression (train split) | 0.793 | 0.771 |
| LLM zero-shot, October 14 run (earlier, near-identical split) | 0.804 | 0.810 |

The supervised baseline is within about one point of the LLM on accuracy;
the LLM's advantage is mostly macro-F1, that is, the minority classes.

**Accuracy by annotator agreement (TF-IDF + logistic regression).**

| Agreement band | Test sentences | Accuracy | Macro-F1 |
|---|---|---|---|
| 100% | 149 | 0.899 | 0.882 |
| 75-99% | 71 | 0.775 | 0.662 |
| 66-74% | 42 | 0.762 | 0.771 |
| 50-65% | 38 | 0.447 | 0.396 |

Accuracy falls steeply where the annotators disagreed. The notebook
computes the same table for the zero-shot and few-shot LLM runs.

## To do in Colab before submission

1. Rerun the final notebook top to bottom. If an existing Drive clone is
   on the `agent` branch, run `git -C /content/drive/MyDrive/equity-agent
   checkout main` first.
2. Update the notebook's Summary table row for sentiment classification
   with the new A3 and A4 numbers.
3. Report whether few-shot prompting and the majority vote changed neutral
   recall, and whether the live NVDA sentiment counts moved.

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
