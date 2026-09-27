# MS_AAI_520_NLP_GenAI_Final
Develop an autonomous Investment Research Agent using dynamic tools, self-reflection, learning, prompt chaining, routing, and iterative evaluation to analyze stocks and markets.

## Project Overview

This project develops an **autonomous Investment Research Agent** that researches and analyzes stocks using agentic AI workflows. The system integrates financial data, news, and company information to generate, evaluate, and refine market research.

### Agent Functions

The Investment Research Agent will:

- **Plan** research steps for a given stock symbol.
- **Use tools dynamically**, including APIs, datasets, and retrieval systems.
- **Self-reflect** to assess the quality and completeness of its analysis.
- **Learn across runs** by maintaining brief notes or memories that improve future research.

### Workflow Patterns

The system implements three agentic workflow patterns:

1. **Prompt Chaining:** Ingest News → Preprocess → Classify → Extract → Summarize
2. **Routing:** Direct information to specialized agents, such as earnings, news, or market analyzers.
3. **Evaluator–Optimizer:** Generate Analysis → Evaluate Quality → Refine Using Feedback

## Technologies & Data Sources

Potential financial data sources include:

- **Yahoo Finance (`yfinance`)** — stock prices and financial data
- **Financial News** — Kaggle datasets, NewsAPI, Reuters, and Yahoo Finance News
- **FRED API** — economic and macroeconomic data
- **SEC EDGAR** — company filings
- **Alpha Vantage** — financial market data

Development will use **Python** following **PEP 8** style guidelines, with **GitHub** for version control and team collaboration.

## Deliverables

The final project will include:

- A complete **team code notebook** submitted as PDF or HTML
- A link to the **GitHub repository**
- Notebook documentation covering:
  - Agent design and workflows
  - Agent functions and capabilities
  - Evaluation and iteration
- An optional **supplemental PDF report** for additional project documentation
