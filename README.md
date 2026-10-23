
# Agentic AI Risk Investigation & Reporting Automation

**An AI-powered workflow for portfolio risk monitoring, automated exception investigation, and evidence-grounded risk reporting.**

[**Live Dashboard**](https://agentic-ai-risk-workflow-1826.streamlit.app/)

## Overview

This project demonstrates an end-to-end, agentic AI workflow that integrates quantitative risk analytics, automated exception detection, AI-assisted investigation, and structured risk reporting.

The system monitors a multi-asset portfolio, evaluates risk metrics against configurable limits, and identifies exceptions requiring further investigation. An AI agent can then select and execute analytical tools, interpret the results, and generate an evidence-grounded investigation report.

Unlike a conventional risk dashboard, the project focuses on **automating the investigation process**, from detecting unusual risk conditions to gathering supporting evidence and preparing findings for human review.

The application is built around three principles:

- **Automated monitoring:** Identify risk exceptions systematically using predefined metrics and policies.
- **Evidence-grounded investigation:** Use quantitative tools and recorded evidence to support AI-generated findings.
- **Transparency and governance:** Maintain investigation traces, analytical limitations, and explicit human-review requirements.

## Key Capabilities

### 1. Portfolio Risk Monitoring

A Python-based risk engine calculates and monitors portfolio-level and instrument-level risk metrics.

**Risk analytics include:**
- Historical Simulation Value at Risk (VaR) and Expected Shortfall (ES)
- Daily P&L, portfolio returns, and annualized volatility
- Portfolio drawdown and concentration
- Equity, rates, and credit exposures
- Instrument-level holdings and portfolio weights
- Historical risk trends and portfolio performance

The Streamlit dashboard provides interactive visualizations, historical snapshots, and portfolio monitoring views.

### 2. Automated Risk Limit Evaluation

A configurable policy engine evaluates portfolio metrics against predefined risk thresholds.

Each evaluation is classified as:

| Status | Description |
|---|---|
| `PASS` | Metric satisfies the configured risk limit |
| `WARNING` | Metric falls within the warning range |
| `BREACH` | Metric exceeds the configured breach threshold |
| `NOT_EVALUATED` | Insufficient data or another condition prevents evaluation |

The system records policy evaluations and identifies exceptions requiring investigation.

Non-PASS results can be routed to the AI investigation workflow, with evaluation reasons preserved when metrics cannot be assessed.

### 3. Agentic AI Risk Investigation

The AI investigation orchestrator uses the OpenAI API to investigate detected risk exceptions.

Rather than following a fixed sequence of calculations, the agent can dynamically select and execute analytical tools based on the exception, available evidence, and investigation requirements.

**Available analytical tools:**

| Tool | Investigation Purpose |
|---|---|
| `get_risk_metrics` | Retrieve portfolio risk and performance metrics, including VaR, ES, volatility, and drawdown |
| `retrieve_risk_policy` | Retrieve risk limits, thresholds, and policy definitions |
| `analyze_positions` | Analyze portfolio exposures, concentration, and changes in position weights |
| `analyze_transactions` | Review trading activity, transaction history, and position changes |
| `get_pnl_attribution` | Identify instrument-level and asset-class P&L contributors |
| `get_var_attribution` | Analyze historical VaR/ES results and tail-risk contributions |
| `analyze_risk_factors` | Examine historical market movements and portfolio risk factors |
| `get_market_context` | Retrieve relevant market context to support investigation of potential risk drivers |
| `run_stress_test` | Evaluate portfolio risk under historical and hypothetical stress scenarios |
| `check_data_quality` | Assess data completeness, consistency, and reconciliation |

The workflow supports dynamic tool selection, structured evidence collection, and investigation across different types of risk exceptions.

### 4. AI-Generated Investigation Reports

The investigation workflow generates structured reports containing:

1. **Executive Summary** — Key findings and risk implications
2. **Exception and Observed Risk** — Breach conditions, metrics, and policy context
3. **Supporting Evidence** — Quantitative findings and analytical results
4. **Potential Drivers** — Possible explanations supported by available evidence
5. **Limitations** — Data, methodology, and investigation constraints
6. **Recommended Human Review** — Follow-up analysis and review considerations

Reports distinguish observed facts from potential explanations and avoid presenting unsupported conclusions as verified findings.

### 5. Investigation Trace & Governance

Each investigation records supporting information for transparency and review.

- Tool calls and execution results
- Structured evidence records
- Model requests and investigation steps
- Tool validation errors and execution status
- Model token usage and estimated API costs
- Investigation status and human-review requirements

The AI supports analytical investigation and report preparation; it does not independently approve risk exceptions, change risk limits, or execute trades.

## Workflow Architecture

```text
Portfolio Data
      |
      v
Quantitative Risk Engine
      |
      v
Risk Limit Evaluation
      |
      v
Automated Exception Detection
      |
      v
AI Investigation Orchestrator
      |
      +----> Risk Metrics & Policy Retrieval
      |
      +----> Position & Transaction Analysis
      |
      +----> P&L and VaR/ES Attribution
      |
      +----> Risk Factor Analysis
      |
      +----> Market Context Analysis
      |
      +----> Run Stress Test
      |
      +----> Data Quality Checks
      |
      v
Evidence Collection & Investigation Trace
      |
      v
AI-Generated Risk Investigation Report
      |
      v
Human Review & Governance
```

The quantitative risk engine and rule-based monitoring operate independently of the AI model. AI is introduced at the investigation stage to interpret evidence, select additional analytical tools, and prepare a structured report.

## Interactive Dashboard

The application is deployed using Streamlit Community Cloud.

**[Launch the Live Dashboard](https://agentic-ai-risk-workflow-1826.streamlit.app/)**

The dashboard includes three main workspaces:

| Workspace | Features |
|---|---|
| Portfolio Overview | Portfolio construction, performance, historical risk metrics, and interactive charts |
| Risk Limit Monitoring | Policy evaluations, risk-limit status, and exception monitoring |
| AI Investigations | Saved AI investigation reports, supporting evidence, execution traces, and governance |

The public demonstration uses pre-generated investigation reports. Visitors can explore the workflow and its outputs without initiating new OpenAI API requests.

## Illustrative Investigation Cases

Two saved investigation cases demonstrate how the workflow handles different types of risk exceptions.

### Case 1: Equity Exposure Limit Breach

**Date:** September 30, 2026  
**Exception:** `EQUITY_EXPOSURE_LIMIT`

| Metric | Value |
|---|---|
| Portfolio NAV | Approximately $188.29 million |
| Equity exposure | 70.46% of NAV |
| Configured breach threshold | 65.00% |
| Excess exposure | 5.46 percentage points |
| Exception status | BREACH |

**Investigation focus:**

- Identify the portfolio positions contributing to equity concentration
- Examine changes in instrument weights and asset-class exposures
- Review transaction activity and potential market-driven exposure changes
- Assess supporting risk attribution and analytical limitations

### Case 2: Daily P&L / VaR Exceedance

**Date:** April 4, 2025  
**Exception:** `PNL_VAR_EXCEEDANCE`

| Metric | Value |
|---|---|
| Portfolio NAV | Approximately $136.83 million |
| Daily P&L | -$5.02 million |
| Prior-day 99% VaR | Approximately $2.49 million |
| Loss / prior-day VaR | 2.01x |
| Exception status | BREACH |

**Investigation focus:**

- Identify the primary contributors to the portfolio loss
- Analyze equity, rates, and credit performance
- Review position concentration and recent trading activity
- Examine historical risk factors and tail-risk attribution
- Assess data quality and methodological limitations

Both cases use illustrative portfolio data and demonstrate evidence-based investigation rather than automated trading or risk approval.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Quantitative analytics | pandas, NumPy |
| Risk configuration | YAML |
| AI integration | OpenAI API |
| Interactive dashboard | Streamlit |
| Visualization | Plotly |
| Automated testing | pytest |
| Version control | Git, GitHub |
| Cloud deployment | Streamlit Community Cloud |

## Repository Structure

```text
agentic-ai-risk-workflow/
|
|-- app/
|   |-- investigation.py
|
|-- config/
|   |-- portfolio.yaml
|   |-- risk_metrics.yaml
|   |-- risk_limits.yaml
|   |-- stress_scenarios.yaml
|
|-- data/
|   |-- sample/
|   |-- outputs/
|
|-- src/
|   |-- portfolio/
|   |-- risk/
|   |-- agent/
|   |-- reporting/
|   |-- governance/
|
|-- scripts/
|   |-- generate_sample_data.py
|   |-- run_live_investigation.py
|
|-- tests/
|
|-- requirements.txt
|-- README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-ai-risk-workflow.git
cd agentic-ai-risk-workflow
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
python -m streamlit run app/investigation.py
```

The dashboard loads the included illustrative portfolio datasets and saved investigation outputs.

### 5. Run Automated Tests

```bash
python -m pytest -q
```

### 6. Run an AI Investigation (Optional)

To generate a new investigation, configure an OpenAI API key in your local environment.

For example, using a local `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit the `.env` file or API credentials to GitHub.

List available exceptions:

```bash
python -m scripts.run_live_investigation --list
```

Start an interactive investigation:

```bash
python -m scripts.run_live_investigation
```

Or select a specific exception:

```bash
python -m scripts.run_live_investigation \
  --portfolio PORT_001 \
  --date 2025-04-04 \
  --rule PNL_VAR_EXCEEDANCE
```

New investigations require API access and may incur usage costs.

## Methodology & Design Considerations

### Quantitative Risk Measurement

The project uses historical portfolio data to calculate risk metrics, including Historical Simulation VaR and Expected Shortfall.

The analytical engine operates independently from AI-generated commentary, helping maintain separation between quantitative calculations and model interpretation.

### Configurable Risk Policies

Risk limits are defined in YAML configuration files rather than embedded directly into investigation prompts.

This supports more consistent monitoring and allows risk thresholds to be modified without redesigning the AI workflow.

### Evidence-Grounded Reasoning

The investigation orchestrator collects analytical tool results and uses them as the basis for report generation.

Reports are intended to distinguish verified observations, possible drivers, and unresolved questions.

### Execution Controls

The orchestrator includes configurable execution limits, including model-call budgets, tool-call budgets, and estimated API cost monitoring.

These controls help constrain investigation execution and support traceability.

### Human-in-the-Loop Governance

Investigation reports are intended to support human risk review. The workflow does not autonomously approve breaches, implement portfolio changes, or replace formal governance processes.

## Limitations

- The portfolio data is synthetic and does not represent a live investment account.
- Risk limits are illustrative and are not institutionally approved policies.
- ETF asset-class classifications are simplified exposure proxies and do not replace detailed sensitivity measures such as DV01 or CS01.
- Historical Simulation risk estimates depend on the available observations and chosen methodology.
- AI-generated findings may be incomplete and require independent validation.
- The public dashboard displays selected saved investigations rather than enabling unrestricted live AI execution.

## Disclaimer

This repository is intended for technical demonstration, research, and educational purposes only.

It does not provide investment advice, operate as a regulated risk management service, or execute investment transactions.

---

*Sponsored by **[SilverStone Risk Intelligence](https://silverstonerisk.com/)**.*
