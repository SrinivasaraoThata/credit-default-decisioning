# Credit Default Risk Model + Agentic Decisioning Layer

Work in progress. See [CLAUDE.md](CLAUDE.md) for the full project plan.

## Problem statement

Predict which credit card applicants are likely to default within the review
period, balancing predictive power against explainability and real-world
practicality. The model is wrapped in an agentic decisioning layer that
explains its decisions using retrieved credit policy language and routes
uncertain cases to human review instead of guessing.

All data used here is public (UCI Default of Credit Card Clients dataset). No
real or proprietary applicant, account, or assessment data is involved.

## Results so far (Day 1)

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.391 | 0.614 | 0.477 |
| XGBoost | 0.776 | 0.474 | 0.614 | 0.535 |

XGBoost is the stronger model and is carried forward for the agentic
decisioning layer. Full write-up, SHAP findings, and the fairness check are
in [docs/day1_findings.md](docs/day1_findings.md).

## Status

Day 1 (modeling) done. Day 2 (agentic decisioning layer) next.
