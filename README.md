# Credit Default Risk Model + Agentic Decisioning Layer

Work in progress.

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

## Agentic decisioning layer (Day 2)

Four agents, wired with LangGraph:

1. **Intake** - validates the application against the expected schema,
   rejects malformed input before it reaches the model.
2. **Risk scoring** - calls the trained XGBoost model directly (a function
   call, not an LLM) and explains the score with SHAP values for that one
   prediction.
3. **Policy explanation** - retrieves relevant credit policy text from a
   local Chroma store and asks Gemini to explain the risk factors in plain
   language, grounded in that retrieved text. Demographic fields (sex, age,
   marital status, education) are never cited as a reason, even though they
   are model inputs.
4. **Decision** - approve / deny / manual review, using risk bands rather
   than a single probability cutoff. Day 1's fairness check found a
   noticeably higher false-positive rate for male applicants than female
   applicants at a fixed threshold; a single cutoff turns every borderline
   case into a confident automated decision right where the model is least
   reliable. The middle band is routed to manual review instead.

Wrapped in FastAPI: `POST /applications/submit` runs the full pipeline and
returns the decision plus a reasoning trace from each agent,
`GET /applications/{id}/status` retrieves a past decision, `GET /health` is
the health check.

Run it locally:

```bash
cp .env.example .env   # add a free Google AI Studio key for GOOGLE_API_KEY
uvicorn api.main:app --reload
```

## Status

Day 1 (modeling) and Day 2 (agentic decisioning layer) done. Day 3
(deployment, architecture diagram, demo) next.
