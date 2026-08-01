# Project 2: Credit Default Risk Model + Agentic Decisioning Layer

## Context
Portfolio project for job applications (Data Scientist AND AI/ML Engineer roles).
Pairs with Project 1 (healthcare claims denial prevention) to show range across
domains, mirroring real parallel-workstream experience. Built entirely on public
data. Goal: $0 cost, 2-3 day build, deployed live, open-sourced on GitHub.

## Problem statement
Predict which credit card applicants are likely to default within the review
period, balancing predictive power against explainability and real-world
practicality, the way a bank actually has to. Then wrap that model in an agentic
decisioning layer that explains its decisions using retrieved credit policy
language, not just a raw SHAP number, and routes uncertain cases to human review
instead of guessing.

All data used here is public (UCI Default of Credit Card Clients dataset). No
real or proprietary applicant, account, or assessment data is involved.

## Two layers

### Layer 1 - Data Scientist showcase (modeling)
- EDA and data cleaning (the dataset has known quirks: undocumented category
  codes, inconsistent labels in places, worth cleaning properly and documenting
  the choices made)
- Feature engineering: payment trend over recent months, credit utilization
  ratio, credit-limit-to-income proxy
- Models: logistic regression (explainable baseline) vs XGBoost (higher
  predictive power), compared directly
- Explainability: SHAP values, both global feature importance and individual
  prediction explanations
- Fairness check: default rate and false-positive rate across age and gender
  groups, a real banking compliance concern
- Written recommendation: deploy or not, with reasoning and limitations spelled
  out plainly

### Layer 2 - AI/ML Engineer showcase (agentic layer, built on top of Layer 1)
1. Intake agent - validates application fields, rejects malformed applications
2. Risk-scoring agent - calls the trained model (direct function call, not an LLM)
3. Policy-explanation agent - retrieves relevant credit policy text and explains
   the decision in plain language grounded in that text (Gemini + Chroma)
4. Decision agent - approve / deny / manual review, using risk bands rather than
   a single threshold (same lesson learned from Project 1's model evaluation)

## Data
UCI "Default of Credit Card Clients" dataset, public, about 30,000 rows.
Demographic info, payment history, bill amounts, and default outcome.

## Tech stack (target: $0 cost)
- Agent orchestration: LangGraph
- LLM: Google Gemini free tier (via Google AI Studio, no credit card required)
- Vector store: Chroma (runs locally in-container)
- Risk model: logistic regression + XGBoost (scikit-learn ecosystem)
- Explainability: SHAP
- API: FastAPI
- Hosting: Google Cloud Run (Always Free tier)
- CI: GitHub Actions (free for public repos)

## API contract
- POST /applications/submit - runs the full agent pipeline, returns decision
  plus reasoning trace from each agent
- GET /applications/{id}/status - retrieve a past decision
- GET /health - health check endpoint for Cloud Run

## Metrics to report in README
- Logistic regression vs XGBoost: AUC, precision, recall, comparison table
- SHAP-based top features driving default risk
- Fairness metrics across age/gender groups
- % of applications auto-decided vs routed to human review

## Repo structure
```
notebooks/                 - EDA and modeling notebook(s)
models/                    - training scripts, saved model artifacts
agents/                    - each agent as its own module
data/                      - dataset, policy_docs/
tests/                     - test coverage for models and agents
docs/                      - architecture diagram, design notes
api/                       - FastAPI app
.github/workflows/         - CI (lint, test)
.env.example
requirements.txt
README.md
LICENSE (MIT)
CLAUDE.md                  - this file
```

## README section order
1. Problem statement
2. Architecture diagram
3. Results / metrics (model comparison, SHAP findings, fairness check)
4. Quick start (runnable in under 2 minutes)
5. Tech stack
6. Live demo link
7. Why I built this
8. Limitations / next steps
9. License

## Additional deliverable
A short slide-style summary (methodology, results, deploy recommendation),
separate from the README, since this project also serves as Data Scientist
interview material and should mirror that presentation format.

## 3-day build plan
- Day 1: EDA, cleaning, feature engineering, train logistic regression and
  XGBoost, SHAP analysis, fairness check, write findings.
- Day 2: Build the 4 agents, wire up Chroma and Gemini for the policy-explanation
  agent, wrap in FastAPI, write tests.
- Day 3: Deploy to Cloud Run, write architecture diagram, README, and short
  slide deck, record a demo.

## Setup & run commands
(Fill in / update as the project takes shape.)
```bash
pip install -r requirements.txt --break-system-packages
pytest tests/ -v
uvicorn api.main:app --reload
ruff check .
```

## Working conventions (for Claude Code)
- Make small, incremental, genuine commits as you go, one logical change per
  commit (for example: "add EDA notebook", "add logistic regression baseline",
  "add SHAP analysis", "add intake agent", "add tests for risk-scoring agent").
  Do not squash the whole build into one large commit.
- Write a test alongside each agent/module as it's built, not all at the end.
- Never fabricate or backdate commit timestamps.
- Never use real/proprietary/confidential data, public dataset only.
- Keep the README updated incrementally rather than writing it all at the end.
- Ask before adding new paid dependencies or services outside the $0 stack
  listed above.

## Authorship
- Author: Srinivasa Rao Thata only. Do not add Claude, Anthropic, or any AI tool
  as a co-author, contributor, or in commit messages, README credits, LICENSE,
  or anywhere else in the repo.
- Do not add "Generated with Claude" or similar attribution lines to commits,
  PRs, or files.

## Writing style
- No AI-sounding tone anywhere in the repo (README, docstrings, comments,
  commit messages). Write plainly and directly, the way a person would describe
  their own project.
- No em dashes anywhere, in any file. Use commas, periods, or parentheses
  instead.
- Avoid generic AI phrasing (leverage, seamless, robust solution, cutting-edge,
  dive into, unlock). Keep language concrete and specific.

## Guardrails
- No real/confidential data, ever. This project uses only the public UCI
  dataset and does not reference any proprietary employer or third-party
  assessment data.
- No fake timestamps or backdated commits, the project was built in a focused
  short sprint, and that's a fine, honest story to tell in interviews.
