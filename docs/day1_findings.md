# Day 1 Findings: Credit Default Risk Model

## Data cleaning

UCI Default of Credit Card Clients dataset, 30,000 rows, no missing values,
no duplicate rows. Two columns had undocumented category codes not covered by
the official data dictionary:

- `EDUCATION`: documented values are 1 (graduate school), 2 (university), 3
  (high school), 4 (others). The data also contained 0, 5, 6, which were
  folded into 4 (others).
- `MARRIAGE`: documented values are 1 (married), 2 (single), 3 (others). The
  data also contained 0, folded into 3 (others).

`PAY_0..PAY_6` (repayment status, most recent to oldest) also contains
undocumented values -2 and 0 alongside the documented -1 (paid duly) and 1-9
(months delayed). These were kept as-is rather than merged into -1, since -2
appears to represent no balance/no consumption and 0 appears to represent a
balance paid via revolving credit, both of which are meaningfully different
from "paid on time."

Overall default rate: 22.1% (moderate class imbalance, handled with
`class_weight='balanced'` for logistic regression and `scale_pos_weight` for
XGBoost).

## Feature engineering

- **pay_trend**: slope of a line fit through the six repayment status values
  in chronological order. Positive means repayment status is worsening over
  time.
- **avg_utilization**: mean of monthly bill amount divided by credit limit
  across six months, clipped to [0, 3] to contain a small number of
  over-limit accounts.
- **credit_limit_to_income_proxy**: this dataset has no income field. Average
  monthly payment amount stands in as a proxy for repayment capacity. This is
  a real limitation of the dataset, not a substitute for actual income data,
  and any production version of this model would need real income
  information instead.

## Model comparison

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.391 | 0.614 | 0.477 |
| XGBoost | 0.776 | 0.474 | 0.614 | 0.535 |

XGBoost outperforms logistic regression on AUC and precision at the same
recall level (both thresholded at the default 0.5 cutoff with class
balancing applied). The precision gap matters in practice: at the same
recall, XGBoost flags fewer good customers as false positives.

## SHAP findings

Most recent repayment status (`PAY_0`) is by far the strongest predictor of
default, more than double the next most important feature. After that,
average utilization, most recent bill amount, credit limit, and recent
payment amounts drive most of the signal. The engineered `pay_trend` and
`credit_limit_to_income_proxy` features rank lower but still contribute,
in the bottom half of the top 15 features. See `shap_summary.png` and
`shap_global_importance.png` for the full plots, and
`shap_feature_importance.csv` for the ranked list.

Practical read: the model is mostly reacting to how someone is behaving
right now (most recent payment status, current utilization), not primarily
to demographic or static account attributes. That is a reasonable and
explainable basis for a credit decision.

## Fairness check

Checked default rate and false-positive rate across `SEX` and age buckets on
the XGBoost test-set predictions (full tables in `fairness_by_sex.csv` and
`fairness_by_age.csv`).

- **Sex**: actual default rates are close (male 23.4%, female 21.3%), but the
  false-positive rate is noticeably higher for male applicants (22.6%) than
  female applicants (17.2%). That is a real gap worth flagging, not just
  noise, since it means male applicants who will not actually default are
  more likely to be wrongly flagged as risky.
- **Age**: false-positive rate rises somewhat with age (18.9% in the 21-30
  bucket up to 25% in 61+), though the 61+ bucket has very few applicants
  (n=58) so that number is not very reliable on its own.

This is a compliance-relevant finding for a real bank and would need
further investigation (e.g. whether it holds after controlling for other
correlated features) before deployment, not just measured once and ignored.

## Deploy recommendation

XGBoost is the stronger model on raw performance and is the one carried
forward into the agentic decisioning layer (Day 2), with SHAP values used to
generate the explanation for each individual decision rather than treating
it as a black box.

Before this could be deployed in a real setting, it would need:

- A real income field instead of the payment-amount proxy used here.
- Deeper fairness analysis, ideally with a fairness-aware training
  constraint if the male/female false-positive gap holds up under further
  scrutiny.
- Threshold/risk-band tuning against a real cost model (cost of a missed
  default vs. cost of wrongly denying a good applicant), rather than the
  default 0.5 cutoff used for this comparison.

For this project, the risk-band approach (approve / deny / manual review)
planned for the decision agent is a reasonable way to handle the current
precision/recall tradeoff: send borderline-probability applications to human
review instead of forcing a binary decision on cases the model is least sure
about.
