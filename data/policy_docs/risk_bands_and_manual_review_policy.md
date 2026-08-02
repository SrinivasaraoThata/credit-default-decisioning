# Risk Bands and Manual Review

Decisions are made using risk bands rather than a single pass/fail cutoff on
predicted default probability. A single hard threshold treats a borderline
case identically to a clear-cut one, and produces confident-looking
decisions in exactly the range where the model is least reliable.

- **Low risk**: predicted default probability below the low-risk band
  boundary. Approve at standard terms.
- **Medium risk**: predicted default probability inside the middle band.
  Route to manual review. This band exists specifically to catch borderline
  cases where an automated approve or deny is more likely to be wrong, and
  where a human reviewer can weigh context the model does not have access
  to (verified income, explanation for a payment gap, and so on).
- **High risk**: predicted default probability above the high-risk band
  boundary. Deny, subject to the applicant's right to request manual
  reconsideration.

The manual review band should not be treated as a rounding error to be
minimized. Sending a larger share of borderline cases to manual review, at
the cost of fewer fully-automated decisions, is an acceptable and often
preferable tradeoff when it reduces the chance of an automated wrong call
for cases the model cannot confidently separate.
