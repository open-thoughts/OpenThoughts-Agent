Build a regression model for the target column `ConcreteCompressiveStrength`.

Inspect `/app/data/train.csv`, which contains `row_id`, features, and the target.
Predict the target for every row of `/app/data/test.csv`, which contains only
`row_id` and features. Feature values may be missing. Do not use row_id as a
feature.

Write `/app/submission.csv` with exactly two columns, in this order:
`row_id,prediction`. Include each test row ID exactly once, without extra rows.
Submit finite numeric predictions. Your predictions are evaluated using RMSE
(lower is better). The verifier reports raw RMSE and a separate validity flag.


Choose your own model and training approach. Python, pandas, and scikit-learn
are installed. Network access is disabled.
