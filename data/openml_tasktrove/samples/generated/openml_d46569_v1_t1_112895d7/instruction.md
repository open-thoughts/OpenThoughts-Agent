Build a classification model for the target column `rating`.

Inspect `/app/data/train.csv`, which contains `row_id`, features, and the target.
Predict the target for every row of `/app/data/test.csv`, which contains only
`row_id` and features. Feature values may be missing. Do not use row_id as a
feature.

Write `/app/submission.csv` with exactly two columns, in this order:
`row_id,prediction`. Include each test row ID exactly once, without extra rows.
Targets are integer class codes from 0 through 9.
Submit integer class predictions. Class meanings, in code order: ["A", "AA", "AAA", "B", "BB", "BBB", "C", "CC", "CCC", "D"].
Your predictions are evaluated using accuracy (higher is better).


Choose your own model and training approach. Python, pandas, and scikit-learn
are installed. Network access is disabled.
