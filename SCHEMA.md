# Data fields

`dataset` and `query_id` jointly identify a question. Dataset names are `2wiki`,
`hotpotqa`, and `musique`. The same 1,800 identities occur in every method.

## Queries and common trajectories

`queries.csv` includes `question` and `reference_answers`; the latter is a JSON
array encoded in a CSV field. References are offline evaluation labels.

`shared_executor/trajectories.jsonl.gz` has one record per question, with
`initial_state` (R0) and two `followup_states` (R1/R2). States preserve the
candidate answer, reasoning, retrieved passages, resolved and unresolved
subgoals, and the next retrieval query. Their recorded `f1` uses the 0–1 scale.

`shared_executor/state_scores.csv` has one row per question and depth 0/1/2.
`F1` and `EM` use the 0–100 scale, matching the manuscript. The per-answer F1 is
rounded to two decimals on the 0–1 scale before aggregation. `reader_call_id`
and `retrieval_call_id` link to their respective call files.

Reader-call records contain the model request, original returned text
`response_content`, response model, finish reason, observed latency in
milliseconds, and input/output/total token counts. Pipeline records additionally
preserve the returned message object. Each request contains its recorded
generation parameters and messages. They are successful calls used by the
reported evaluation.

Retrieval-call records contain the query, returned passages, and observed
latency. Passage records preserve corpus IDs, titles, original text, retrieval
scores, and source URLs. Common trajectories accumulate the evidence available
at a state; the retrieval-call files contain the passages returned by individual
requests.

## Allocation

`allocation/action_values.csv` gives the recorded priority for every question's
first and second follow-up for each of the four methods.

`allocation/action_records.csv` contains 1,800 rows per method. `global_action`
is the actual saved allocation order, starting at 1. `from_depth` and `to_depth`
specify the executed transition. `predicted_value` is its recorded priority.
`F1_before` and `F1_after` are the selected question's scores, and `overall_F1`
is the mean over all 1,800 questions after that action.

`allocation/terminal_per_query.csv` records the answer and scores at the
allocated depth. `calls` and `total_tokens` include initialization and the
follow-ups selected for that question. They support Table 3.

## Additional pipelines

Both pipeline directories contain one trajectory per question and separate
Reader/retrieval call files. The trajectory lists its call IDs. RAG-QPP records
its selected context depth `k`, `qpp_score`, and `lambda_q`. Structured Planning
records the generated plan, committed steps, filled templates, and final answer.

`pipelines/action_records.csv` gives `event_index` within each question's
dependency order. It does not impose a global wall-clock order across questions.
`kind` distinguishes Reader calls from retrievals; `stage` identifies the plan,
step extraction, subanswer, or final generation.

`pipelines/terminal_per_query.csv` includes `calls`, `retrievals`, separate
input/output token counts, and their total, alongside F1 and EM.

## Paper and Oracle files

`paper/table2.csv` contains full-precision `AUBPC`, `F1_10`, `F1_25`, and `F1_50`.
`paper/table3.csv` contains dataset F1, overall F1/EM, average calls/retrievals
per question, and total tokens before conversion to millions. The paper rounds
these values to three decimals.

`paper/figure2_curves.csv` gives all 1,801 budget points for the four allocation
methods. `budget_ratio = actions / 3600`. AUBPC is trapezoidal curve area divided
by the evaluated budget width of 0.5, so it has the same 0–100 scale as F1.

`offline_oracle/curve.csv` contains the exact optimum at every budget and the
corresponding GARA curve. Each `allocation_B*.csv` reports the separately
optimized depth and F1 for all questions at that checkpoint.

`paper/primary_auc_intervals.csv` preserves the original saved primary AUBPC
contrasts and intervals. Its adjusted level is 98.3333% for the three
prespecified comparisons.
