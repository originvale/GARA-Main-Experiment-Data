# GARA Main Experiment Data

Evaluation records for Tables 2 and 3 of **State-Aware Dynamic Retrieval Allocation
for Resource-Constrained Multi-Hop Question Answering**.

The release covers 1,800 evaluation questions: 600 each from 2WikiMultiHopQA,
HotpotQA, and MuSiQue. It contains retrieval trajectories, Reader requests and
outputs, per-question scores, allocation records, and the reported summary data.

## Contents

| Directory or file | Contents |
| --- | --- |
| `queries.csv` | Benchmark IDs, questions, and offline reference answers |
| `shared_executor/` | The common R0/R1/R2 trajectories, 5,400 Reader calls, 5,400 retrieval records, and state scores |
| `allocation/` | Recorded action values, the four 1,800-action allocation sequences, and terminal per-question results |
| `pipelines/rag_qpp/` | RAG-QPP retrievals, Reader outputs, and completed trajectories |
| `pipelines/structured_planning/` | Plans, retrievals, evidence extraction, subanswers, and final outputs |
| `pipelines/terminal_per_query.csv` | Per-question quality and resource counts for both additional pipelines |
| `pipelines/action_records.csv` | Within-question dependency order for pipeline operations |
| `offline_oracle/` | The budgetwise optimum curve and allocations at the three reported checkpoints |
| `paper/` | Full-precision values for Tables 2 and 3, Figure 2, and the primary AUBPC intervals |

Table 2 compares Adaptive-RAG, MBA-RAG, Current-quality, and GARA under a common
follow-up quota of 1,800 actions. The initial R0 state is outside that quota.
Each question has two possible follow-ups, so 10%, 25%, and 50% capacity correspond
to 360, 900, and 1,800 actions. Table 3 also includes RAG-QPP and Structured
Planning with their recorded resource use.

The offline Oracle curve optimizes each budget separately under the prefix
constraint. Its checkpoint allocations are separate solutions; the curve is an
upper envelope rather than a single nested allocation sequence.

## Read and verify

All CSVs use UTF-8. The larger records use gzip-compressed JSON Lines. Each
decompressed line is a JSON object, readable with Python's `gzip` and `json`.

```bash
python scripts/verify_data.py
```

The verifier uses the Python standard library. It checks file integrity, query
coverage, state/action links, recorded priorities, quality curves, checkpoint
metrics, terminal scores, and Reader-call/token totals. It reads the saved
scores and outputs without training a model or calling an API. The release
preparation also checked every released answer against the study's evaluator.
Field definitions are in [SCHEMA.md](SCHEMA.md).

## Sources and citation

- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [HotpotQA and its Wikipedia corpus](https://hotpotqa.github.io/)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)
- [GARA source code](https://github.com/originvale/GARA)

Please cite the GARA paper and the source benchmarks when using these records.
The licensing and attribution of benchmark material and retrieved passages are
described in [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
