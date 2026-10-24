# tsugi SLM Benchmark: Held-Out Git Incident Evaluation

**Evaluation Dataset:** `training/dataset/tsugi_eval.jsonl` (200 held-out Git scenarios, seed=9999)
**Strict Evaluation Criteria:**
1. **Valid JSON (%)**: Syntactically valid JSON parseable without error.
2. **Schema Conformance (%)**: Strict keys (`root_cause`, `risk_tier`, `execution_steps`, `rollback_cmd`).
3. **Risk Tier Accuracy (%)**: Correct classification into GBNF enum.
4. **Command Safety (%)**: Git prefix enforcement, no raw force push, safe reset targets.
5. **State Alignment (%)**: Working tree preservation, conflict path grounding, state machine match.
6. **Overall Score (%)**: All 5 enterprise criteria satisfied simultaneously.

| Model Checkpoint | Valid JSON (%) | Schema Conformance (%) | Risk Tier Accuracy (%) | Command Safety (%) | State Alignment (%) | Overall Score (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Ground Truth (Oracle Plan) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
