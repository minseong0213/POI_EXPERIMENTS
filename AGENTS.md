# POI experiment agent rules

These rules apply to the entire repository.

## Agent roles and models

- Use the project-scoped `poi_experiment_runner` custom agent for experiment implementation and
  execution. Its model, reasoning effort, and core instructions are fixed in
  `.codex/agents/poi-experiment-runner.toml`.
- Use the project-scoped `independent_experiment_audit` custom agent for independent review. Its
  model, reasoning effort, and core instructions are fixed in
  `.codex/agents/independent-experiment-audit.toml`.
- Do not override either role's model or reasoning effort at spawn time unless the user explicitly
  changes the role configuration.
- Start both roles as separate agents. Do not let the producing agent act as the sole reviewer.
- Start every independent review with `fork_turns="none"`. Its initial task must contain the
  original requirements and artifact paths, but not the producing agent's conclusions.
- Every persisted review must record the requested model, reasoning effort, role, and review scope.

## Mandatory independent review

An agent must not describe an experiment, stage, delivery bundle, or goal as complete until a
fresh-context review agent has independently checked it against the original experiment plan.

1. Give the reviewer the original requirement documents and artifact paths. Do not give it the
   producing agent's conclusion or ask it to confirm a claimed result.
2. The reviewer must not modify experiment code or evidence. It may write only inside a dedicated
   `reports/99_independent_review/` directory when explicitly asked to persist its verdict.
3. Existing `_SUCCESS.json`, `report.md`, audit output, and delivery manifests are evidence, not
   proof. The reviewer must inspect source tables, metadata, plots, and the code that generated them.
4. The review must report every requirement as `PASS`, `PARTIAL`, `FAIL`, or `NOT_RUN`, with exact
   paths, observed cardinalities, and limitations.
5. Required checks include data splits and leakage features; model, region, seed, condition, and
   metric cardinality; attack constraints and provenance; test isolation; PNG/SVG rules; and the
   requested regional and clean/adversarial coverage of SHAP and LIME.
6. A `PARTIAL`, `FAIL`, or `NOT_RUN` item must remain visible in status reports and delivery notes.
   The producing agent must not mark the containing scope complete unless the original plan
   explicitly allows that item to remain pending.
7. After a fix or rerun, use a new fresh-context reviewer. The producing agent must not review its
   own correction as the sole verifier.

Follow [the independent review protocol](docs/experiments/independent_review_protocol.md) for the
required review table and completion gate.
