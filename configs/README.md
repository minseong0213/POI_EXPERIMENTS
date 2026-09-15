# Configuration layout

`protocol.yaml` is the shared experimental contract. `runs/` contains small operational
runs, while `stages/` contains the canonical configurations for the 00–15 robustness
plan. A stage without an implemented configuration remains absent rather than being
represented by a misleading placeholder.

`legacy/` contains runnable configurations for older modules whose inputs do not match
the current 32-condition attack study. They are retained for compatibility and must not
be used as evidence that the corresponding current-plan stage ran.

Canonical stage defaults write to `artifacts/poi-robustness-next/`; orchestrated runs may
override that root with `RESULT_DIR`. This prevents an accidental rerun from modifying the
frozen `poi-adversarial-20260915-001` evidence tree.

Configuration names describe purpose and stage. Qualifiers such as `final`, `complete`,
and `enhanced` are prohibited because they do not identify protocol differences. YAML
contains declarative parameters and paths only; credentials and scheduler settings belong
in `experiments/*.env`, and measured results belong in `artifacts/`.

The removed `robustness_complete.yaml`, `robustness_enhanced.yaml`, and
`robustness_final.yaml` duplicated the same analysis with stale artifact paths. Their
single compatibility successor is `legacy/14_robustness_analysis.yaml`, using the newest
input set formerly named `final`. The incompatible `defense_final.yaml` omitted the
`ensemble_models` key required by its runner, so the merged
`legacy/10_filtered_region_classification.yaml` keeps the newer region predictions and the
explicit three-model list required by the code.
