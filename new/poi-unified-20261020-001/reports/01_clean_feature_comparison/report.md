# Unified clean three-feature comparison

**Status: GENERATED_PENDING_INDEPENDENT_REVIEW.** This report is exploratory validation evidence from a previously observed development partition. It is not pristine confirmatory or locked-test evidence.

## Design and thresholds

The comparison uses full five features, coordinates only, and the three categorical features across eight models, 17 one-vs-rest regions, and seeds 42/202/340. Every threshold is selected on the same 1,693 clean calibration POIs. Calibrated and fixed-0.5 results are both scored on the same 1,707 evaluation POIs; fixed-0.5 full-validation reproduction is reported separately.

Tree models use balanced one-vs-rest sample weights. TabPFN fits are unweighted, so fixed-0.5 F1 differences are not interpreted as pure feature-information effects. Prediction coverage is 100% for every required stratum.

## Full-feature calibrated point estimates

| Model | Mean F1 over three seed macros | Sample SD across three seed macros |
|---|---:|---:|
| tabpfn_v3 | 0.993520409 | 0.000169891 |
| xgboost | 0.993469063 | 0.000745874 |
| lightgbm | 0.992880906 | 0.000488548 |
| tabpfn_v2_5 | 0.992397746 | 0.000665328 |
| catboost | 0.991682486 | 0.000294613 |
| tabpfn_v2_6 | 0.991140696 | 0.001927374 |
| decision_tree | 0.991047076 | 0.000167299 |
| random_forest | 0.982727410 | 0.000815796 |

The user-specified highest point-estimate rule retains **tabpfn_v3** as the common downstream classifier (mean F1 0.993520409). The runner-up is xgboost and the point difference is 0.000051346; its paired conditional 95% interval for leader minus runner-up is [-0.002980558, 0.003233395]. This small difference is not described as statistical superiority.

## Uncertainty scope

The 2,000 paired bootstrap draws resample original evaluation POI IDs with shared multiplicities across all models, feature sets, regions, and seeds. Each draw averages the same 17 regions and three fixed model seeds before differences are formed. The percentile intervals are marginal and conditional on the 51 fitted strata and calibration-selected thresholds. They exclude training-algorithm, calibration-selection, spatial-dependence, model-selection, and locked-test uncertainty and are not simultaneous intervals or null-calibrated tests.

## Provenance and limits

Historical v2.5/v3 feature-ablation predictions come from one authenticated archived run but lack per-fit checkpoint attestations (PARTIAL). Prior whole-dataset coordinate access remains a historical isolation limitation (PARTIAL). New full-v3 persisted states are the attack/defense replay baseline and are compared diagnostically with historical full-v3 predictions; they do not silently replace the historical feature-table source.

Generated cardinalities: {"interval_rows": 144, "macro_by_seed_rows": 216, "macro_seed_summary_rows": 72, "metric_rows": 3672, "pair_rows": 1656, "replicate_rows": 288000, "same_model_feature_pair_rows": 144, "threshold_rows": 1224}.

No attack, defense, locked-test, or unseen-test performance claim is made here.
