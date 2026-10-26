"""Explain the clean-validation winner on clean and representative attack data."""
import argparse
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.multitest import multipletests
import yaml

from .attack_evaluate import EXPECTED_MODELS, _attack_frame, _load_clean
from .data import CATEGORICAL, FEATURES, NUMERIC


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _matrix(frame, features=FEATURES, dtype=np.float32):
    """Numeric matrix for TabPFN, or an explicitly requested explanation dtype."""
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError("Explanation input contains non-finite values")
    return values


def _dataframe_sha256(frame, features=FEATURES):
    """Hash exact ordered values, feature names, and pandas dtypes."""
    selected = frame.loc[:, list(features)]
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "columns": list(features),
        "dtypes": [str(selected[name].dtype) for name in features],
        "rows": len(selected),
    }, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\n")
    digest.update(selected.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()


def _fitted_object_sha256(value):
    """Bind the actual fitted Python object bytes used by this process."""
    buffer = io.BytesIO()
    pickle.dump(value, buffer, protocol=5)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _transformed_matrix_attestation(values):
    """Hash dense or CSR transformed training values including structural arrays."""
    from scipy import sparse
    digest = hashlib.sha256()
    if sparse.issparse(values):
        matrix = values.tocsr(copy=False)
        header = {
            "format": "csr", "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "data_dtype": str(matrix.data.dtype), "indices_dtype": str(matrix.indices.dtype),
            "indptr_dtype": str(matrix.indptr.dtype),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        for array in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        matrix = np.ascontiguousarray(values)
        header = {"format": "dense", "shape": list(matrix.shape), "dtype": str(matrix.dtype)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(matrix.tobytes())
    return {**header, "sha256": digest.hexdigest()}


def _category_contract(train, features):
    contract = {}
    for feature in features:
        if feature not in CATEGORICAL:
            continue
        vocabulary = list(dict.fromkeys(train[feature].astype(str).tolist()))
        numeric = {}
        for original in vocabulary:
            value = float(original)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite training category in {feature}")
            numeric.setdefault(value, []).append(original)
        contract[feature] = {"vocabulary": vocabulary, "numeric_lookup": numeric}
    return contract


def _tree_prediction_frame(values, features, category_contract):
    """Preserve raw frames; map numeric explanation categories to exact train strings."""
    from_frame = isinstance(values, pd.DataFrame)
    if from_frame:
        missing = set(features) - set(values.columns)
        if missing:
            raise ValueError(f"Tree prediction frame misses features: {sorted(missing)}")
        frame = values.loc[:, list(features)].copy()
    else:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(features):
            raise ValueError("Tree prediction array has the wrong feature shape")
        if array.dtype == np.float32:
            raise ValueError("Tree prediction arrays must preserve float64 numeric precision")
        frame = pd.DataFrame(np.asarray(array, dtype=np.float64), columns=features)
    for feature in NUMERIC:
        if feature in frame:
            numeric = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError(f"Non-finite numeric tree input: {feature}")
            if not from_frame:
                frame[feature] = numeric
    for feature, contract in category_contract.items():
        exact = set(contract["vocabulary"])
        lookup = contract["numeric_lookup"]
        normalized = []
        for raw in frame[feature].tolist():
            if isinstance(raw, str) and raw in exact:
                normalized.append(raw)
                continue
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unknown category for {feature}: {raw!r}") from error
            if not np.isfinite(numeric):
                raise ValueError(f"Non-finite category for {feature}")
            candidates = lookup.get(numeric, [])
            if len(candidates) != 1:
                reason = "ambiguous" if candidates else "unknown"
                raise ValueError(
                    f"{reason} numeric category for {feature}: {raw!r}; candidates={candidates}"
                )
            normalized.append(candidates[0])
        frame[feature] = normalized
    return frame


def _positions(frame, region, per_class, seed):
    positive = frame.index[frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    negative = frame.index[~frame.region.eq(region)].to_series().sample(
        n=per_class, random_state=seed).to_numpy()
    return np.concatenate([positive, negative])


def _beeswarm(records, figures, name, title, features=FEATURES):
    order = records.POI_ID.drop_duplicates().tolist()
    values = records.pivot(index="POI_ID", columns="feature", values="feature_value").loc[order, features]
    shap_values = records.pivot(index="POI_ID", columns="feature", values="shap_value").loc[order, features]
    base = records.groupby("POI_ID").base_value.first().loc[order].to_numpy()
    shap.summary_plot(shap_values.to_numpy(), values.to_numpy(), feature_names=features,
                      plot_type="dot", color_bar=True, color_bar_label="Feature value",
                      show=False, max_display=len(features))
    figure = plt.gcf(); axis = plt.gca()
    axis.axvline(0, color="#444444", linewidth=.8, zorder=0)
    axis.set_xlabel("SHAP value"); figure.suptitle(title)
    _save(figure, figures, name)
    return values.to_numpy(), shap_values.to_numpy(), base


def _odds_ratios(train, regions):
    """Use the canonical separation-aware logistic reference, not tree/TabPFN odds."""
    from .odds_ratio import design_matrix, fit_associations
    design, _ = design_matrix(train, train)
    rows, failures = [], []
    for region in regions:
        result, notes = fit_associations(design, train.region.eq(region).astype(int).to_numpy())
        rows.extend(dict(region=region, **row) for row in result)
        unavailable = [row for row in result if not row['identifiable']]
        if unavailable or notes:
            failures.append(dict(region=region, unavailable_coefficients=len(unavailable),
                                 reasons=sorted({row['failure_reason'] for row in unavailable}),
                                 warnings=notes))
    result = pd.DataFrame(rows)
    result['q_value'] = np.nan
    finite = np.isfinite(result.p_value.to_numpy(float))
    if finite.any():
        result.loc[finite, 'q_value'] = multipletests(result.loc[finite, 'p_value'], method='fdr_bh')[1]
    return result, pd.DataFrame(failures, columns=['region', 'unavailable_coefficients', 'reasons', 'warnings'])


class ModelFactory:
    def __init__(self, model_id, cfg, train):
        self.model_id, self.cfg, self.train = model_id, cfg, train
        self.tabpfn_runtime = {}
        self.last_fit_attestation = {}
        self._checkpoint_sha256 = None
        self._last_fitted_model = None
        self._last_probe = None

    def fit(self, features, truth):
        if self.model_id.startswith("tabpfn_"):
            configured_checkpoint = self.cfg.get("official_checkpoint")
            checkpoint_path = (
                Path(str(configured_checkpoint)).expanduser()
                if configured_checkpoint else None
            )
            expected_checkpoint_sha = str(
                self.cfg.get("expected_checkpoint_sha256", "")
            ).lower()
            require_attestation = bool(self.cfg.get("require_checkpoint_attestation", False))
            if require_attestation and (
                checkpoint_path is None or not checkpoint_path.is_file()
                or len(expected_checkpoint_sha) != 64
            ):
                raise RuntimeError(
                    "TabPFN replay requires an explicit official_checkpoint and SHA-256"
                )
            if (
                checkpoint_path is not None and checkpoint_path.is_file()
                and self._checkpoint_sha256 is None
            ):
                self._checkpoint_sha256 = _file_sha256(checkpoint_path)
            checkpoint_sha = self._checkpoint_sha256
            if expected_checkpoint_sha and checkpoint_sha != expected_checkpoint_sha:
                raise RuntimeError("Resolved TabPFN checkpoint SHA-256 differs from config")
            if checkpoint_path is None and not os.environ.get("TABPFN_TOKEN"):
                token_path = Path(os.environ.get("TABPFN_TOKEN_FILE", "/root/.config/poi/tabpfn_token"))
                if token_path.is_file():
                    os.environ["TABPFN_TOKEN"] = token_path.read_text().strip()
            if checkpoint_path is None and not os.environ.get("TABPFN_TOKEN"):
                raise RuntimeError("TABPFN_TOKEN or TABPFN_TOKEN_FILE is required")
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            import tabpfn
            import torch
            versions = {"tabpfn_v2_5": ModelVersion.V2_5, "tabpfn_v2_6": ModelVersion.V2_6,
                        "tabpfn_v3": ModelVersion.V3}
            categorical = [index for index, feature in enumerate(features) if feature in CATEGORICAL]
            model = TabPFNClassifier.create_default_for_version(
                versions[self.model_id], device=self.cfg["device"],
                n_estimators=self.cfg["n_estimators"], random_state=self.cfg["seed"],
                # TabPFN 8.1 accepts Path here but save_fit_state JSON-serializes
                # constructor parameters.  Preserve the same path while making
                # the official fitted-state archive serializable.
                model_path=str(checkpoint_path) if checkpoint_path is not None else "auto",
                categorical_features_indices=categorical, show_progress_bar=False,
                memory_saving_mode="auto")
            train_matrix = _matrix(self.train, features)
            model.fit(train_matrix, truth)
            probe_indices = np.unique(np.linspace(
                0, len(train_matrix) - 1, min(64, len(train_matrix)), dtype=int
            ))
            probe = np.ascontiguousarray(train_matrix[probe_indices], dtype=np.float32)
            probe_prediction = np.ascontiguousarray(
                model.predict_proba(probe)[:, 1], dtype=np.float64
            )
            probe_repeat_prediction = np.ascontiguousarray(
                model.predict_proba(probe)[:, 1], dtype=np.float64
            )
            if not np.array_equal(probe_prediction, probe_repeat_prediction):
                raise RuntimeError("TabPFN same-input replay prediction is not deterministic")
            matrix = np.ascontiguousarray(train_matrix, dtype=np.float32)
            labels = np.ascontiguousarray(np.asarray(truth), dtype=np.int8)
            self.last_fit_attestation = {
                "format": "poi-tabpfn25-fit-replay-v1",
                "model_id": self.model_id,
                "seed": int(self.cfg["seed"]),
                "n_estimators": int(self.cfg["n_estimators"]),
                "feature_order": list(features),
                "categorical_feature_indices": categorical,
                "train_rows": len(matrix),
                "train_matrix_dtype": str(matrix.dtype),
                "train_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "truth_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                "official_checkpoint_path": (
                    str(checkpoint_path.resolve()) if checkpoint_path is not None else None
                ),
                "official_checkpoint_sha256": checkpoint_sha,
                "fit_mode": "fit_preprocessors",
                "memory_saving_mode": "auto",
                "probe_indices": probe_indices.tolist(),
                "probe_matrix_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
                "probe_prediction_dtype": str(probe_prediction.dtype),
                "probe_prediction_sha256": hashlib.sha256(
                    probe_prediction.tobytes()
                ).hexdigest(),
                "probe_repeat_prediction_sha256": hashlib.sha256(
                    probe_repeat_prediction.tobytes()
                ).hexdigest(),
                "tabpfn_version": tabpfn.__version__,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "requested_device": str(self.cfg["device"]),
                "preprocessor_state_binding": (
                    "official TabPFN persisted fit-state plus post-load probe prediction hash"
                ),
            }
            self._last_fitted_model = model
            self._last_probe = probe
            self.tabpfn_runtime = {"tabpfn": tabpfn.__version__, "torch": torch.__version__,
                                   "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   "official_checkpoint_path": (
                                       str(checkpoint_path.resolve())
                                       if checkpoint_path is not None else None
                                   ),
                                   "official_checkpoint_sha256": checkpoint_sha}
            return lambda values: model.predict_proba(np.asarray(values, dtype=np.float32))[:, 1]
        from .ablation import transformer
        from .benchmark import estimator
        params = yaml.safe_load(Path(self.cfg["tree_config"]).read_text())["models"][self.model_id]
        category_contract = _category_contract(self.train, features)
        transform = transformer(features)
        train_x = transform.fit_transform(self.train)
        positive, negative = truth.sum(), len(truth) - truth.sum()
        weights = np.where(truth == 1, len(truth) / (2 * positive), len(truth) / (2 * negative))
        model = estimator(self.model_id, params, self.cfg["seed"])
        model.fit(train_x, truth, sample_weight=weights)
        def predict(values):
            frame = _tree_prediction_frame(values, features, category_contract)
            return model.predict_proba(transform.transform(frame))[:, 1]
        labels = np.ascontiguousarray(np.asarray(truth), dtype=np.int8)
        probe_indices = np.unique(np.linspace(
            0, len(self.train) - 1, min(64, len(self.train)), dtype=int
        ))
        probe_frame = self.train.iloc[probe_indices].loc[:, list(features)].copy()
        probe_prediction = np.ascontiguousarray(predict(probe_frame), dtype=np.float64)
        probe_repeat = np.ascontiguousarray(predict(probe_frame), dtype=np.float64)
        probe_difference = np.abs(probe_prediction - probe_repeat)
        probe_tolerance = float(self.cfg.get("repeat_prediction_absolute_tolerance", 1e-12))
        if (
            probe_tolerance < 0 or not np.isfinite(probe_tolerance)
            or not np.allclose(probe_prediction, probe_repeat, atol=probe_tolerance, rtol=0)
        ):
            raise RuntimeError("Tree same-frame probe prediction exceeds repeat tolerance")
        transformed = _transformed_matrix_attestation(train_x)
        output_features = list(map(str, transform.get_feature_names_out()))
        self.last_fit_attestation = {
            "format": "poi-tree-fit-attestation-v1",
            "model_id": self.model_id,
            "seed": int(self.cfg["seed"]),
            "feature_order": list(features),
            "categorical_features": [name for name in features if name in CATEGORICAL],
            "train_rows": len(self.train),
            "raw_train_frame_sha256": _dataframe_sha256(self.train, features),
            "raw_train_dtypes": {
                feature: str(self.train[feature].dtype) for feature in features
            },
            "category_vocabulary": {
                feature: value["vocabulary"] for feature, value in category_contract.items()
            },
            "category_numeric_mapping_policy": (
                "exact_train_string_or_unique_numeric_equivalent; ambiguous_and_unknown_fail"
            ),
            "truth_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "sample_weight_sha256": hashlib.sha256(
                np.ascontiguousarray(weights, dtype=np.float64).tobytes()
            ).hexdigest(),
            "transformed_train": transformed,
            "transformed_feature_count": len(output_features),
            "transformed_feature_names_sha256": hashlib.sha256(
                ("\n".join(output_features) + "\n").encode()
            ).hexdigest(),
            "transformer_class": f"{type(transform).__module__}.{type(transform).__qualname__}",
            "transformer_fitted_state_sha256": _fitted_object_sha256(transform),
            "estimator_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "estimator_params_sha256": hashlib.sha256(
                json.dumps(params, sort_keys=True, separators=(",", ":"), default=str).encode()
            ).hexdigest(),
            "estimator_fitted_state_sha256": _fitted_object_sha256(model),
            "prediction_input_contract": (
                "raw_dataframe_for_benchmark; float64_array_for_SHAP_LIME; "
                "exact_train_category_vocabulary"
            ),
            "probe_indices": probe_indices.tolist(),
            "probe_raw_frame_sha256": _dataframe_sha256(probe_frame, features),
            "probe_prediction_dtype": str(probe_prediction.dtype),
            "probe_prediction_sha256": hashlib.sha256(probe_prediction.tobytes()).hexdigest(),
            "probe_repeat_prediction_sha256": hashlib.sha256(probe_repeat.tobytes()).hexdigest(),
            "probe_predictions_exactly_equal": bool(np.array_equal(probe_prediction, probe_repeat)),
            "probe_repeat_max_absolute_difference": float(probe_difference.max(initial=0.0)),
            "probe_repeat_absolute_tolerance": probe_tolerance,
            "probe_predictions_within_tolerance": True,
        }
        return predict

    def persist_last_fit_state(self, path):
        """Persist and reload the official lightweight TabPFN fitted-state archive."""
        if self._last_fitted_model is None or self._last_probe is None:
            raise RuntimeError("No fitted TabPFN state is available to persist")
        path = Path(path)
        if path.suffix != ".tabpfn_fit":
            raise ValueError("Persisted TabPFN fit-state path must end with .tabpfn_fit")
        model_class = type(self._last_fitted_model)
        if not hasattr(self._last_fitted_model, "save_fit_state") or not hasattr(
            model_class, "load_from_fit_state"
        ):
            raise RuntimeError("Installed TabPFN lacks official fitted-state persistence APIs")
        self._last_fitted_model.save_fit_state(path)
        if not path.is_file():
            raise RuntimeError("TabPFN fitted-state persistence did not create an archive")
        restored = model_class.load_from_fit_state(path, device=self.cfg["device"])
        restored_prediction = np.ascontiguousarray(
            restored.predict_proba(self._last_probe)[:, 1], dtype=np.float64
        )
        restored_sha = hashlib.sha256(restored_prediction.tobytes()).hexdigest()
        if restored_sha != self.last_fit_attestation.get("probe_prediction_sha256"):
            raise RuntimeError("Reloaded TabPFN fitted state changes fixed probe predictions")
        self.last_fit_attestation.update({
            "fitted_state_format": "tabpfn_official_fit_state_v1",
            "fitted_state_api": (
                "TabPFNClassifier.save_fit_state/load_from_fit_state"
            ),
            "fitted_state_path": str(path.resolve()),
            "fitted_state_sha256": _file_sha256(path),
            "fitted_state_size_bytes": path.stat().st_size,
            "restored_probe_prediction_sha256": restored_sha,
            "restored_on_device": str(self.cfg["device"]),
        })
        return dict(self.last_fit_attestation)


def _explain(predict, background, values, features, seed):
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation",
                              feature_names=features, seed=seed)
    return explainer(values, max_evals=2 * len(features) + 1, batch_size="auto", silent=True)


def _lime_arrays(train_values, features=FEATURES):
    encoded = train_values.copy()
    categorical_names, reverse = {}, {}
    for index, feature in enumerate(features):
        if feature not in CATEGORICAL:
            continue
        observed = np.unique(train_values[:, index])
        mapping = {value: ordinal for ordinal, value in enumerate(observed)}
        encoded[:, index] = np.array([mapping[value] for value in train_values[:, index]])
        categorical_names[index] = [str(value) for value in observed]
        reverse[index] = observed
    return encoded, categorical_names, reverse


def run(config_path, output_override=None):
    started = time.time(); cfg = yaml.safe_load(Path(config_path).read_text())
    output = Path(output_override or cfg["output"])
    if (output / "_SUCCESS.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed {output}")
    ranking_path = os.environ.get("SELECTION_RANKING", cfg["selection_ranking"])
    ranking = pd.read_csv(ranking_path)
    if set(ranking.model) != EXPECTED_MODELS or ranking.model.duplicated().any():
        raise ValueError("Complete eight-model clean ranking is required")
    model_id = ranking.sort_values("mean", ascending=False).iloc[0].model
    explanation_dtype = np.float32 if model_id.startswith("tabpfn_") else np.float64
    clean, manifest = _load_clean(os.environ.get("DATA_DIR", cfg["data_dir"]))
    train = clean.loc[clean.split.eq("train")].reset_index(drop=True)
    validation = clean.loc[clean.split.eq("validation")].reset_index(drop=True)
    attacks = pd.read_parquet(os.environ.get("ATTACKS_FILE", cfg["attacks_file"]))
    attack_rows = attacks.loc[attacks.split.eq("validation") &
                              attacks.attack_condition.eq(cfg["representative_attack"])]
    attack = _attack_frame(attack_rows, validation.POI_ID.tolist())
    evaluations = {"clean": validation, "adversarial": attack}
    regions = sorted(manifest.region.unique())
    tables, figures, html, shards = (output / "tables", output / "figures", output / "html",
                                     output / "work" / "shap")
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(exist_ok=True)
    html.mkdir(exist_ok=True); shards.mkdir(parents=True, exist_ok=True)
    train_values = _matrix(train, dtype=explanation_dtype)
    background_positions = np.linspace(0, len(train) - 1, min(cfg["background_rows"], len(train)), dtype=int)
    background = train_values[background_positions]
    factory = ModelFactory(model_id, cfg, train)
    detailed = []
    for region in regions:
        truth = train.region.eq(region).astype(int).to_numpy()
        predict = factory.fit(FEATURES, truth)
        positions = _positions(validation, region, cfg["explain_per_class"], cfg["seed"])
        for condition, frame in evaluations.items():
            shard = shards / f"{region}-{condition}.csv"
            if shard.exists():
                part = pd.read_csv(shard)
                if len(part) != len(positions) * len(FEATURES):
                    raise RuntimeError(f"Incomplete SHAP shard {shard}")
                detailed.append(part); continue
            values = _matrix(frame, dtype=explanation_dtype)[positions]
            explanation = _explain(predict, background, values, FEATURES, cfg["seed"])
            rows = []
            for row_index, position in enumerate(positions):
                for feature_index, feature in enumerate(FEATURES):
                    rows.append({"model": model_id, "region": region, "condition": condition,
                                 "POI_ID": frame.POI_ID.iloc[position],
                                 "true_region": frame.region.iloc[position], "feature": feature,
                                 "feature_value": values[row_index, feature_index],
                                 "shap_value": explanation.values[row_index, feature_index],
                                 "base_value": explanation.base_values[row_index]})
            part = pd.DataFrame(rows); temporary = shard.with_suffix(".csv.tmp")
            part.to_csv(temporary, index=False); os.replace(temporary, shard)
            detailed.append(part)
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    detailed = pd.concat(detailed, ignore_index=True)
    expected = len(regions) * 2 * cfg["explain_per_class"] * 2 * len(FEATURES)
    if len(detailed) != expected or set(detailed.model) != {model_id}:
        raise RuntimeError("SHAP result cardinality is incomplete")
    detailed.to_csv(tables / "shap_values.csv", index=False)
    grouped = (detailed.assign(abs_shap=detailed.shap_value.abs())
               .groupby(["region", "condition", "feature"], as_index=False)
               .abs_shap.mean().rename(columns={"abs_shap": "mean_abs_shap"}))
    grouped.to_csv(tables / "shap_grouped_importance.csv", index=False)
    summary = grouped.groupby(["condition", "feature"], as_index=False).mean_abs_shap.mean()
    summary.to_csv(tables / "shap_global_summary.csv", index=False)
    representative_arrays = {}
    for region in regions:
        for condition in evaluations:
            subset = detailed.loc[detailed.region.eq(region) & detailed.condition.eq(condition)]
            arrays = _beeswarm(subset, figures, f"shap_beeswarm_{region.lower()}_{condition}",
                               f"{model_id} SHAP — {region} vs others — {condition}")
            if region == cfg["representative_region"]:
                representative_arrays[condition] = arrays
    values, shap_values, base = representative_arrays["clean"]
    for feature in NUMERIC:
        shap.dependence_plot(FEATURES.index(feature), shap_values, values,
                             feature_names=FEATURES, show=False, interaction_index=None)
        _save(plt.gcf(), figures, f"shap_dependence_{feature.lower()}")
    explanation = shap.Explanation(values=shap_values[0], base_values=float(base[0]),
                                   data=values[0], feature_names=FEATURES)
    shap.plots.waterfall(explanation, max_display=len(FEATURES), show=False)
    _save(plt.gcf(), figures, "shap_waterfall_representative")
    shap.save_html(str(html / "shap_force_representative.html"),
                   shap.force_plot(float(base[0]), shap_values[0], values[0], feature_names=FEATURES))

    representative_truth = train.region.eq(cfg["representative_region"]).astype(int).to_numpy()
    representative_predict = factory.fit(FEATURES, representative_truth)
    encoded_train, categorical_names, reverse = _lime_arrays(train_values)
    sample = encoded_train[np.linspace(0, len(encoded_train) - 1,
                                       min(cfg["lime_background_rows"], len(encoded_train)), dtype=int)]
    encoded_value = values[0].copy()
    for index, observed in reverse.items():
        encoded_value[index] = int(np.where(observed == encoded_value[index])[0][0])
    def lime_predict(matrix):
        decoded = np.asarray(matrix, dtype=explanation_dtype).copy()
        for index, observed in reverse.items():
            ordinal = np.rint(decoded[:, index]).astype(int).clip(0, len(observed) - 1)
            decoded[:, index] = observed[ordinal]
        probability = representative_predict(decoded)
        return np.column_stack([1 - probability, probability])
    lime = LimeTabularExplainer(sample, feature_names=FEATURES,
        class_names=["other_region", cfg["representative_region"]],
        categorical_features=list(reverse), categorical_names=categorical_names,
        mode="classification", random_state=cfg["seed"], discretize_continuous=True)
    lime_result = lime.explain_instance(encoded_value, lime_predict, num_features=len(FEATURES),
                                        num_samples=cfg["lime_samples"])
    lime_result.save_to_file(str(html / "lime_representative.html"))
    pd.DataFrame(lime_result.as_list(label=1), columns=["rule", "weight"]).to_csv(
        tables / "lime_representative.csv", index=False)
    _save(lime_result.as_pyplot_figure(label=1), figures, "lime_representative")

    top_feature = summary.groupby("feature").mean_abs_shap.mean().sort_values(ascending=False).index[0]
    reduced = [feature for feature in FEATURES if feature != top_feature]
    reduced_predict = factory.fit(reduced, representative_truth)
    reduced_train = _matrix(train, reduced, dtype=explanation_dtype)
    reduced_values = _matrix(validation, reduced, dtype=explanation_dtype)[
        _positions(validation, cfg["representative_region"], cfg["explain_per_class"], cfg["seed"])]
    reduced_explanation = _explain(reduced_predict, reduced_train[background_positions], reduced_values,
                                   reduced, cfg["seed"])
    pd.DataFrame({"feature": reduced,
                  "mean_abs_shap": np.abs(reduced_explanation.values).mean(axis=0),
                  "removed_feature": top_feature}).sort_values("mean_abs_shap", ascending=False).to_csv(
        tables / "shap_top_feature_removed.csv", index=False)
    shap.summary_plot(reduced_explanation.values, reduced_values, feature_names=reduced,
                      plot_type="dot", color_bar=True, color_bar_label="Feature value",
                      show=False, max_display=len(reduced))
    plt.gca().axvline(0, color="#444444", linewidth=.8, zorder=0)
    _save(plt.gcf(), figures, "shap_summary_top_feature_removed")

    odds, failures = _odds_ratios(train, regions)
    odds.to_csv(tables / "odds_ratios.csv", index=False); failures.to_csv(
        tables / "odds_ratio_failures.csv", index=False)
    representative = odds.loc[odds.region.eq(cfg["representative_region"]) &
                              odds.feature.ne("const") & odds.identifiable & odds.odds_ratio_representable].copy()
    representative["magnitude"] = representative.log_odds.abs()
    representative = representative.nlargest(15, "magnitude").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(representative.odds_ratio, range(len(representative)),
                xerr=[representative.odds_ratio - representative.ci_low,
                      representative.ci_high - representative.odds_ratio], fmt="o")
    ax.set_yticks(range(len(representative)), representative.feature)
    ax.axvline(1, color="black", linewidth=.8); ax.set_xscale("log")
    if representative.empty:
        ax.text(.5, .5, "Finite ordinary-MLE odds ratios unavailable; see fit diagnostics",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_title(f"Logistic-reference odds ratios — {cfg['representative_region']} vs others")
    _save(fig, figures, "odds_ratio_forest_representative")
    metadata = {"status": "GENERATED_PENDING_INDEPENDENT_REVIEW", "model": model_id, "scope": "validation",
                "test_used": False, "representative_attack": cfg["representative_attack"],
                "seed": cfg["seed"], "regions": len(regions), "shap_algorithm": "permutation",
                "explanation_matrix_dtype": np.dtype(explanation_dtype).name,
                "tree_inference_representation": (
                    "float64 SHAP/LIME arrays mapped to exact clean-train categorical strings"
                    if explanation_dtype == np.float64 else "existing float32 TabPFN matrix contract"
                ),
                "shap_beeswarm_pngs": len(list(figures.glob("shap_beeswarm_*.png"))),
                "background_rows": len(background), "top_feature": top_feature,
                "top_feature_removed": True, "runtime": factory.tabpfn_runtime,
                "python": platform.python_version(), "seconds": time.time() - started}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(
        {"global_shap": summary.to_dict(orient="records"), "top_feature": top_feature,
         "odds_failures": failures.to_dict(orient="records")}, indent=2) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "report.md").write_text(
        f"# 07. 선택 모델 SHAP·LIME·odds ratio\n\nclean validation 1위 `{model_id}`의 17개 "
        f"OvR 모델을 clean과 `{cfg['representative_attack']}`에서 설명했다. SHAP은 5개 원본 "
        "피처의 음·양 방향, 점 분포와 피처값 색상을 함께 표시하는 beeswarm dot plot이며 "
        "17지역×2조건 34개를 300dpi PNG로 저장했다.\n\n"
        f"전체 조건 평균 |SHAP| 1위 `{top_feature}`를 제거한 설명도 재계산했다. LIME, "
        "좌표 dependence, waterfall/force와 별도 다변량 Binomial GLM의 odds ratio·95% CI·"
        "p-value·FDR q-value는 tables, figures, html에 기록했다. test는 사용하지 않았다.\n")
    (output / "_SUCCESS.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/13_explainability_selected.yaml")
    parser.add_argument("--output")
    args = parser.parse_args(); run(args.config, args.output)


if __name__ == "__main__":
    main()
