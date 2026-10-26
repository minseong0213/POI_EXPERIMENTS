"""Executable train/validation-only integrity and vocabulary gate for defense training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from poi.adversarial import (
    ATTACK_SCHEMA_VERSION,
    QUERY_COUNT_SCOPE,
    SURROGATE_ID,
    attack_condition_specs,
)
from poi.data import CATEGORICAL, FEATURES, load_clean_splits
from poi.ure.preprocessing import UREPreprocessor

from .data import (
    ATTACK_PROVENANCE,
    _strict_boolean,
    _validate_attack_integrity,
    raw_feature_matrix,
)
from .schema import POIFeatureCodebook


def run(
    *,
    data_dir: str | Path,
    attack_table: str | Path,
    output: str | Path,
    attack_generation_config: str | Path = "configs/stages/01_attack_generation.yaml",
    surrogate_artifact_dir: str | Path | None = None,
    clean_only: bool = False,
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Validate raw precision and frozen train support without opening locked test."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite defense preflight: {output}")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "scope": "clean train/validation and generated train/validation attacks only",
        "data_dir": str(data_dir),
        "attack_table": str(attack_table),
        "attack_generation_config": str(attack_generation_config),
        "surrogate_artifact_dir": (
            str(surrogate_artifact_dir) if surrogate_artifact_dir is not None else None
        ),
        "test_rows_read": 0,
        "model_encoding_dtype": "float32 after raw-float64 integrity validation",
        "unknown_category_policy": (
            "fail preflight: proposed and URE encoders have no unknown-category bucket; "
            "all-intent scoring must remain representable even for quarantined rows"
        ),
        "implementation_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("preflight.py", "data.py", "schema.py")
        },
        "dependency_sha256": {
            f"src/poi/{name}": _sha256(
                Path(__file__).resolve().parents[1] / name
            )
            for name in ("adversarial.py", "data.py", "geospatial.py")
        },
        "ure_preprocessing_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "ure" / "preprocessing.py"
        ),
    }
    try:
        clean, _ = load_clean_splits(data_dir, splits=("train", "validation"))
        for feature in FEATURES:
            clean[feature] = pd.to_numeric(clean[feature], errors="raise")
        train = clean.loc[clean["split"].eq("train")].reset_index(drop=True)
        validation = clean.loc[clean["split"].eq("validation")].reset_index(drop=True)
        if len(train) != 23_800 or len(validation) != 3_400:
            raise ValueError(
                f"Frozen clean split counts differ: train={len(train)}, validation={len(validation)}"
            )
        codebook = POIFeatureCodebook.fit(train)
        ure = UREPreprocessor.fit(train)
        train_raw = raw_feature_matrix(train)
        validation_raw = raw_feature_matrix(validation)
        train_tuples = {tuple(row) for row in train_raw[:, 2:]}
        unseen_validation_tuples = np.asarray(
            [tuple(row) not in train_tuples for row in validation_raw[:, 2:]], dtype=bool
        )
        unknown_by_feature: dict[str, int] = {}
        for offset, feature in enumerate(CATEGORICAL, start=2):
            vocabulary = set(train_raw[:, offset])
            unknown_by_feature[feature] = int(
                (~np.isin(validation_raw[:, offset], list(vocabulary))).sum()
            )
        if any(unknown_by_feature.values()):
            raise ValueError(f"Clean validation has unknown category codes: {unknown_by_feature}")
        # Both production encoders must consume every clean validation row.
        codebook.encode(validation)
        ure.transform(validation)
        encoded_train = codebook.encode(train).cpu().numpy()
        raw_min = train_raw[:, :2].min(axis=0)
        raw_max = train_raw[:, :2].max(axis=0)
        model_min = np.asarray(codebook.numeric_model_min)
        model_max = np.asarray(codebook.numeric_model_max)
        if (model_min < raw_min).any() or (model_max > raw_max).any():
            raise RuntimeError("Inward float32 deployment bounds escape raw train bounds")
        if (encoded_train[:, :2] < raw_min).any() or (encoded_train[:, :2] > raw_max).any():
            raise RuntimeError("Float32 clean-train encoding escapes raw train bounds")
        report["clean"] = {
            "status": "PASS",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "regions_train": int(train["region"].nunique()),
            "regions_validation": int(validation["region"].nunique()),
            "raw_numeric_dtype": str(train_raw.dtype),
            "raw_numeric_min": raw_min.tolist(),
            "raw_numeric_max": raw_max.tolist(),
            "inward_float32_numeric_min": model_min.tolist(),
            "inward_float32_numeric_max": model_max.tolist(),
            "encoded_boundary_inside_raw_bounds": True,
            "category_cardinalities": list(codebook.cardinalities),
            "valid_train_category_tuples": len(train_tuples),
            "validation_rows_with_unseen_train_tuple": int(unseen_validation_tuples.sum()),
            "validation_unknown_individual_codes": unknown_by_feature,
            "proposed_validation_encoding": "PASS",
            "ure_validation_encoding": "PASS",
            "source_sha256": {
                name: _sha256(Path(data_dir) / name)
                for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
            },
        }

        attack_path = Path(attack_table)
        if clean_only:
            report.update(
                {
                    "status": "CLEAN_PASS_ATTACK_PENDING",
                    "training_ready": False,
                    "attack": {
                        "status": "NOT_RUN",
                        "reason": "clean_only requested; generated attack table remains pending",
                        "expected_path": str(attack_path),
                    },
                }
            )
        else:
            if not attack_path.is_file():
                raise FileNotFoundError(f"Generated attack table is absent: {attack_path}")
            start_snapshot = _attack_file_snapshot(attack_path)
            generation = _validate_producer_lineage(
                attack_path,
                Path(attack_generation_config),
                Path(data_dir),
                surrogate_artifact_dir=(
                    Path(surrogate_artifact_dir)
                    if surrogate_artifact_dir is not None else None
                ),
            )
            inventory = FullAttackInventoryValidator(
                clean,
                specs=generation["specs"],
                generation_seeds=generation["generation_seeds"],
                generation_identity_sha256=generation["generation_identity_sha256"],
                surrogate_checkpoint_sha256=generation["surrogate_checkpoint_sha256"],
            )
            counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            total = unsupported = batches = 0
            seen_splits: set[str] = set()
            for chunk in _attack_chunks(attack_path, batch_rows=batch_rows):
                _validate_attack_chunk_schema(chunk)
                for feature in FEATURES:
                    chunk[f"{feature}_clean"] = pd.to_numeric(
                        chunk[f"{feature}_clean"], errors="raise"
                    )
                    chunk[f"{feature}_adv"] = pd.to_numeric(
                        chunk[f"{feature}_adv"], errors="raise"
                    )
                splits = set(chunk["split"].astype(str))
                if not splits.issubset({"train", "validation"}):
                    raise RuntimeError("Locked test rows entered defense preflight")
                seen_splits.update(splits)
                inventory.consume(chunk)
                _validate_attack_integrity(clean, chunk)
                supported = _strict_boolean(chunk, "valid_category_codebook")
                unsupported += int((~supported).sum())
                if supported.all():
                    codebook.encode(raw_feature_matrix(chunk, "_clean"))
                    codebook.encode(raw_feature_matrix(chunk, "_adv"))
                    ure.transform(raw_feature_matrix(chunk, "_clean"))
                    ure.transform(raw_feature_matrix(chunk, "_adv"))
                total += len(chunk)
                batches += 1
                for key, group in chunk.groupby(
                    ["generation_seed", "attack_condition", "split"], sort=False
                ):
                    row = counts[tuple(key)]
                    row["rows"] += len(group)
                    for column in (
                        "primary_evaluation_eligible", "constraints_valid",
                        "label_preserved", "valid_category_codebook",
                        "valid_clean_category_combo", "valid_attack_category_combo",
                        "valid_category_combo", "within_budget",
                    ):
                        row[f"{column}_rows"] += int(_strict_boolean(group, column).sum())
            if seen_splits != {"train", "validation"}:
                raise ValueError(f"Attack preflight requires train+validation, got {seen_splits}")
            if unsupported:
                raise ValueError(
                    f"{unsupported} all-intent attack rows contain categories unsupported "
                    "by the clean-train proposed/URE encoders"
                )
            inventory_summary = inventory.finalize()
            end_snapshot = _attack_file_snapshot(attack_path)
            if start_snapshot != end_snapshot:
                raise RuntimeError("Attack table changed while preflight streamed it")
            if start_snapshot["sha256"] != generation["attack_table_sha256"]:
                raise RuntimeError("Attack table differs from immutable producer metadata")
            rows = [
                {"generation_seed": key[0], "attack_condition": key[1], "split": key[2], **value}
                for key, value in sorted(counts.items())
            ]
            pd.DataFrame(rows).to_csv(output / "attack_counts_by_seed_condition.csv", index=False)
            report.update(
                {
                    "status": "PASS",
                    "training_ready": True,
                    "attack": {
                        "status": "PASS",
                        "path": str(attack_path),
                        "sha256": end_snapshot["sha256"],
                        "rows": total,
                        "stream_batches": batches,
                        "group_rows": len(rows),
                        "splits": sorted(seen_splits),
                        "unsupported_all_intent_category_rows": unsupported,
                        "raw_integrity_precision": "float64 before model encoding",
                        "complete_validated_inventory": inventory_summary,
                        "producer_lineage": generation["report"],
                        "stable_snapshot_before_after": True,
                    },
                }
            )
    except Exception as error:
        report.update(
            {"status": "FAIL", "training_ready": False, "error": f"{type(error).__name__}: {error}"}
        )
        (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _validate_attack_chunk_schema(frame: pd.DataFrame) -> None:
    required = set(ATTACK_PROVENANCE)
    required.update(f"{feature}_clean" for feature in FEATURES)
    required.update(f"{feature}_adv" for feature in FEATURES)
    required.update(f"changed_{feature.lower()}" for feature in FEATURES)
    required.update(
        {
            "attack_schema_version", "generation_identity_sha256",
            "surrogate_checkpoint_sha256", "query_count_scope",
            "victim_evaluation_scope", "transfer_evaluated",
        }
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Attack preflight is missing columns: {sorted(missing)}")


def _attack_chunks(path: Path, *, batch_rows: int) -> Iterator[pd.DataFrame]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            yield batch.to_pandas()
    elif path.suffix == ".csv":
        yield from pd.read_csv(path, chunksize=batch_rows)
    else:
        raise ValueError("Attack preflight requires CSV or Parquet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FullAttackInventoryValidator:
    """Exact 3-seed x 32-condition membership check backed by compact bitsets."""

    def __init__(
        self,
        clean: pd.DataFrame,
        *,
        specs: list[dict[str, Any]],
        generation_seeds: list[int],
        generation_identity_sha256: str,
        surrogate_checkpoint_sha256: str,
        expected_split_counts: dict[str, int] | None = None,
    ) -> None:
        if set(clean["split"]) != {"train", "validation"} or clean["POI_ID"].duplicated().any():
            raise ValueError("Inventory clean manifest must have unique train/validation POIs")
        if generation_seeds != [42, 202, 340] or len(specs) != 32:
            raise ValueError("Inventory contract requires seeds [42,202,340] and 32 conditions")
        self.specs = {str(spec["condition"]): spec for spec in specs}
        if len(self.specs) != 32:
            raise ValueError("Attack conditions must be 32 unique names")
        self.generation_seeds = tuple(generation_seeds)
        self.generation_identity_sha256 = generation_identity_sha256
        self.surrogate_checkpoint_sha256 = surrogate_checkpoint_sha256
        self.id_truth: dict[str, tuple[str, str, int]] = {}
        self.split_ids: dict[str, list[str]] = {}
        for split in ("train", "validation"):
            frame = clean.loc[clean["split"].eq(split)].reset_index(drop=True)
            ids = frame["POI_ID"].astype(str).tolist()
            self.split_ids[split] = ids
            for index, (poi_id, region) in enumerate(zip(ids, frame["region"].astype(str))):
                self.id_truth[poi_id] = (split, region, index)
        self.expected_split_counts = expected_split_counts or {
            "train": 23_800,
            "validation": 3_400,
        }
        observed_split_counts = {
            split: len(ids) for split, ids in self.split_ids.items()
        }
        if observed_split_counts != self.expected_split_counts:
            raise ValueError(
                "Inventory clean split counts differ from contract: "
                f"{observed_split_counts} != {self.expected_split_counts}"
            )
        self.coverage = {
            (seed, condition, split): np.zeros(len(self.split_ids[split]), dtype=bool)
            for seed in self.generation_seeds
            for condition in self.specs
            for split in ("train", "validation")
        }
        self.rows = 0

    def consume(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        ids = frame["POI_ID"].astype(str).to_numpy()
        unknown = sorted(set(ids).difference(self.id_truth))
        if unknown:
            raise ValueError(f"Attack inventory contains unknown POI_ID values: {unknown[:5]}")
        expected_split = np.asarray([self.id_truth[value][0] for value in ids])
        observed_split = frame["split"].astype(str).to_numpy()
        if not np.array_equal(observed_split, expected_split):
            raise ValueError("Attack split differs from frozen clean manifest truth")
        expected_region = np.asarray([self.id_truth[value][1] for value in ids])
        if not np.array_equal(frame["region"].astype(str).to_numpy(), expected_region):
            raise ValueError("Attack region differs from frozen clean manifest truth")
        seed_values = pd.to_numeric(
            frame["generation_seed"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(seed_values).all()
            or not np.equal(seed_values, np.floor(seed_values)).all()
        ):
            raise ValueError("Attack generation seeds must be finite integers")
        seeds = seed_values.astype(np.int64)
        if not np.isin(seeds, self.generation_seeds).all():
            raise ValueError("Attack inventory has an undeclared generation seed")
        conditions = frame["attack_condition"].astype(str).to_numpy()
        if not np.isin(conditions, list(self.specs)).all():
            raise ValueError("Attack inventory has an undeclared attack condition")
        scalar = {
            "generation_identity_sha256": self.generation_identity_sha256,
            "surrogate_checkpoint_sha256": self.surrogate_checkpoint_sha256,
            "attack_schema_version": ATTACK_SCHEMA_VERSION,
            "query_count_scope": QUERY_COUNT_SCOPE,
            "source_model": SURROGATE_ID,
            "victim_model": SURROGATE_ID,
            "victim_evaluation_scope": "direct_white_box_source_model",
        }
        for column, expected in scalar.items():
            if not frame[column].eq(expected).all():
                raise ValueError(f"Attack {column} differs from immutable producer identity")
        if _strict_boolean(frame, "transfer_evaluated").any():
            raise ValueError("Primary generated attacks must not claim transfer evaluation")
        if not _strict_boolean(frame, "attack_attempted").all():
            raise ValueError("Full generated inventory must contain attempted rows")
        for condition, group in frame.groupby("attack_condition", sort=False):
            spec = self.specs[str(condition)]
            if not group["attack_method"].astype(str).eq(str(spec["method"])).all():
                raise ValueError(f"Attack method differs for condition {condition}")
            expected_numeric = 0.0 if spec["numeric_budget_m"] is None else float(spec["numeric_budget_m"])
            expected_implicit = spec["numeric_budget_m"] is None
            if not np.equal(
                pd.to_numeric(group["numeric_budget_m"], errors="raise").to_numpy(float),
                expected_numeric,
            ).all():
                raise ValueError(f"Numeric budget differs for condition {condition}")
            if not np.equal(
                pd.to_numeric(group["categorical_budget_l0"], errors="raise").to_numpy(float),
                float(spec["categorical_budget_l0"]),
            ).all():
                raise ValueError(f"Categorical budget differs for condition {condition}")
            if not np.equal(
                _strict_boolean(group, "numeric_budget_was_implicit_zero"), expected_implicit
            ).all():
                raise ValueError(f"Implicit numeric-budget flag differs for {condition}")
        positions = np.asarray([self.id_truth[value][2] for value in ids], dtype=np.int64)
        temporary = pd.DataFrame(
            {"seed": seeds, "condition": conditions, "split": observed_split, "position": positions}
        )
        for key, group in temporary.groupby(["seed", "condition", "split"], sort=False):
            normalized = (int(key[0]), str(key[1]), str(key[2]))
            selected = group["position"].to_numpy(np.int64)
            if len(np.unique(selected)) != len(selected) or self.coverage[normalized][selected].any():
                raise ValueError(f"Duplicate POI membership in attack group {normalized}")
            self.coverage[normalized][selected] = True
        self.rows += len(frame)

    def finalize(self) -> dict[str, Any]:
        incomplete = {
            str(key): int((~coverage).sum())
            for key, coverage in self.coverage.items()
            if not coverage.all()
        }
        if incomplete:
            preview = dict(list(incomplete.items())[:5])
            raise ValueError(f"Incomplete attack group POI membership: {preview}")
        expected_rows = sum(len(values) for values in self.coverage.values())
        if self.rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} attack rows, got {self.rows}")
        return {
            "status": "PASS",
            "rows": self.rows,
            "expected_rows": expected_rows,
            "clean_development_pois": sum(len(ids) for ids in self.split_ids.values()),
            "expected_clean_development_pois": sum(self.expected_split_counts.values()),
            "clean_pois_by_split": {
                split: len(ids) for split, ids in self.split_ids.items()
            },
            "expected_clean_pois_by_split": self.expected_split_counts,
            "groups": len(self.coverage),
            "expected_groups": 192,
            "generation_seed_count": len(self.generation_seeds),
            "attack_condition_count": len(self.specs),
            "generation_seeds": list(self.generation_seeds),
            "attack_conditions": sorted(self.specs),
            "unique_complete_poi_membership": True,
            "split_region_manifest_truth": True,
        }


def _validate_producer_lineage(
    attack_path: Path,
    generation_config_path: Path,
    data_dir: Path,
    *,
    surrogate_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    stage = attack_path.parent.parent
    required = {
        "metadata": stage / "metadata.json",
        "marker": stage / "_PENDING_REVIEW.json",
        "request": stage / "work" / "generation_request.json",
        "identity": stage / "work" / "generation_identity.json",
        "manifest": stage / "work" / "shard_manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Attack producer completion lineage is missing: {missing}")
    metadata = json.loads(required["metadata"].read_text())
    marker = json.loads(required["marker"].read_text())
    request = json.loads(required["request"].read_text())
    identity = json.loads(required["identity"].read_text())
    manifest = json.loads(required["manifest"].read_text())
    if marker != metadata or metadata.get("status") != "generated_pending_fresh_independent_review":
        raise ValueError("Attack producer marker and metadata completion states differ")
    local_cfg = yaml.safe_load(generation_config_path.read_text())
    specs = attack_condition_specs(local_cfg)
    seeds = [int(value) for value in local_cfg["generation_seeds"]]
    if seeds != [42, 202, 340] or len(specs) != 32:
        raise ValueError("Local attack config differs from frozen 3x32 contract")
    if local_cfg.get("splits") != ["train", "validation"]:
        raise ValueError("Local attack config must generate train and validation only")
    if request.get("config_sha256") != _sha256(generation_config_path):
        raise ValueError("Attack generation config changed after immutable request")
    if request.get("conditions") != specs or request.get("generation_seeds") != seeds:
        raise ValueError("Immutable request conditions/seeds differ from local config")
    if (
        request.get("format") != "poi-generation-request-v2"
        or request.get("attack_schema_version") != ATTACK_SCHEMA_VERSION
        or request.get("query_count_scope") != QUERY_COUNT_SCOPE
        or request.get("splits") != ["train", "validation"]
        or request.get("features") != FEATURES
    ):
        raise ValueError("Immutable request schema/scope differs from the producer contract")
    expected_rows = 27_200 * 32 * 3
    if request.get("expected_rows") != expected_rows or request.get("expected_shards") != 192:
        raise ValueError("Immutable request row/shard cardinality is not the full contract")
    clean_hashes = {
        name: _sha256(data_dir / name)
        for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
    }
    if request.get("dataset_sha256") != clean_hashes:
        raise ValueError("Immutable request dataset hashes differ from configured clean data")
    producer_source_paths = {
        name: Path(__file__).resolve().parents[1] / name
        for name in ("adversarial.py", "data.py", "geospatial.py")
    }
    request_code = request.get("code_sha256")
    if not isinstance(request_code, dict):
        raise ValueError("Immutable request lacks producer source hashes")
    request_code_by_name: dict[str, str] = {}
    for original_path, expected_hash in request_code.items():
        name = Path(original_path).name
        if name in request_code_by_name:
            raise ValueError(f"Immutable request repeats producer source basename: {name}")
        request_code_by_name[name] = expected_hash
    if set(request_code_by_name) != set(producer_source_paths):
        raise ValueError("Immutable request producer source inventory differs from contract")
    current_producer_code = {
        name: _sha256(path) for name, path in producer_source_paths.items()
    }
    if request_code_by_name != current_producer_code:
        raise ValueError("Producer request source hashes differ from current implementation")
    identity_without_self = dict(identity)
    observed_identity_sha = identity_without_self.pop("generation_identity_sha256", None)
    if identity.get("generation_request_sha256") != _canonical_sha256(request):
        raise ValueError("Generation identity does not bind the immutable request")
    if identity.get("format") != "poi-generation-identity-v2":
        raise ValueError("Generation identity format differs from producer contract")
    if observed_identity_sha != _canonical_sha256(identity_without_self):
        raise ValueError("Generation identity self-hash is invalid")
    if (
        manifest.get("format") != "poi-shard-manifest-v2"
        or manifest.get("generation_identity_sha256") != observed_identity_sha
    ):
        raise ValueError("Shard manifest differs from generation identity")
    expected_shard_names = {
        f"{split}-g{seed}-{spec['condition']}.parquet"
        for seed in seeds
        for split in ("train", "validation")
        for spec in specs
    }
    shards = manifest.get("shards", {})
    if set(shards) != expected_shard_names:
        raise ValueError("Shard manifest does not register all 192 shards")
    expected_split_rows = {"train": 23_800, "validation": 3_400}
    for name, entry in shards.items():
        split = name.split("-g", 1)[0]
        if (
            not isinstance(entry, dict)
            or entry.get("rows") != expected_split_rows[split]
            or not _is_sha256(entry.get("sha256"))
            or not _is_sha256(entry.get("ordered_poi_ids_sha256"))
            or not _is_sha256(entry.get("key_sha256"))
        ):
            raise ValueError(f"Shard manifest has incomplete provenance: {name}")
    shard_directory = stage / "work" / "shards"
    present_shards = {path.name: path for path in shard_directory.glob("*.parquet")}
    if present_shards:
        if set(present_shards) != expected_shard_names:
            raise ValueError("Available producer shard bytes differ from the 192-entry manifest")
        for name, path in present_shards.items():
            if _sha256(path) != shards[name]["sha256"]:
                raise ValueError(f"Producer shard hash differs from manifest: {name}")
        shard_artifact_verification = "verified_all_192_manifest_hashes"
    else:
        shard_artifact_verification = (
            "manifest_only_not_reverified; final combined table independently hashed"
        )
    if metadata.get("generation_request", {}).get("sha256") != _canonical_sha256(request):
        raise ValueError("Producer metadata differs from the immutable generation request")
    metadata_identity = dict(metadata.get("generation_identity", {}))
    metadata_identity.pop("path", None)
    if metadata_identity != identity:
        raise ValueError("Producer metadata generation identity differs")
    if (
        metadata.get("shard_manifest", {}).get("sha256") != _sha256(required["manifest"])
        or metadata.get("shard_manifest", {}).get("registered_shards") != 192
    ):
        raise ValueError("Producer metadata shard-manifest hash differs")
    checkpoint_sha = identity.get("surrogate_checkpoint_sha256")
    if not _is_sha256(checkpoint_sha) or metadata.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("Producer metadata checkpoint hash differs from generation identity")
    if not _is_sha256(identity.get("surrogate_encoder_sha256")) or not _is_sha256(
        identity.get("surrogate_training_history_sha256")
    ):
        raise ValueError("Generation identity has invalid surrogate artifact hashes")
    checkpoint_candidates = []
    if surrogate_artifact_dir is not None:
        checkpoint_candidates.append(surrogate_artifact_dir / "surrogate.pt")
    checkpoint_candidates.append(stage / "work" / "surrogate" / "surrogate.pt")
    metadata_checkpoint_value = metadata.get("checkpoint", {}).get("path")
    if metadata_checkpoint_value:
        checkpoint_candidates.append(Path(str(metadata_checkpoint_value)))
    selected_surrogate: tuple[Path, Path, Path] | None = None
    inspected_directories: list[str] = []
    deduplicated_candidates: list[Path] = []
    for candidate in checkpoint_candidates:
        if candidate not in deduplicated_candidates:
            deduplicated_candidates.append(candidate)
    for checkpoint in deduplicated_candidates:
        encoder = checkpoint.with_name("surrogate_encoder.json")
        history = checkpoint.with_name("surrogate_training.csv")
        inspected_directories.append(str(checkpoint.parent))
        available = [path.is_file() for path in (checkpoint, encoder, history)]
        if any(available) and not all(available):
            raise FileNotFoundError(
                f"Producer surrogate artifact set is incomplete: {checkpoint.parent}"
            )
        if all(available):
            for path, expected in (
                (checkpoint, checkpoint_sha),
                (encoder, identity["surrogate_encoder_sha256"]),
                (history, identity["surrogate_training_history_sha256"]),
            ):
                if _sha256(path) != expected:
                    raise ValueError(f"Producer lineage artifact hash differs: {path}")
            selected_surrogate = (checkpoint, encoder, history)
            break
    if selected_surrogate is None:
        raise FileNotFoundError(
            "Producer surrogate checkpoint/encoder/history bytes are required; inspected "
            f"{inspected_directories}"
        )
    checkpoint, encoder, history = selected_surrogate
    table_meta = metadata.get("tables", {}).get("attacks", {})
    if table_meta.get("rows") != expected_rows or not _is_sha256(table_meta.get("sha256")):
        raise ValueError("Producer metadata attack row count is incomplete")
    if (
        metadata.get("source_model") != SURROGATE_ID
        or metadata.get("generation_seeds") != seeds
        or metadata.get("attack_conditions") != 32
        or metadata.get("attack_rows") != expected_rows
        or metadata.get("loaded_feature_splits") != ["train", "validation"]
        or metadata.get("test_used") is not False
    ):
        raise ValueError("Producer metadata scope/cardinality differs from full generation")
    if metadata.get("dataset_sha256") != {
        name: clean_hashes[name] for name in ("poi_data_region.csv", "sample_manifest.csv")
    }:
        raise ValueError("Producer metadata clean hashes differ from configured data")
    return {
        "specs": specs,
        "generation_seeds": seeds,
        "generation_identity_sha256": observed_identity_sha,
        "surrogate_checkpoint_sha256": checkpoint_sha,
        "attack_table_sha256": table_meta.get("sha256"),
        "report": {
            "status": metadata["status"],
            "metadata_sha256": _sha256(required["metadata"]),
            "pending_marker_sha256": _sha256(required["marker"]),
            "generation_request_sha256": _sha256(required["request"]),
            "generation_identity_sha256": observed_identity_sha,
            "generation_identity_file_sha256": _sha256(required["identity"]),
            "shard_manifest_sha256": _sha256(required["manifest"]),
            "registered_shards": 192,
            "shard_artifact_bytes": shard_artifact_verification,
            "surrogate_checkpoint_sha256": checkpoint_sha,
            "surrogate_encoder_sha256": identity["surrogate_encoder_sha256"],
            "surrogate_training_history_sha256": identity[
                "surrogate_training_history_sha256"
            ],
            "surrogate_artifact_bytes": "verified_checkpoint_encoder_history",
            "local_generation_config_sha256": _sha256(generation_config_path),
            "clean_dataset_sha256": clean_hashes,
            "producer_request_code_sha256": request_code_by_name,
            "artifact_paths": {
                **{name: str(path.resolve()) for name, path in required.items()},
                "surrogate_checkpoint": str(checkpoint.resolve()),
                "surrogate_encoder": str(encoder.resolve()),
                "surrogate_training_history": str(history.resolve()),
                "shard_directory": str(shard_directory.resolve()),
                "attack_table": str(attack_path.resolve()),
                "generation_config": str(generation_config_path.resolve()),
                "data_dir": str(data_dir.resolve()),
                **{
                    f"producer_source_{name}": str(path.resolve())
                    for name, path in producer_source_paths.items()
                },
            },
        },
    }


def _attack_file_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError("Attack table is empty")
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "sha256": _sha256(path),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stages/08_proposed_tabpfn_defense.yaml")
    parser.add_argument("--output")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=65_536)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    report = run(
        data_dir=cfg["data_dir"], attack_table=cfg["attack_table"],
        attack_generation_config=cfg["attack_generation_config"],
        surrogate_artifact_dir=cfg.get("surrogate_artifact_dir"),
        output=args.output or cfg["preflight_output"], clean_only=args.clean_only,
        batch_rows=args.batch_rows,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
