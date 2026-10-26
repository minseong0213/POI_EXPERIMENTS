"""Materialize and optionally run the declared Stage-14 defense ablations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


STRUCTURAL_FLAGS = {
    "use_reliability_gate",
    "use_reconstruction_path",
    "use_multiscale_encoder",
    "use_feature_encoders",
    "use_category_relations",
    "use_corruption_heads",
}
ALLOWED_KINDS = {"reference", "structural", "objective", "optimization"}


def resolve_ablation_configs(
    matrix_path: str | Path,
) -> list[tuple[str, bool, dict[str, Any]]]:
    """Resolve a matrix into complete Stage-08 configs with auditable labels."""
    matrix_path = Path(matrix_path)
    matrix = yaml.safe_load(matrix_path.read_text())
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        # Repository configs use repository-root-relative paths.  A matrix outside the
        # repository may instead point to a file relative to itself.
        repository_relative = Path.cwd() / base_path
        base_path = repository_relative if repository_relative.is_file() else matrix_path.parent / base_path
    base = yaml.safe_load(base_path.read_text())
    variants = matrix.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Ablation matrix must contain a non-empty variants list")

    resolved: list[tuple[str, bool, dict[str, Any]]] = []
    seen: set[str] = set()
    for variant in variants:
        name = str(variant["name"])
        kind = str(variant["kind"])
        if name in seen:
            raise ValueError(f"Duplicate ablation name: {name}")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"Unsupported ablation kind for {name}: {kind}")
        seen.add(name)
        overrides = copy.deepcopy(variant.get("overrides", {}))
        config = _deep_merge(copy.deepcopy(base), overrides)
        config["run_label"] = name
        config["ablation_kind"] = kind
        config["ablation_changes"] = list(variant.get("changes", []))
        config["output"] = str(variant["output"])
        _validate_variant(name, kind, base, config)
        resolved.append((name, bool(variant.get("train", True)), config))
    return resolved


def materialize_ablation_configs(
    matrix_path: str | Path, output_dir: str | Path
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, train, config in resolve_ablation_configs(matrix_path):
        config["ablation_train"] = train
        path = output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    return paths


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _validate_variant(
    name: str, kind: str, base: dict[str, Any], config: dict[str, Any]
) -> None:
    base_model = base["model"]
    model = config["model"]
    changed_flags = {
        key for key in STRUCTURAL_FLAGS if model.get(key, True) != base_model.get(key, True)
    }
    consistency_changed = (
        config["loss_weights"]["consistency"]
        != base["loss_weights"]["consistency"]
    )
    optimization_changed = (
        model.get("unfreeze_last_blocks", 0) != base_model.get("unfreeze_last_blocks", 0)
        or config.get("fine_tuning", {}) != base.get("fine_tuning", {})
    )
    if kind == "reference" and (changed_flags or consistency_changed or optimization_changed):
        raise ValueError(f"Reference {name} changes the model or objective")
    if kind == "structural" and not changed_flags:
        raise ValueError(f"Structural ablation {name} changes no forward-path flag")
    if kind == "objective" and (changed_flags or optimization_changed or not consistency_changed):
        raise ValueError(f"Objective ablation {name} must change only its objective")
    if kind == "optimization" and (changed_flags or consistency_changed or not optimization_changed):
        raise ValueError(f"Optimization ablation {name} must change only trainability")
    if not model.get("use_reconstruction_path", True):
        if float(config["loss_weights"]["reconstruction"]) != 0.0:
            raise ValueError(f"{name} disables reconstruction but retains its loss")
    if not model.get("use_corruption_heads", True):
        if float(config["loss_weights"]["feature_detection"]) != 0.0:
            raise ValueError(f"{name} disables corruption heads but retains their loss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", default="configs/stages/14_proposed_structure_ablation.yaml"
    )
    parser.add_argument(
        "--write-dir",
        default="artifacts/poi-remediation-20261001/work/stage14_resolved_configs",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Train each non-reference materialized config sequentially.",
    )
    args = parser.parse_args()
    paths = materialize_ablation_configs(args.matrix, args.write_dir)
    if args.execute:
        from .train import run

        for path in paths:
            cfg = yaml.safe_load(path.read_text())
            if cfg.pop("ablation_train"):
                run(path)
    else:
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
