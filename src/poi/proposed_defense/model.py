"""Structural adapter composed with the official TabPFN v2.5 model forward."""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .schema import POIFeatureCodebook
from .structural import StructuralDefenseAdapter, StructuralDefenseOutput

TABPFN_SOURCE_COMMIT = "316663a3eb32c40afe1068744cc16cc9900b98a0"
TABPFN_VERSION = "8.1.0"
TABPFN_V25_ARCHITECTURE_SHA256 = "bc40f347ca2d96ee2771222b8e7a06570ed19d8385124fcc3d9e9768a48bde0d"
CHECKPOINT_COMPATIBILITY_PATCH = "copy_mutable_block_container_before_recompute_v1"


@dataclass
class ProposedDefenseOutput:
    region_logits: torch.Tensor
    base_logits: torch.Tensor
    query_embeddings: torch.Tensor
    structural: StructuralDefenseOutput


@dataclass
class ProjectedRegionOutput:
    """Regional output evaluated on a finite, train-support-valid hard repair."""

    region_logits: torch.Tensor
    base_logits: torch.Tensor
    query_embeddings: torch.Tensor


class ProposedTabPFNDefense(nn.Module):
    """A real structural modification around an official TabPFN v2.5 checkpoint.

    The same repaired feature matrix is expanded into one binary in-context prompt per
    region.  The official transformer returns a query embedding for every OvR prompt;
    seventeen separate output weights map the corresponding embeddings to OvR logits.
    """

    def __init__(
        self,
        base_model: nn.Module,
        codebook: POIFeatureCodebook,
        regions: tuple[str, ...],
        *,
        hidden_dim: int = 64,
        numeric_scales: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
        activation_checkpointing: bool = True,
        use_reliability_gate: bool = True,
        use_reconstruction_path: bool = True,
        use_multiscale_encoder: bool = True,
        use_feature_encoders: bool = True,
        use_category_relations: bool = True,
        use_corruption_heads: bool = True,
    ) -> None:
        super().__init__()
        if len(regions) != 17:
            raise ValueError(f"The experiment requires exactly 17 regions, got {len(regions)}")
        embedding_dim = getattr(base_model, "embedding_dim", None)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise TypeError("base_model must expose the official Architecture.embedding_dim")
        self.base_model = base_model
        self.adapter = StructuralDefenseAdapter(
            codebook,
            hidden_dim=hidden_dim,
            numeric_scales=numeric_scales,
            use_reliability_gate=use_reliability_gate,
            use_reconstruction_path=use_reconstruction_path,
            use_multiscale_encoder=use_multiscale_encoder,
            use_feature_encoders=use_feature_encoders,
            use_category_relations=use_category_relations,
            use_corruption_heads=use_corruption_heads,
        )
        self.regions = tuple(regions)
        self.region_weights = nn.Parameter(torch.empty(len(regions), embedding_dim))
        self.region_bias = nn.Parameter(torch.zeros(len(regions)))
        nn.init.normal_(self.region_weights, std=embedding_dim**-0.5)
        self.activation_checkpointing = activation_checkpointing
        self.unfrozen_block_count = 0
        if activation_checkpointing:
            _install_checkpoint_safe_v25_block_forwards(self.base_model)
        self.freeze_official_model()

    @property
    def checkpoint_compatibility_patch(self) -> str | None:
        return CHECKPOINT_COMPATIBILITY_PATCH if self.activation_checkpointing else None

    @property
    def codebook(self) -> POIFeatureCodebook:
        return self.adapter.codebook

    @property
    def architecture_flags(self) -> dict[str, bool]:
        """Forward-path switches persisted with every checkpoint."""
        return {
            "use_reliability_gate": self.adapter.use_reliability_gate,
            "use_reconstruction_path": self.adapter.use_reconstruction_path,
            "use_multiscale_encoder": self.adapter.use_multiscale_encoder,
            "use_feature_encoders": self.adapter.use_feature_encoders,
            "use_category_relations": self.adapter.use_category_relations,
            "use_corruption_heads": self.adapter.use_corruption_heads,
        }

    def freeze_official_model(self) -> None:
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self.unfrozen_block_count = 0

    def unfreeze_last_blocks(self, count: int) -> None:
        blocks = getattr(self.base_model, "blocks", None)
        if blocks is None:
            raise TypeError("Official TabPFN v2.5 model does not expose blocks")
        if count < 0 or count > len(blocks):
            raise ValueError(f"Cannot unfreeze {count} of {len(blocks)} blocks")
        self.freeze_official_model()
        if count:
            for block in list(blocks)[-count:]:
                block.requires_grad_(True)
        self.unfrozen_block_count = count

    def train(self, mode: bool = True) -> "ProposedTabPFNDefense":
        super().train(mode)
        # A frozen checkpoint must remain deterministic while preserving gradients with
        # respect to repaired inputs.  If final blocks are explicitly unfrozen, only
        # those blocks follow training mode.
        self.base_model.eval()
        if mode and self.unfrozen_block_count:
            for block in list(self.base_model.blocks)[-self.unfrozen_block_count:]:
                block.train(True)
        return self

    def _performance_options(self) -> Any:
        try:
            from tabpfn.architectures.interface import PerformanceOptions
        except ImportError:
            # A test double does not need TabPFN installed.
            return None
        return PerformanceOptions(
            force_recompute_layer=self.activation_checkpointing,
            save_peak_memory_factor=None,
            use_chunkwise_inference=False,
        )

    def forward(
        self,
        encoded_features: torch.Tensor,
        context_region_indices: torch.Tensor,
    ) -> ProposedDefenseOutput:
        if encoded_features.ndim != 2 or encoded_features.shape[1] != 5:
            raise ValueError("encoded_features must have shape [context+query, 5]")
        context_count = int(context_region_indices.numel())
        if context_count <= 0 or context_count >= len(encoded_features):
            raise ValueError("A non-empty context and query are both required")
        if torch.any(context_region_indices < 0) or torch.any(
            context_region_indices >= len(self.regions)
        ):
            raise ValueError("context_region_indices is outside the 17-region vocabulary")

        structural = self.adapter(encoded_features)
        projected = self._forward_official(structural.repaired_features, context_region_indices)
        return ProposedDefenseOutput(
            region_logits=projected.region_logits,
            base_logits=projected.base_logits,
            query_embeddings=projected.query_embeddings,
            structural=structural,
        )

    def forward_projected_features(
        self,
        repaired_features: torch.Tensor,
        context_region_indices: torch.Tensor,
    ) -> ProjectedRegionOutput:
        """Evaluate the official model and own OvR heads on an already hard repair."""
        if not torch.isfinite(repaired_features).all():
            raise ValueError("Projected features must be finite")
        return self._forward_official(repaired_features, context_region_indices)

    def _forward_official(
        self,
        repaired: torch.Tensor,
        context_region_indices: torch.Tensor,
    ) -> ProjectedRegionOutput:
        if repaired.ndim != 2 or repaired.shape[1] != 5:
            raise ValueError("repaired features must have shape [context+query, 5]")
        context_count = int(context_region_indices.numel())
        if context_count <= 0 or context_count >= len(repaired):
            raise ValueError("A non-empty context and query are both required")
        region_count = len(self.regions)
        x_prompt = repaired.unsqueeze(1).expand(-1, region_count, -1)
        region_axis = torch.arange(region_count, device=repaired.device)
        y_prompt = (context_region_indices[:, None] == region_axis[None, :]).to(repaired.dtype)
        output = self.base_model(
            x_prompt,
            y_prompt,
            only_return_standard_out=False,
            categorical_inds=[[2, 3, 4] for _ in range(region_count)],
            performance_options=self._performance_options(),
            task_type="multiclass",
        )
        if not isinstance(output, dict) or "test_embeddings" not in output:
            raise TypeError(
                "Official TabPFN forward must return test_embeddings when "
                "only_return_standard_out=False"
            )
        embeddings = output["test_embeddings"]
        if embeddings.ndim != 3 or embeddings.shape[1] != region_count:
            raise RuntimeError(f"Unexpected TabPFN embedding shape: {embeddings.shape}")
        region_logits = torch.einsum("qrd,rd->qr", embeddings, self.region_weights)
        region_logits = region_logits + self.region_bias
        base_logits = output.get("standard")
        if not isinstance(base_logits, torch.Tensor):
            base_logits = torch.empty(0, device=embeddings.device)
        return ProjectedRegionOutput(
            region_logits=region_logits,
            base_logits=base_logits,
            query_embeddings=embeddings,
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return adapter/head state and any explicitly unfrozen official blocks."""
        result: dict[str, torch.Tensor] = {}
        blocks = list(getattr(self.base_model, "blocks", []))
        first_unfrozen = len(blocks) - self.unfrozen_block_count
        for key, value in self.state_dict().items():
            if not key.startswith("base_model."):
                result[key] = value.detach().cpu()
            elif self.unfrozen_block_count and key.startswith("base_model.blocks."):
                block_index = int(key.split(".")[2])
                if block_index >= first_unfrozen:
                    result[key] = value.detach().cpu()
        return result


def load_official_tabpfn25(
    *,
    checkpoint_path: str | Path | None,
    device: str | torch.device,
    download_if_missing: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load only the official v2.5 Architecture, with provenance checks."""
    try:
        import tabpfn
        from tabpfn.errors import TabPFNHuggingFaceGatedRepoError
        from tabpfn.model_loading import load_model_criterion_config
    except ImportError as error:
        raise RuntimeError(
            "TabPFN is required; run with .venv-tabpfn/bin/python and vendor/tabpfn v8.1.0"
        ) from error
    if tabpfn.__version__ != TABPFN_VERSION:
        raise RuntimeError(
            f"Expected TabPFN {TABPFN_VERSION}, found {tabpfn.__version__}; refusing an "
            "unrecorded architecture substitution"
        )
    if download_if_missing:
        _configure_priorlabs_token_file()
    try:
        models, _, configs, _ = load_model_criterion_config(
            model_path=checkpoint_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2.5",
            download_if_not_exists=download_if_missing,
        )
    except TabPFNHuggingFaceGatedRepoError as error:
        raise RuntimeError(
            "Official TabPFN 8.1 validated the PriorLabs TABPFN_TOKEN but the "
            "separate anonymous Hugging Face checkpoint request was rejected. "
            "TABPFN_TOKEN is not an HF token and must not be copied into HF_TOKEN. "
            "Seed the exact official checkpoint in ~/.cache/tabpfn, pass "
            "checkpoint_path explicitly, or provide a separately authorized HF token."
        ) from error
    if len(models) != 1:
        raise RuntimeError(f"Expected one v2.5 checkpoint model, got {len(models)}")
    model = models[0]
    module_name = type(model).__module__
    config_name = getattr(configs[0], "name", "")
    if not module_name.endswith("tabpfn_v2_5") or config_name != "TabPFN-v2.5":
        raise RuntimeError(
            f"Loaded architecture is not official v2.5: {module_name}, {config_name}"
        )
    model.to(device)
    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    source_file = Path(inspect.getsourcefile(type(model)) or "").resolve()
    if _sha256(source_file) != TABPFN_V25_ARCHITECTURE_SHA256:
        raise RuntimeError("Imported TabPFN v2.5 architecture source hash is not frozen v8.1.0")
    source_repository, runtime_commit = _runtime_source_revision(source_file)
    if runtime_commit is not None and runtime_commit != TABPFN_SOURCE_COMMIT:
        raise RuntimeError(
            f"Loaded TabPFN source commit {runtime_commit} differs from frozen "
            f"{TABPFN_SOURCE_COMMIT}"
        )
    provenance = {
        "tabpfn_version": tabpfn.__version__,
        "tabpfn_source_commit": TABPFN_SOURCE_COMMIT,
        "architecture_module": module_name,
        "architecture_config_name": config_name,
        "architecture_source_file": str(source_file),
        "architecture_source_sha256": _sha256(source_file),
        "source_repository": str(source_repository) if source_repository else None,
        "runtime_source_commit": runtime_commit,
        "runtime_source_commit_verified": runtime_commit == TABPFN_SOURCE_COMMIT,
        "checkpoint_path": str(resolved_path) if resolved_path else None,
        "checkpoint_sha256": _sha256(resolved_path) if resolved_path and resolved_path.is_file() else None,
    }
    return model, provenance


def _runtime_source_revision(source_file: Path) -> tuple[Path | None, str | None]:
    """Find and attest the Git revision containing the imported architecture source."""
    for candidate in (source_file.parent, *source_file.parents):
        if candidate.name in {"site-packages", "dist-packages"}:
            break
        if (candidate / ".git").exists():
            try:
                commit = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(f"Cannot attest TabPFN source revision at {candidate}") from error
            return candidate, commit
    # A wheel installation has no Git metadata.  Attest it against the repository's
    # pinned official vendor checkout only when the exact architecture file hashes
    # match, then record that checkout's revision.
    repository_root = Path(__file__).resolve().parents[3]
    vendor = repository_root / "vendor" / "tabpfn"
    vendor_source = vendor / "src" / "tabpfn" / "architectures" / "tabpfn_v2_5.py"
    if vendor_source.is_file() and _sha256(vendor_source) == _sha256(source_file):
        try:
            commit = subprocess.run(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("Cannot attest pinned vendor TabPFN revision") from error
        return vendor, commit
    return None, None


def _resolve_checkpoint_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path).expanduser().resolve()
    default = Path.home() / ".cache" / "tabpfn" / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    return default.resolve() if default.exists() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_priorlabs_token_file() -> None:
    """Load an explicitly named PriorLabs token file without logging its contents."""
    if os.environ.get("TABPFN_TOKEN"):
        return
    token_file = os.environ.get("TABPFN_TOKEN_FILE")
    if not token_file:
        return
    path = Path(token_file).expanduser()
    try:
        token = path.read_text().strip()
    except OSError as error:
        raise RuntimeError(f"Cannot read TABPFN_TOKEN_FILE: {path}") from error
    if not token:
        raise RuntimeError(f"TABPFN_TOKEN_FILE is empty: {path}")
    os.environ["TABPFN_TOKEN"] = token


def _install_checkpoint_safe_v25_block_forwards(base_model: nn.Module) -> None:
    """Work around v8.1.0's mutable-list activation-checkpointing incompatibility.

    ``TabPFNV2p5.forward`` gives each block a length-one list, and the official block
    pops the tensor to reduce peak references.  PyTorch's non-reentrant checkpoint
    retains that same Python list for recomputation, where it is then empty.  Copying
    the container at the block boundary preserves the official block computation,
    checkpoint keys, and parameters while making recomputation valid.  This explicit
    instance patch can be removed when the upstream architecture stops mutating its
    checkpoint input.
    """
    if not type(base_model).__module__.endswith("tabpfn_v2_5"):
        return
    blocks = getattr(base_model, "blocks", None)
    if blocks is None:
        raise TypeError("Official TabPFN v2.5 model does not expose transformer blocks")
    for block in blocks:
        if getattr(block, "_poi_checkpoint_safe_forward", False):
            continue
        official_forward = block.forward

        def safe_forward(
            self: nn.Module,
            tensor_container: list[torch.Tensor],
            *args: Any,
            _official_forward: Any = official_forward,
            **kwargs: Any,
        ) -> Any:
            if len(tensor_container) != 1:
                raise RuntimeError(
                    "TabPFN v2.5 checkpoint wrapper expected a length-one tensor container"
                )
            return _official_forward([tensor_container[0]], *args, **kwargs)

        block.forward = types.MethodType(safe_forward, block)
        block._poi_checkpoint_safe_forward = True
