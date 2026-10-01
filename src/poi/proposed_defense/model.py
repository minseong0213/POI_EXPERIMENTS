"""Structural adapter composed with the official TabPFN v2.5 model forward."""

from __future__ import annotations

import hashlib
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


@dataclass
class ProposedDefenseOutput:
    region_logits: torch.Tensor
    base_logits: torch.Tensor
    query_embeddings: torch.Tensor
    structural: StructuralDefenseOutput


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
    ) -> None:
        super().__init__()
        if len(regions) != 17:
            raise ValueError(f"The experiment requires exactly 17 regions, got {len(regions)}")
        embedding_dim = getattr(base_model, "embedding_dim", None)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise TypeError("base_model must expose the official Architecture.embedding_dim")
        self.base_model = base_model
        self.adapter = StructuralDefenseAdapter(
            codebook, hidden_dim=hidden_dim, numeric_scales=numeric_scales
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
    def codebook(self) -> POIFeatureCodebook:
        return self.adapter.codebook

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
        repaired = structural.repaired_features
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
        return ProposedDefenseOutput(
            region_logits=region_logits,
            base_logits=base_logits,
            query_embeddings=embeddings,
            structural=structural,
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
    models, _, configs, _ = load_model_criterion_config(
        model_path=checkpoint_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.5",
        download_if_not_exists=download_if_missing,
    )
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
    provenance = {
        "tabpfn_version": tabpfn.__version__,
        "tabpfn_source_commit": TABPFN_SOURCE_COMMIT,
        "architecture_module": module_name,
        "architecture_config_name": config_name,
        "checkpoint_path": str(resolved_path) if resolved_path else None,
        "checkpoint_sha256": _sha256(resolved_path) if resolved_path and resolved_path.is_file() else None,
    }
    return model, provenance


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
