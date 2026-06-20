from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from music_genre_classifier.model import MusicGenreLightningModule


class EnsembleONNXModule(nn.Module):
    """Two-fold ensemble exported as one ONNX graph.

    The module returns averaged class probabilities, not raw logits.
    """

    def __init__(self, models: list[MusicGenreLightningModule]) -> None:
        super().__init__()

        if len(models) < 1:
            raise ValueError("At least one model is required for export")

        self.models = nn.ModuleList(models)

    def forward(self, batch_images: Tensor) -> Tensor:
        probabilities = [F.softmax(model(batch_images), dim=1) for model in self.models]

        return torch.stack(probabilities, dim=0).mean(dim=0)


def export_onnx(cfg: DictConfig) -> None:
    """Export one or several trained folds as a single ONNX model."""
    if _as_bool(cfg.dvc.enabled) and _as_bool(cfg.export.pull_model_before_export):
        _pull_model_with_dvc(cfg)

    onnx_path = Path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(cfg.export.device))
    checkpoint_paths = _resolve_checkpoint_paths(cfg)

    print(f"Exporting checkpoints: {[str(path) for path in checkpoint_paths]}")
    print(f"Device: {device}")
    print(f"ONNX path: {onnx_path}")

    lightning_models = [
        _load_model(checkpoint_path=checkpoint_path, device=device)
        for checkpoint_path in checkpoint_paths
    ]

    ensemble = EnsembleONNXModule(lightning_models)
    ensemble.eval()
    ensemble.to(device)

    dummy_input = torch.randn(
        int(cfg.export.batch_size),
        int(cfg.export.input_channels),
        int(cfg.data.image_size),
        int(cfg.data.image_size),
        device=device,
    )

    with torch.inference_mode():
        pytorch_output = ensemble(dummy_input).detach().cpu().numpy()

    torch.onnx.export(
        ensemble,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=int(cfg.export.opset_version),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
    )

    print(f"Saved ONNX model: {onnx_path}")

    metadata = {
        "type": "two_fold_ensemble" if len(checkpoint_paths) == 2 else "ensemble",
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "input_name": "input",
        "output_name": "probabilities",
        "input_shape": [
            "batch_size",
            int(cfg.export.input_channels),
            int(cfg.data.image_size),
            int(cfg.data.image_size),
        ],
        "output_shape": ["batch_size", int(cfg.model.num_classes)],
        "opset_version": int(cfg.export.opset_version),
        "returns": "averaged_softmax_probabilities",
    }

    metadata_path = onnx_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved ONNX metadata: {metadata_path}")

    if _as_bool(cfg.export.verify):
        _verify_onnx_model(
            onnx_path=onnx_path,
            dummy_input=dummy_input.detach().cpu().numpy(),
            pytorch_output=pytorch_output,
            tolerance=float(cfg.export.verify_tolerance),
        )


def _verify_onnx_model(
    onnx_path: Path,
    dummy_input: np.ndarray,
    pytorch_output: np.ndarray,
    tolerance: float,
) -> None:
    print("Checking ONNX model structure...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print("Running ONNX Runtime verification...")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["probabilities"],
        {"input": dummy_input},
    )[0]

    max_abs_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(pytorch_output - onnx_output)))

    print(f"ONNX verification max_abs_diff: {max_abs_diff:.8f}")
    print(f"ONNX verification mean_abs_diff: {mean_abs_diff:.8f}")

    if max_abs_diff > tolerance:
        raise RuntimeError(
            f"ONNX output differs from PyTorch output too much: "
            f"max_abs_diff={max_abs_diff}, tolerance={tolerance}"
        )

    print("ONNX verification passed.")


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> MusicGenreLightningModule:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model = MusicGenreLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def _resolve_checkpoint_paths(cfg: DictConfig) -> list[Path]:
    checkpoint_paths_from_config = OmegaConf.select(cfg, "export.checkpoint_paths")

    if checkpoint_paths_from_config:
        checkpoint_paths = [Path(path) for path in checkpoint_paths_from_config]
    else:
        checkpoint_paths = [Path(cfg.export.checkpoint_path)]

    missing_paths = [path for path in checkpoint_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Some checkpoint paths do not exist: "
            + ", ".join(str(path) for path in missing_paths)
        )

    return checkpoint_paths


def _pull_model_with_dvc(cfg: DictConfig) -> None:
    model_remote = str(cfg.dvc.models_remote)
    model_target = str(cfg.dvc.model_target)

    command = [
        "dvc",
        "pull",
        "-r",
        model_remote,
        model_target,
    ]

    print(f"Pulling model with DVC: {' '.join(command)}")
    subprocess.run(command, check=True)


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_config)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")

    return device


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}
