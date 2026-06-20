from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from music_genre_classifier.model import MusicGenreLightningModule


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SpectrogramImageDataset(Dataset):
    """Dataset for already prepared spectrogram images."""

    def __init__(self, input_dir: str | Path, image_size: int) -> None:
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.image_paths = sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No image files were found in: {self.input_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        relative_path = image_path.relative_to(self.input_dir).as_posix()
        return tensor, relative_path


def infer_audio(cfg: DictConfig) -> None:
    """Run inference for one audio file."""
    if _as_bool(cfg.dvc.enabled) and _as_bool(cfg.dvc.pull_model_before_infer):
        _pull_model_with_dvc(cfg)

    audio_path = Path(cfg.inference.audio_path)
    output_json = Path(cfg.inference.output_json)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(cfg.inference.device))
    class_names = _load_class_names(
        class_names_path=cfg.inference.class_names_path,
        num_classes=int(cfg.model.num_classes),
    )

    checkpoint_paths = _resolve_checkpoint_paths(cfg)
    models = [
        _load_model(checkpoint_path=checkpoint_path, device=device)
        for checkpoint_path in checkpoint_paths
    ]

    image_tensor = _audio_file_to_model_input(audio_path=audio_path, cfg=cfg).to(device)

    probabilities = _predict_probabilities(
        models=models,
        batch_images=image_tensor.unsqueeze(0),
    )[0]

    result = _format_single_prediction(
        filename=audio_path.name,
        probabilities=probabilities,
        class_names=class_names,
        top_k=int(cfg.inference.top_k),
    )

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved prediction: {output_json}")


def infer_images(cfg: DictConfig) -> None:
    """Run inference for a folder with already prepared spectrogram images."""
    if _as_bool(cfg.dvc.enabled) and _as_bool(cfg.dvc.pull_model_before_infer):
        _pull_model_with_dvc(cfg)

    input_dir = Path(cfg.inference.input_dir)
    output_csv = Path(cfg.inference.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(cfg.inference.device))
    class_names = _load_class_names(
        class_names_path=cfg.inference.class_names_path,
        num_classes=int(cfg.model.num_classes),
    )

    checkpoint_paths = _resolve_checkpoint_paths(cfg)
    models = [
        _load_model(checkpoint_path=checkpoint_path, device=device)
        for checkpoint_path in checkpoint_paths
    ]

    dataset = SpectrogramImageDataset(
        input_dir=input_dir,
        image_size=int(cfg.data.image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.inference.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )

    rows: list[dict[str, Any]] = []

    print(f"Input directory: {input_dir}")
    print(f"Images found: {len(dataset)}")
    print(f"Checkpoints used: {[str(path) for path in checkpoint_paths]}")
    print(f"Device: {device}")

    with torch.inference_mode():
        for batch_images, batch_filenames in dataloader:
            batch_images = batch_images.to(device)
            probabilities = _predict_probabilities(
                models=models,
                batch_images=batch_images,
            )

            for index, filename in enumerate(batch_filenames):
                rows.append(
                    _format_single_prediction(
                        filename=filename,
                        probabilities=probabilities[index],
                        class_names=class_names,
                        top_k=int(cfg.inference.top_k),
                    )
                )

    _write_predictions(output_csv=output_csv, rows=rows)
    print(f"Saved predictions: {output_csv}")


def _audio_file_to_model_input(audio_path: Path, cfg: DictConfig) -> Tensor:
    waveform, sample_rate = _load_audio_with_soundfile(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = int(cfg.inference.sample_rate)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    chunk_samples = int(float(cfg.inference.chunk_seconds) * target_sample_rate)
    if waveform.shape[1] < chunk_samples:
        padding = chunk_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :chunk_samples]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=int(cfg.inference.n_fft),
        hop_length=int(cfg.inference.hop_length),
        n_mels=int(cfg.inference.n_mels),
        f_min=float(cfg.inference.f_min),
        f_max=float(cfg.inference.f_max),
        power=2.0,
    )

    mel = mel_transform(waveform)
    log_mel = torch.log1p(mel)

    log_mel = log_mel.squeeze(0)
    log_mel = log_mel - log_mel.min()
    log_mel = log_mel / (log_mel.max() + 1e-8)

    image = log_mel.unsqueeze(0).repeat(3, 1, 1)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(int(cfg.data.image_size), int(cfg.data.image_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return image.float()


def _load_audio_with_soundfile(audio_path: Path) -> tuple[Tensor, int]:
    try:
        audio, sample_rate = sf.read(
            str(audio_path),
            always_2d=True,
            dtype="float32",
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to read audio file. For the most reliable local test on Windows, "
            "use WAV. MP3 decoding may depend on the available libsndfile/codec build."
        ) from error

    waveform = torch.from_numpy(audio).transpose(0, 1)
    return waveform, int(sample_rate)


def _predict_probabilities(
    models: list[MusicGenreLightningModule],
    batch_images: Tensor,
) -> Tensor:
    probabilities_per_model = []

    with torch.inference_mode():
        for model in models:
            logits = model(batch_images)
            probabilities = F.softmax(logits, dim=1)
            probabilities_per_model.append(probabilities)

    return torch.stack(probabilities_per_model, dim=0).mean(dim=0)


def _format_single_prediction(
    filename: str,
    probabilities: Tensor,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    top_probabilities, top_indices = probabilities.topk(
        k=min(top_k, len(class_names)),
        dim=0,
    )

    predicted_index = int(top_indices[0].item())

    result: dict[str, Any] = {
        "filename": filename,
        "predicted_class_id": predicted_index,
        "predicted_genre": class_names[predicted_index],
        "confidence": float(top_probabilities[0].item()),
    }

    for rank in range(top_indices.shape[0]):
        class_index = int(top_indices[rank].item())
        result[f"top_{rank + 1}_class_id"] = class_index
        result[f"top_{rank + 1}_genre"] = class_names[class_index]
        result[f"top_{rank + 1}_probability"] = float(top_probabilities[rank].item())

    return result


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
    checkpoint_paths_from_config = OmegaConf.select(cfg, "inference.checkpoint_paths")

    if checkpoint_paths_from_config:
        checkpoint_paths = [Path(path) for path in checkpoint_paths_from_config]
    else:
        checkpoint_paths = [Path(cfg.inference.checkpoint_path)]

    missing_paths = [path for path in checkpoint_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Some checkpoint paths do not exist: "
            + ", ".join(str(path) for path in missing_paths)
        )

    return checkpoint_paths


def _load_class_names(class_names_path: str | Path, num_classes: int) -> list[str]:
    path = Path(class_names_path)

    if not path.exists():
        raise FileNotFoundError(f"Class names file does not exist: {path}")

    if path.suffix.lower() == ".json":
        class_names = _load_class_names_from_json(path)
    else:
        class_names = _load_class_names_from_csv(path)

    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)} from {path}"
        )

    return class_names


def _load_class_names_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        if all(str(key).isdigit() for key in data):
            return [str(data[str(index)]) for index in range(len(data))]

        if all(isinstance(value, int) for value in data.values()):
            return [
                class_name
                for class_name, _ in sorted(data.items(), key=lambda item: item[1])
            ]

    raise ValueError(f"Unsupported JSON class names format: {path}")


def _load_class_names_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(file)
            rows = list(reader)

            if not rows:
                raise ValueError(f"Empty class names CSV: {path}")

            fieldnames = reader.fieldnames or []
            preferred_columns = [
                "genre",
                "genre_name",
                "class_name",
                "name",
                "title",
                "label",
            ]

            selected_column = None
            for column in preferred_columns:
                if column in fieldnames:
                    selected_column = column
                    break

            if selected_column is None:
                selected_column = fieldnames[0]

            return [str(row[selected_column]) for row in rows]

        reader = csv.reader(file)
        return [str(row[0]) for row in reader if row]


def _write_predictions(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No predictions were produced")

    fieldnames = list(rows[0].keys())

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_config)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")

    return device


def _pull_model_with_dvc(cfg: DictConfig) -> None:
    model_remote = str(cfg.dvc.models_remote)

    command = [
        "dvc",
        "pull",
        "-r",
        model_remote,
        "saved_model.dvc",
    ]

    print(f"Pulling model with DVC: {' '.join(command)}")
    subprocess.run(command, check=True)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}
