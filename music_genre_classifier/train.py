"""Lightning training entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from music_genre_classifier.data import MusicGenreDataModule
from music_genre_classifier.model import MusicGenreLightningModule


def train(cfg: DictConfig) -> None:
    """Run Lightning training from Hydra config."""
    seed_everything(cfg.seed, workers=True)

    fold_output_dir = Path(cfg.train.output_dir) / f"fold_{cfg.data.fold_id}"
    checkpoint_dir = fold_output_dir / "checkpoints"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _save_resolved_config(cfg, fold_output_dir)

    datamodule = MusicGenreDataModule(
        train_csv=cfg.data.train_csv,
        spectrogram_dir=cfg.data.spectrogram_dir,
        genres_csv=cfg.data.genres_csv,
        batch_size=cfg.train.batch_size,
        image_size=cfg.data.image_size,
        num_classes=cfg.data.num_classes,
        num_folds=cfg.data.num_folds,
        fold_id=cfg.data.fold_id,
        seed=cfg.seed,
        num_workers=cfg.data.num_workers,
        max_items_per_class=cfg.data.max_items_per_class,
    )

    model = MusicGenreLightningModule(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.train.learning_rate,
        label_smoothing=cfg.train.label_smoothing,
        mixup_alpha=cfg.train.mixup_alpha,
        use_blurpool=cfg.model.use_blurpool,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        run_name=cfg.logging.run_name,
        log_model=cfg.logging.log_model,
    )

    hyperparameters = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    hyperparameters["git_commit_id"] = _get_git_commit_id()
    mlflow_logger.log_hyperparams(hyperparameters)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=cfg.train.save_every_n_train_steps,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
        ],
        logger=mlflow_logger,
        default_root_dir=fold_output_dir,
        log_every_n_steps=1,
    )

    ckpt_path = None
    last_checkpoint_path = checkpoint_dir / "last.ckpt"
    if cfg.train.resume and last_checkpoint_path.exists():
        ckpt_path = str(last_checkpoint_path)
        print(f"Resuming from checkpoint: {ckpt_path}")

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    _log_artifacts_to_mlflow(
        mlflow_logger=mlflow_logger,
        fold_output_dir=fold_output_dir,
    )

    print(f"Saved training artifacts to: {fold_output_dir}")


def _save_resolved_config(cfg: DictConfig, output_dir: Path) -> None:
    """Save full resolved Hydra config near model artifacts."""
    config_path = output_dir / "config.yaml"
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_path.write_text(config_yaml, encoding="utf-8")


def _log_artifacts_to_mlflow(
    mlflow_logger: MLFlowLogger,
    fold_output_dir: Path,
) -> None:
    """Log important local files as MLflow artifacts."""
    run_id = mlflow_logger.run_id

    config_path = fold_output_dir / "config.yaml"
    if config_path.exists():
        mlflow_logger.experiment.log_artifact(
            run_id=run_id,
            local_path=str(config_path),
        )


def _get_git_commit_id() -> str:
    """Return current git commit id if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


def _flatten_dict(
    dictionary: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested dict for MLflow params."""
    flattened = {}

    for key, value in dictionary.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            flattened.update(
                _flatten_dict(
                    value,
                    parent_key=full_key,
                    separator=separator,
                )
            )
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[full_key] = value
        else:
            flattened[full_key] = str(value)

    return flattened