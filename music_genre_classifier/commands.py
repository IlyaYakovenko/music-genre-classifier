"""Command implementations for the project CLI."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from music_genre_classifier.train import train as run_train


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra config from the local configs directory."""
    config_dir = Path.cwd() / "configs"

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir.resolve()),
        job_name="music_genre_classifier",
    ):
        return compose(
            config_name="config",
            overrides=overrides or [],
        )


def show_config(*overrides: str) -> None:
    """Print composed Hydra config."""
    config = load_config(list(overrides))
    print(OmegaConf.to_yaml(config))


def train(*overrides: str) -> None:
    """Train model with PyTorch Lightning."""
    config = load_config(list(overrides))
    run_train(config)
