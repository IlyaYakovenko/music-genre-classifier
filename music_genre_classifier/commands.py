from __future__ import annotations

from hydra import compose, initialize
from omegaconf import OmegaConf

from music_genre_classifier.inference import infer_audio, infer_images
from music_genre_classifier.train import train as run_train


def _load_config(overrides: tuple[str, ...]):
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=list(overrides))


def show_config(*overrides: str) -> None:
    """Print resolved Hydra config."""
    config = _load_config(overrides)
    print(OmegaConf.to_yaml(config, resolve=True))


def train(*overrides: str) -> None:
    """Run model training."""
    config = _load_config(overrides)
    run_train(config)


def infer_audio_command(*overrides: str) -> None:
    """Run inference for one audio file."""
    config = _load_config(overrides)
    infer_audio(config)


def infer_images_command(*overrides: str) -> None:
    """Run inference for prepared spectrogram images."""
    config = _load_config(overrides)
    infer_images(config)
