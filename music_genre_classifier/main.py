"""Command line entry point for music genre classifier."""

import fire

from music_genre_classifier.commands import show_config, train


def main() -> None:
    """Run command line interface."""
    fire.Fire(
        {
            "show-config": show_config,
            "train": train,
        }
    )