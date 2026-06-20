from __future__ import annotations

import fire

from music_genre_classifier.commands import (
    infer_audio_command,
    infer_images_command,
    show_config,
    train,
)


def main() -> None:
    fire.Fire(
        {
            "show-config": show_config,
            "train": train,
            "infer-audio": infer_audio_command,
            "infer-images": infer_images_command,
        }
    )


if __name__ == "__main__":
    main()
