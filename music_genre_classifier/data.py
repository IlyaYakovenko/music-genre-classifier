"""Data module and dataset for music genre classification."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


AUTHOR_EXCLUDED_FILES = {
    "010449.png",
    "005589.png",
    "004921.png",
    "019511.png",
    "013375.png",
    "024247.png",
    "024156.png",
}


def spectrogram_name_from_audio_name(audio_filename: str) -> str:
    """Convert audio filename from CSV to spectrogram filename."""
    return Path(audio_filename).with_suffix(".png").name


class MusicSpectrogramDataset(Dataset):
    """PyTorch dataset for spectrogram image classification."""

    def __init__(
        self,
        items: list[Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ):
        self.items = items
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one spectrogram image and its label."""
        image_path = self.items[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class MusicGenreDataModule(LightningDataModule):
    """LightningDataModule for music genre spectrograms."""

    def __init__(
        self,
        train_csv: str,
        spectrogram_dir: str,
        genres_csv: str,
        batch_size: int,
        image_size: int,
        num_classes: int,
        num_folds: int,
        fold_id: int,
        seed: int,
        num_workers: int = 0,
        max_items_per_class: int | None = None,
    ):
        super().__init__()

        self.train_csv = Path(train_csv)
        self.spectrogram_dir = Path(spectrogram_dir)
        self.genres_csv = Path(genres_csv)

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_folds = num_folds
        self.fold_id = fold_id
        self.seed = seed
        self.num_workers = num_workers
        self.max_items_per_class = max_items_per_class

        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        self.train_dataset: MusicSpectrogramDataset | None = None
        self.val_dataset: MusicSpectrogramDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create train and validation datasets."""
        items, labels, class_to_idx = self._build_items_and_labels()

        self.class_to_idx = class_to_idx
        self.idx_to_class = {index: name for name, index in class_to_idx.items()}

        train_indices, val_indices = self._build_split(labels)

        train_items = [items[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]

        val_items = [items[index] for index in val_indices]
        val_labels = [labels[index] for index in val_indices]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    size=self.image_size,
                    padding=0,
                    pad_if_needed=True,
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MusicSpectrogramDataset(
            items=train_items,
            labels=train_labels,
            transform=train_transform,
        )
        self.val_dataset = MusicSpectrogramDataset(
            items=val_items,
            labels=val_labels,
            transform=val_transform,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Classes: {self.class_to_idx}")

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule is not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("DataModule is not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _build_items_and_labels(self) -> tuple[list[Path], list[int], dict[str, int]]:
        """Build image paths and integer labels from train.csv and genres.csv."""
        if not self.train_csv.exists():
            raise FileNotFoundError(f"train.csv not found: {self.train_csv}")

        if not self.spectrogram_dir.exists():
            raise FileNotFoundError(
                f"Spectrogram directory not found: {self.spectrogram_dir}"
            )

        if not self.genres_csv.exists():
            raise FileNotFoundError(f"genres.csv not found: {self.genres_csv}")

        train_df = pd.read_csv(self.train_csv)
        genres_df = pd.read_csv(self.genres_csv)

        required_train_columns = {"filename", "genre"}
        missing_train_columns = required_train_columns - set(train_df.columns)
        if missing_train_columns:
            raise ValueError(
                f"Missing columns in train.csv: {sorted(missing_train_columns)}"
            )

        if "genre" not in genres_df.columns:
            raise ValueError("genres.csv must contain 'genre' column.")

        class_names = genres_df["genre"].tolist()
        class_to_idx = {
            class_name: index for index, class_name in enumerate(class_names)
        }

        if len(class_to_idx) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes in genres.csv, "
                f"found {len(class_to_idx)}"
            )

        unknown_genres = set(train_df["genre"].unique()) - set(class_to_idx)
        if unknown_genres:
            raise ValueError(
                f"Genres from train.csv are missing in genres.csv: "
                f"{sorted(unknown_genres)}"
            )

        image_files = sorted(self.spectrogram_dir.glob("*.png"))
        image_names = {image_path.name for image_path in image_files}

        rows = []
        for _, row in train_df.iterrows():
            image_name = spectrogram_name_from_audio_name(row["filename"])

            if image_name in AUTHOR_EXCLUDED_FILES:
                continue

            if image_name not in image_names:
                continue

            rows.append(
                {
                    "image_path": self.spectrogram_dir / image_name,
                    "genre": row["genre"],
                }
            )

        if self.max_items_per_class is not None:
            rows = self._sample_rows_per_class(rows)

        items = [row["image_path"] for row in rows]
        labels = [class_to_idx[row["genre"]] for row in rows]

        print(f"Usable spectrogram images: {len(items)}")
        print(f"Class distribution: {Counter(row['genre'] for row in rows)}")

        if len(set(labels)) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes in selected data, "
                f"found {len(set(labels))}"
            )

        return items, labels, class_to_idx

    def _sample_rows_per_class(self, rows: list[dict]) -> list[dict]:
        """Limit dataset size for quick debug runs."""
        sampled_rows = []
        class_counts: Counter[str] = Counter()

        rows_copy = rows.copy()
        random.Random(self.seed).shuffle(rows_copy)

        for row in rows_copy:
            genre = row["genre"]

            if class_counts[genre] >= self.max_items_per_class:
                continue

            sampled_rows.append(row)
            class_counts[genre] += 1

        return sampled_rows

    def _build_split(self, labels: list[int]) -> tuple[list[int], list[int]]:
        """Build stratified train/validation split."""
        class_distribution = Counter(labels)
        min_class_count = min(class_distribution.values())
        effective_num_folds = min(self.num_folds, min_class_count)

        if effective_num_folds < 2:
            raise ValueError(
                "Not enough samples per class for validation split. "
                "Increase data.max_items_per_class."
            )

        if self.fold_id >= effective_num_folds:
            raise ValueError(
                f"fold_id={self.fold_id} is invalid for "
                f"effective_num_folds={effective_num_folds}"
            )

        splitter = StratifiedKFold(
            n_splits=effective_num_folds,
            shuffle=True,
            random_state=self.seed,
        )

        indices = list(range(len(labels)))

        for current_fold_id, (train_indices, val_indices) in enumerate(
            splitter.split(indices, labels)
        ):
            if current_fold_id == self.fold_id:
                return train_indices.tolist(), val_indices.tolist()

        raise RuntimeError("Could not build stratified split.")
