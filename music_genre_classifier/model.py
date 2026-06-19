"""Lightning model for music genre classification."""

from __future__ import annotations

import antialiased_cnns
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import MaxPool, Mish, xse_resnext101
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class LazyMaxBlurPool2d(nn.Module):
    """MaxPool followed by BlurPool with channels inferred on first forward."""

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.blurpool: nn.Module | None = None
        self.blurpool_stride = _as_int_stride(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MaxPool followed by BlurPool."""
        x = self.maxpool(x)

        if self.blurpool is None:
            self.blurpool = antialiased_cnns.BlurPool(
                channels=x.shape[1],
                stride=self.blurpool_stride,
            ).to(device=x.device, dtype=x.dtype)

        return self.blurpool(x)


def _as_int_stride(stride) -> int:
    """Convert MaxPool stride to int for BlurPool."""
    if isinstance(stride, tuple):
        return int(stride[0])
    return int(stride)


def convert_maxpool_to_blurpool(model: nn.Module) -> nn.Module:
    """Replace MaxPool2d layers with LazyMaxBlurPool2d layers."""
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_maxpool_to_blurpool(module)

        if isinstance(module, nn.MaxPool2d):
            model._modules[name] = LazyMaxBlurPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
            )

    return model


def build_xse_resnext_model(
    num_classes: int,
    use_blurpool: bool = True,
) -> nn.Module:
    """Build author-like xse_resnext101 model."""
    model = xse_resnext101(
        n_out=num_classes,
        act_cls=Mish,
        sa=1,
        pool=MaxPool,
        pretrained=False,
    )

    if use_blurpool:
        model = convert_maxpool_to_blurpool(model)

    return model


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross entropy for soft labels, used for MixUp."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


class MusicGenreLightningModule(LightningModule):
    """LightningModule for music genre classification."""

    def __init__(
        self,
        num_classes: int = 19,
        learning_rate: float = 2e-3,
        label_smoothing: float = 0.15,
        mixup_alpha: float = 0.4,
        use_blurpool: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_xse_resnext_model(
            num_classes=num_classes,
            use_blurpool=use_blurpool,
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
        )
        self.train_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
        )
        self.val_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run model forward pass."""
        return self.model(images)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        images, labels = batch

        if self.hparams.mixup_alpha > 0:
            mixed_images, soft_labels = self._mixup(images, labels)
            logits = self(mixed_images)
            loss = soft_cross_entropy(logits, soft_labels)
        else:
            logits = self(images)
            loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=1)

        train_acc = self.train_accuracy(preds, labels)
        train_f1_macro = self.train_f1_macro(preds, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_f1_macro",
            train_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=1)

        val_acc = self.val_accuracy(preds, labels)
        val_f1_macro = self.val_f1_macro(preds, labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_f1_macro",
            val_f1_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    def _mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp to a batch."""
        batch_size = images.size(0)
        num_classes = self.hparams.num_classes

        lam = torch.distributions.Beta(
            self.hparams.mixup_alpha,
            self.hparams.mixup_alpha,
        ).sample().to(images.device)

        permutation = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[permutation]

        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        permuted_one_hot = labels_one_hot[permutation]

        soft_labels = lam * labels_one_hot + (1.0 - lam) * permuted_one_hot

        return mixed_images, soft_labels