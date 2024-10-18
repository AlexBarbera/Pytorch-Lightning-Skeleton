import lightning
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch


class LightningWrapper(lightning.LightningModule):
    def __init__(self, model):
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        y = self.model(batch)

        # TODO calc loss
        loss = None

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [torch.optim.Adam(self.model.parameters())], []
