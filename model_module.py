import torch
from torch import nn
from pytorch_lightning import LightningModule

class CustomModel(LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Linear(input_dim, 1)  # Простая модель для предсказания среднего

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)