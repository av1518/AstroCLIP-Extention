import lightning as L
import torch, torch.nn as nn, torch.nn.functional as F

# from fillm.run.model import * Commented out by Andreas
import torch.nn.functional as F


class OutputExtractor(L.LightningModule):
    """
    A PyTorch Lightning module for extracting outputs from a neural network model. It
    is a general-purpose class that can be used with any neural network model (referred to as the backbone).
    Its primary function is to pass input data through the given network and extract the output, which could
    be feature embeddings, predictions, or any other kind of output produced by the network.

    Attributes:
        backbone (torch.nn.Module): The neural network model from which outputs are to be extracted.

    Methods:
        forward(batch): Passes data through the network and returns the output.
        predict(batch, batch_idx, dataloader_idx): Wrapper for the forward method, used for making predictions.
    """

    def __init__(self, backbone: torch.nn.Module):
        super(OutputExtractor, self).__init__()
        self.backbone = backbone
        self.backbone.eval()  # Set the model to evaluation mode by default

    def forward(self, batch):
        x, _ = batch
        print('x loaded')
        z_emb = self.backbone(x)  # extract the embdeddings using the backbone model
        return z_emb

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)
