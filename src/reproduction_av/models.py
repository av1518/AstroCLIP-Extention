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
        print("x loaded")
        z_emb = self.backbone(x)  # extract the embdeddings using the backbone model
        return z_emb

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)


class AstroCLIP(L.LightningModule):
    def __init__(self, image_encoder, spectrum_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder

        # freeze all layers in image encoder except for the last MLP layer
        for name, param in self.image_encoder.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        # freeze all layers in spectrum encoder except for the last MLP layer
        for name, param in self.spectrum_encoder.named_parameters():
            if "mlp.9" not in name:
                param.requires_grad = False

        # define the final MLP layer
        in_features = self.image_encoder.backbone.fc.in_features
        self.image_encoder.backbone.fc = nn.Linear(in_features, 128)
        in_features = self.spectrum_encoder.encoder.mlp[9].in_features
        self.spectrum_encoder.encoder.mlp[9] = nn.Linear(in_features, 128)

    def forward(self, x, image=True):
        if image:
            return self.image_encoder(x)
        else:
            return self.spectrum_encoder(x)


class ExtendedMLP(nn.Module):
    def __init__(self, dropout=0.1):
        super(ExtendedMLP, self).__init__()
        self.additional_layers = nn.Sequential(
            nn.Linear(6, 128),  # 1st layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),  # 2nd layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),  # 3rd layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),  # 4th layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),  # 5th layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),  # 6th layer
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.additional_layers(x)
