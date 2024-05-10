import lightning as L
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.functional as F
from src.loss import CLIPLoss
import numpy as np
from torch.optim import lr_scheduler


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
        # print("x loaded")
        z_emb = self.backbone(x)  # extract the embdeddings using the backbone model
        return z_emb

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)


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


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.PReLU(num_parameters=layer_sizes[i + 1]))
            layers.append(nn.Dropout(dropout))

        self.additional_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.additional_layers(x)


class ExtendedSpender(nn.Module):
    def __init__(self, sp_layers, dropout=0.1):
        super(ExtendedSpender, self).__init__()
        ssds, spec_model = torch.hub.load("pmelchior/spender", "desi_edr_galaxy")
        self.spec_encoder = spec_model.encoder
        self.extended_mlp = MLP(layer_sizes=sp_layers, dropout=dropout)

        # Freeze all layers in the Spender encoder (except the extended MLP layer)
        for name, param in self.spec_encoder.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.spec_encoder(x)  # Get the encoding from the original model
        x = self.extended_mlp(x)  # Pass it through the extended MLP
        return x


class AstroCLIP(L.LightningModule):
    """
    A class that loads the pretrained models, freezes all the layers except the last one in image encoder and is then trained using
    CLIP loss
    """

    def __init__(self, image_encoder, spectrum_encoder, image_transforms):
        super().__init__()

        self.image_transforms = image_transforms
        self.image_encoder = image_encoder

        # Freeze all layers except the last one
        for name, child in self.image_encoder.backbone.named_children():
            if name != "fc":
                for param in child.parameters():
                    param.requires_grad = False

        self.spectrum_encoder = spectrum_encoder

        self.temperature = nn.Parameter(torch.tensor(np.log(15.5)))  # fixed temperature
        self.loss = CLIPLoss()

    def forward(self, x, image=True):
        if image:
            return self.image_encoder((x, None))
        else:
            return self.spectrum_encoder(x)

    def training_step(self, batch):
        im = batch["image"].transpose(1, 3)
        im = self.image_transforms(im)

        spec = batch["spectrum"].squeeze(-1)
        im_emb = self.image_encoder((im, None))
        sp_emb = self.spectrum_encoder(spec)

        loss = self.loss(im_emb, sp_emb, self.temperature)
        loss_no_temperature = self.loss(im_emb, sp_emb, 1.0)
        self.log("train_loss_no_temperature", loss_no_temperature)
        self.log("train_loss", loss)
        self.log("temperature", self.temperature)
        return loss

    def validation_step(self, batch, batch_idx):
        im = batch["image"].transpose(1, 3)
        sp = batch["spectrum"].squeeze(-1)
        im_emb = self.image_encoder((im, None))
        sp_emb = self.spectrum_encoder(sp)

        val_loss = self.loss(im_emb, sp_emb, self.temperature)
        self.log("validation_loss", val_loss)
        val_loss_no_temperature = self.loss(im_emb, sp_emb, 1.0)
        self.log("validation_loss_no_temperature", val_loss_no_temperature)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=5e-5, weight_decay=0.2
        )  # self.params fetches all the trainable parameters of the model
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
