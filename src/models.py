import lightning as L
import torch, torch.nn as nn
import torch.nn.functional as F

try:
    from loss import CLIPLoss
except ModuleNotFoundError:
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


class AlternateSpender(nn.Module):
    def __init__(self, sp_layers, dropout=0.1):
        super(AlternateSpender, self).__init__()
        _, spec_model = torch.hub.load("pmelchior/spender", "desi_edr_galaxy")
        self.spec_encoder = spec_model.encoder
        self.spec_encoder.mlp = MLP(layer_sizes=sp_layers, dropout=dropout)

        for param in self.spec_encoder.parameters():
            param.requires_grad = False

        # Ensure the new MLP is trainable
        for param in self.spec_encoder.mlp.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.spec_encoder(x)


class AstroCLIP(L.LightningModule):
    """
    @brief A class that defines the unified AstroCLIP model. It loads pretrained
    models, freezes all the layers except the last one in the image encoder, and is
    then trained using CLIP loss.

    This class is responsible for initialising the model, freezing layers, and defining
    the training and validation steps. It also includes methods for configuring optimisers
    and printing trainable parameters.
    """

    def __init__(self, image_encoder, spectrum_encoder, image_transforms, lr=5e-5):
        """
        @brief Constructor for the AstroCLIP class.

        @param image_encoder The image encoder model.
        @param spectrum_encoder The spectrum encoder model.
        @param image_transforms The transformations to be applied to images.
        @param lr Learning rate for the optimiser.
        """
        super().__init__()

        self.image_transforms = image_transforms
        self.image_encoder = image_encoder
        self.lr = lr
        self.save_hyperparameters()

        # Freeze all layers except the last one
        for name, child in self.image_encoder.backbone.named_children():
            if name != "fc":
                for param in child.parameters():
                    param.requires_grad = False

        self.spectrum_encoder = spectrum_encoder

        self.temperature = nn.Parameter(
            torch.tensor(np.log(15.5)), requires_grad=False
        )  # fixed temperature
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

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, on_epoch=True, logger=True)

        self.log(
            "train_loss_no_temperature", loss_no_temperature, on_epoch=True, logger=True
        )
        self.log("train_loss", loss)
        self.log("temperature", self.temperature)
        return loss

    def validation_step(self, batch, batch_idx):
        im = batch["image"].transpose(1, 3)
        im = self.image_transforms(im)
        sp = batch["spectrum"].squeeze(-1)
        im_emb = self.image_encoder((im, None))
        sp_emb = self.spectrum_encoder(sp)

        val_loss = self.loss(im_emb, sp_emb, self.temperature)
        self.log("validation_loss", val_loss)
        val_loss_no_temperature = self.loss(im_emb, sp_emb, 1.0)
        self.log("validation_loss_no_temperature", val_loss_no_temperature)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.2
        )  # self.params fetches all the trainable parameters of the model
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def print_trainable_parameters(self):
        print("Trainable Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Shape: {param.shape}")
