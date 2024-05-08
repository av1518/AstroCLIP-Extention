# %%
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from src.tutorial_helpers import (
    load_model_from_ckpt,
    forward,
    slice,
    fnc,
    dr2_rgb,
    scatter_plot_as_images,
)
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset

from pl_bolts.models.self_supervised import Moco_v2
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from src.models import OutputExtractor, ExtendedSpender, AstroCLIP
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import (
    Compose,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomRotation,
    RandomErasing,
    ToTensor,
    CenterCrop,
    ToPILImage,
    InterpolationMode,
)

import wandb
from pytorch_lightning.loggers import WandbLogger


wandb.login()


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
# %% Get datasets


def main():

    CACHE_DIR = "C:\datasets_astroclip"
    dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
    dataset.set_format(type="torch", columns=["image", "spectrum"])
    # %%
    # Create the dataloaders
    train_dataloader = DataLoader(
        dataset["train"], batch_size=10, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        dataset["test"], batch_size=10, shuffle=False, num_workers=0
    )

    # %% Load image and spectrum models
    checkpoint_path = "data/weights/resnet50.ckpt"
    moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # extract the backbone model
    backbone = moco_model.encoder_q
    im_encoder = OutputExtractor(backbone)
    sp_encoder = ExtendedSpender()

    # %%
    # Setting up image augmentations
    image_transforms = Compose(
        [
            RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            CenterCrop(96),
        ]
    )

    wandb_logger = WandbLogger(log_model="all", project="astroclip", name="clip")

    model = AstroCLIP(im_encoder, sp_encoder, image_transforms)

    trainer = L.Trainer(
        callbacks=[
            ModelCheckpoint(
                every_n_epochs=1,
            )
        ],
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
