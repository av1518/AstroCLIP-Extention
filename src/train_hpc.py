# %
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from datasets import load_dataset
from pl_bolts.models.self_supervised import Moco_v2
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
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

print("imports done")

wandb.login()
# %% Get datasets

print("starting main")


def main():
    sp_layers = [6, 128, 256, 256, 256, 256, 128]
    lr = 1e-5

    torch.set_float32_matmul_precision("medium")

    CACHE_DIR = "data/"
    print("Loading dataset")
    dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
    dataset.set_format(type="torch", columns=["image", "spectrum"])
    print("Dataset loaded")

    # Create the dataloaders
    print("Creating dataloaders")
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=512,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset["test"],
        batch_size=512,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    print("Dataloaders created")

    #  Load image and spectrum models
    print("load")
    checkpoint_path = "data/weights/resnet50.ckpt"
    moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # extract the backbone model
    backbone = moco_model.encoder_q
    im_encoder = OutputExtractor(backbone)
    sp_encoder = ExtendedSpender(sp_layers=sp_layers)

    # Setting up image augmentations
    image_transforms = Compose(
        [
            RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            CenterCrop(96),
        ]
    )

    wandb_logger = WandbLogger(
        log_model="all", project="astroclip", name=f"{sp_layers}, lr={lr}"
    )

    model = AstroCLIP(im_encoder, sp_encoder, image_transforms, lr=lr)

    trainer = L.Trainer(
        max_epochs=2,
        callbacks=[
            ModelCheckpoint(
                every_n_epochs=1,
            )
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

# %%
