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
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomRotation,
    RandomErasing,
    ToTensor,
    CenterCrop,
    ToPILImage,
)
from pl_bolts.models.self_supervised import Moco_v2
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from src.models import OutputExtractor, ExtendedSpender, AstroCLIP, AlternateSpender
from torch.utils.data import Subset, DataLoader


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
# %% Get datasets
CACHE_DIR = "C:\datasets_astroclip"
dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
# %%
dataset.set_format(type="torch", columns=["image", "spectrum"])

# %%
# Create the dataloaders
train_dataloader = DataLoader(
    dataset["train"], batch_size=10, shuffle=True, num_workers=0
)
val_dataloader = DataLoader(
    dataset["test"], batch_size=10, shuffle=False, num_workers=0
)


# %%

sp_layers = [256, 128, 128, 128]
lr = 5e-4

checkpoint_path = "data/weights/resnet50.ckpt"
moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
# extract the backbone model
backbone = moco_model.encoder_q
im_encoder = OutputExtractor(backbone)
sp_encoder = AlternateSpender(sp_layers)

# Setting up image augmentations
image_transforms = Compose(
    [
        # RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        CenterCrop(96),
    ]
)

CLIP = AstroCLIP(
    image_encoder=im_encoder,
    spectrum_encoder=sp_encoder,
    image_transforms=image_transforms,
    lr=lr,
)

CLIP.print_trainable_parameters()
# %%
batch = next(iter(train_dataloader))
spectra = batch["spectrum"]
spectra = spectra.squeeze(-1)  # This removes the last dimension if it's of size 1
CLIP.spectrum_encoder(spectra).shape


# %%
import torch.hub

github = "pmelchior/spender"
sdss, spec_model = torch.hub.load(github, "desi_edr_galaxy")
sp_encoder = spec_model.encoder

# print(sp_encoder)
# print(sp_encoder.mlp)
sp_encoder.mlp = nn.Identity()
# print(sp_encoder.mlp)
print(sp_encoder)


# %%


# %%

print(spec_encoder)
# Now pass it to the model
s = spec_encoder(spectra)
print(s)
