# %%
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from tutorial_helpers import (
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

# from fillm.run.model import * Commented out by Andreas
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
# %% Datasets
CACHE_DIR = "C:\datasets_astroclip"
dataset = load_dataset("datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
dataset.set_format(type="torch", columns=["image", "spectrum"])

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=512, shuffle=True, num_workers=10
)
val_dataloader = torch.utils.data.DataLoader(
    dataset["test"], batch_size=512, shuffle=False, num_workers=10
)

# Define Transforms to be used during training
image_transforms = Compose(
    [
        # ToRGB(),
        # ToTensor(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        CenterCrop(96),
    ]
)
# %%
data = train_dataloader.dataset.data
shape = train_dataloader.dataset.data.shape
# %% Visualise some examples
index = 4

raw_image_ex = image_transforms(dataset["train"][index]["image"].T).T
rgb_image_ex = dr2_rgb(raw_image_ex.T, bands=["g", "r", "z"])
spectrum_ex = dataset["train"][index]["spectrum"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(raw_image_ex)
ax[0].set_title("Image")
ax[1].imshow(rgb_image_ex)
ax[1].set_title("RGB Image")
ax[2].plot(spectrum_ex)
ax[2].set_title("Spectrum")

plt.show()
# %%
