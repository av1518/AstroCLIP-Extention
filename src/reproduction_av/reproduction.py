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
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from models import OutputExtractor
from torch.utils.data import Subset


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
# %% Datasets
CACHE_DIR = "C:\datasets_astroclip"
dataset = load_dataset("datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
dataset.set_format(type="torch", columns=["image", "spectrum"])

# Create the dataloaders
train_dataloader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=10, shuffle=True, num_workers=0
)
val_dataloader = torch.utils.data.DataLoader(
    dataset["test"], batch_size=10, shuffle=False, num_workers=0
)

# %% testing the dataloader - DOES NOT WORK
# Fetch a single batch of images
batch = next(iter(train_dataloader))
print("images loaded")
print(type(batch))
print(len(batch))
# %%
## Fetch a few batches from the DataLoader
for i, batch in enumerate(train_dataloader):
    # Access the 'spectrum' part of the batch
    spectra = batch["spectrum"]

    # Iterate over each spectrum in the batch
    for j, spectrum in enumerate(spectra):
        print(f"Batch {i}, Spectrum {j}: Shape = {spectrum.shape}")

    # Optional: Break the loop after a few batches to avoid too much output
    if i == 2:
        break


# %%
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
index = 1

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
# %% Load image embdedder

checkpoint_path = "../../data/weights/resnet50.ckpt"
moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
# extract the backbone model
backbone = moco_model.encoder_q
img_model = OutputExtractor(backbone).to("cuda")

num_params = np.sum(np.fromiter((p.numel() for p in img_model.parameters()), int))
print(f"Number of parameters in image model: {num_params:,}")

fc = img_model.backbone.fc
print(fc)
# print number of parameters in the last fully connected layer
num_params = np.sum(np.fromiter((p.numel() for p in fc.parameters()), int))

# %% Loading spectrum model
import torch.hub

github = "pmelchior/spender"

# get the spender code and show list of pretrained models
print(torch.hub.list(github))


# print out details for SDSS model from paper II
print(torch.hub.help(github, "desi_edr_galaxy"))

# # load instrument and spectrum model from the hub
sdss, model = torch.hub.load(github, "desi_edr_galaxy")
print(model)

# %%
# Fetch a single batch of images
batch = next(iter(train_dataloader))
print("images loaded")
print(type(batch))
print(len(batch))

batch = next(iter(train_dataloader))
spectra = batch["spectrum"]
spectra = spectra.squeeze(-1)  # This removes the last dimension if it's of size 1

# Now pass it to the model
s = model.encode(spectra)
print(s)
# %% Modify the model : Freeze all layers except the last one
# Step 1: Modify the Final MLP Layer
# Get the number of in_features from the last Linear layer
in_features = model.encoder.mlp[9].in_features

# Replace the last Linear layer with a new one having out_features=128
model.encoder.mlp[9] = torch.nn.Linear(in_features, 128)
print(model.encoder)

# Step 2: Freeze Other Layers
# Freeze all layers in the encoder except for the last MLP layer
for name, param in model.encoder.named_parameters():
    if "mlp.9" not in name:  # Check if the parameter is not part of the last MLP layer
        param.requires_grad = False

# %%
mlp = model.encoder.mlp

total_mlp_parameters = 0
for i, layer in enumerate(mlp):
    if isinstance(layer, torch.nn.Linear):
        # Calculate the number of parameters in a linear layer
        num_parameters = layer.in_features * layer.out_features
        if layer.bias is not None:
            num_parameters += layer.out_features  # Add bias parameters if present
        print(f"Layer {i} - Linear: {num_parameters} parameters")
        total_mlp_parameters += num_parameters

print(f"Total parameters in the MLP: {total_mlp_parameters:,}")

# %%
