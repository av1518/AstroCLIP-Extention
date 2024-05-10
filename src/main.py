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

from src.models import OutputExtractor, ExtendedSpender
from torch.utils.data import Subset, DataLoader


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})
# %% Get datasets
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
# %% make sure this works
for x in train_dataloader:
    print(type(x))
    break
# %% testing the dataloader - DOES NOT WORK
# Fetch a single batch of images
batch = next(iter(train_dataloader))
print("images loaded")
print(type(batch))
print(len(batch))  # 2 because it has two keys: image and spectrum
print(len(batch["image"]))  # 10 because the batch size is 10
# Load the image of the first batch
test_image = batch["image"][0]
print(test_image.shape)

# plot the image
plt.imshow(test_image)
plt.show()

# Load the spectrum of the first batch
test_spectrum = batch["spectrum"][0]
print(test_spectrum.shape)

# plot the spectrum
plt.plot(test_spectrum)
plt.show()

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
checkpoint_path = "data/weights/resnet50.ckpt"
moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
# extract the backbone model
backbone = moco_model.encoder_q
img_model = OutputExtractor(backbone).to("cuda")

num_params = np.sum(np.fromiter((p.numel() for p in img_model.parameters()), int))
print(f"Number of parameters in image model: {num_params:,}")

fc = img_model.backbone.fc
print("Final Layer structure:", fc)
# print number of parameters in the last fully connected layer
fc_num_params = np.sum(np.fromiter((p.numel() for p in fc.parameters()), int))
print(
    f"Number of params in final layer of image embedder (to be trained in CLIP): {fc_num_params:,}"
)

# Freeze all but the last layers of the image encoder
for name, child in img_model.backbone.named_children():
    if name != "fc":
        for param in child.parameters():
            param.requires_grad = False

# %% Test the image model
# pass the random image to the model
img_model.eval()
batch = next(iter(train_dataloader))
im, sp = batch["image"].transpose(1, 3).to("cuda"), batch["spectrum"].squeeze()
# %%
im_embedding = img_model((im, None)).to("cpu")
print(im_embedding.shape)


# %% Load spectrum model
import torch.hub

github = "pmelchior/spender"
# get the spender code and show list of pretrained models
print(torch.hub.list(github))

# print out details for SDSS model from paper II
print(torch.hub.help(github, "desi_edr_galaxy"))

# # load instrument and spectrum model from the hub
sdss, spec_model = torch.hub.load(github, "desi_edr_galaxy")
# print(spec_model)

# # move the model to the GPU
# spec_model = spec_model.to("cuda")

# %%
batch = next(iter(train_dataloader))
spectra = batch["spectrum"]
spectra = spectra.squeeze(-1)  # This removes the last dimension if it's of size 1

# %%
spec_encoder = spec_model.encoder
print(spec_encoder)
# Now pass it to the model
s = spec_encoder(spectra)
print(s)
# %%
from src.models import ExtendedMLP

spec_encoder.extended_mlp = ExtendedMLP()
spec_encoder

# %% Freeze all layers except the last one
# Freeze all layers in the encoder except for the last MLP layer
for name, param in spec_encoder.named_parameters():
    if "extended_mlp" not in name:
        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(spec_encoder)
print(f"Number of trainable parameters in extended_mlp: {num_params}")

# for i, (name, param) in enumerate(spec_encoder.extended_mlp.named_parameters()):
#     if param.requires_grad:
#         print(f"{i}: {name} - {param.numel()} parameters; requires_grad={param.requires_grad}")


# %%
random_batch = next(iter(train_dataloader))
random_spectrum = random_batch["spectrum"].squeeze(-1)

random_spectrum.shape

spec_encoder.eval()
output = spec_encoder(random_spectrum)
print(output.shape)

# %% Load ExtendedSpender
extended_encoder = ExtendedSpender([6, 128, 128])

# pass the random spectrum to the model
sp_embeddings = extended_encoder(random_spectrum).to("cpu")
random_spectrum = random_batch["spectrum"].squeeze(-1)
print(sp_embeddings.shape)

# %%
from src.loss import CLIPLoss

clip = CLIPLoss()
logits, logits_t = clip.get_cosine_matrix(im_embedding, sp_embeddings)

# %%
