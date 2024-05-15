# %% Use trained AstroCLIP model to embed entire dataset
from tqdm import tqdm
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
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from src.utils import dr2_rgb

import matplotlib.pyplot as plt

print("imports done")
# %%
CACHE_DIR = "C:\datasets_astroclip"
dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
dataset.set_format(type="torch", columns=["image", "spectrum", "redshift"])

testdata = DataLoader(dataset["test"], batch_size=512, shuffle=False, num_workers=0)
# %%
checkpoint_path = "data/weights/resnet50.ckpt"
moco_model = Moco_v2.load_from_checkpoint(checkpoint_path=checkpoint_path)
# extract the backbone model
backbone = moco_model.encoder_q
im_encoder = OutputExtractor(backbone)
print("image model loaded")
sp_layers = [6, 128, 256, 256, 256, 256, 128]
sp_encoder = ExtendedSpender(sp_layers=sp_layers)
print("spectrum model loaded")

image_transforms = Compose(
    [
        RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        CenterCrop(96),
    ]
)

CLIP = AstroCLIP.load_from_checkpoint(
    "model_checkpoints/hpc-15-05-epoch=49-step=15450.ckpt",
    image_encoder=im_encoder,
    spectrum_encoder=sp_encoder,
    image_transforms=image_transforms,
)

im_embeddings = []
images = []
redshifts = []
spectra = []
source_spec = []

count = 0
for batch in tqdm(testdata):
    source_spec.append(batch["spectrum"])
    redshifts.append(batch["redshift"])
    im_embeddings.append(
        CLIP(image_transforms(batch["image"].transpose(1, 3).to("cuda")))
        .detach()
        .cpu()
        .numpy()
    )
    spectra.append(
        CLIP(batch["spectrum"].squeeze().to("cuda"), image=False).detach().cpu().numpy()
    )
    processed_images = []
    # Loop through each image in the batch
    for i in batch["image"]:
        image_cpu = i.cpu()
        image_transposed = image_cpu.T

        # Convert the transposed image to RGB format using specified bands
        rgb_image = dr2_rgb(image_transposed, bands=["g", "r", "z"])

        # Clip the RGB values to be within the range [0, 1]
        clipped_image = np.clip(rgb_image, 0, 1)
        processed_images.append(clipped_image)

    stacked_images = np.stack(processed_images, axis=0)
    images.append(stacked_images)

    count += 1
    if count == 3:
        break

# %%
images = np.concatenate(images, axis=0)
spectra = np.concatenate(spectra, axis=0)
im_embeddings = np.concatenate(im_embeddings, axis=0)
redshifts = np.concatenate(redshifts, axis=0)
source_spec = np.concatenate(source_spec, axis=0)

print(images.shape, im_embeddings.shape, redshifts.shape)
# %%Plot the first 64 images
plt.figure(figsize=[10, 10])
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, i * 8 + j + 1)
        plt.imshow(images[i * 8 + j])
        plt.axis("off")
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# %%
l = np.linspace(3586.7408577, 10372.89543574, len(source_spec[0]))
ind_query = 15
plt.figure(figsize=[15, 5])
plt.subplot(1, 2, 1)
plt.imshow(images[ind_query])
plt.axis("off")
plt.subplot(1, 2, 2)
plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5)


# %%

image_features = im_embeddings / np.linalg.norm(im_embeddings, axis=-1, keepdims=True)
spectra_features = spectra / np.linalg.norm(spectra, axis=-1, keepdims=True)
spectral_similarity = spectra_features[ind_query] @ image_features.T
# %%
# Sort the indices of the 'similarity' array in descending order
ind_sorted = np.argsort(spectral_similarity)[::-1]
plt.figure(figsize=[10, 10])
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, i * 8 + j + 1)
        plt.imshow(images[ind_sorted[i * 8 + j]])
        plt.axis("off")
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# %%image similarity
