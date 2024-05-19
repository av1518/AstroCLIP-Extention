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

    # count += 1
    # if count == 3:
    #     break

# %%
images = np.concatenate(images, axis=0)
spectra = np.concatenate(spectra, axis=0)
im_embeddings = np.concatenate(im_embeddings, axis=0)
redshifts = np.concatenate(redshifts, axis=0)
source_spec = np.concatenate(source_spec, axis=0)

print(images.shape, im_embeddings.shape, redshifts.shape)
# %%
np.savez(
    "data/embeddings.npz",
    images=images,
    im_embeddings=im_embeddings,
    spectra=spectra,
    redshifts=redshifts,
    source_spec=source_spec,
)
# %%
# Load the embeddings
emb = np.load("data/embeddings.npz")
images = emb["images"]
im_embeddings = emb["im_embeddings"]
spectra = emb["spectra"]
redshifts = emb["redshifts"]
source_spec = emb["source_spec"]


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
ind_query = 4
plt.figure(figsize=[15, 5])
plt.subplot(1, 2, 1)
plt.imshow(images[ind_query])
plt.axis("off")
plt.subplot(1, 2, 2)
plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5)

# normalise the embedding by diving by the L2 norm
image_features = im_embeddings / np.linalg.norm(im_embeddings, axis=-1, keepdims=True)
spectra_features = spectra / np.linalg.norm(spectra, axis=-1, keepdims=True)
# compute the similarity between the query spectrum and all the image embeddings (i.e. dot product)
cross_similarity = (
    spectra_features[ind_query] @ image_features.T
)  # vector of similarity scores
# %% Spectrum query -> top 64 image retrieval
# Sort the indices of the 'similarity' array in descending order
print("Query with spectrum-> image cross similarity")
ind_sorted = np.argsort(cross_similarity)[::-1]
plt.figure(figsize=[10, 10])
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, i * 8 + j + 1)
        plt.imshow(images[ind_sorted[i * 8 + j]])
        plt.axis("off")
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# %% visualise the embeddings
from sklearn.decomposition import PCA
from src.utils import scatter_plot_as_images
from sklearn.neighbors import NearestNeighbors
from matplotlib.pyplot import plot, ylim, title, legend

im_pca = PCA(n_components=4).fit_transform(image_features)
sp_pca = PCA(n_components=4).fit_transform(spectra_features)
scatter_plot_as_images(im_pca, images, nx=30, ny=30)
scatter_plot_as_images(sp_pca, images, nx=30, ny=30)


# %% Nearest Neighbour retrieval
ind_query = 4

spectral_similarity = spectra_features[ind_query] @ spectra_features.T
image_similarity = image_features[ind_query] @ image_features.T
cross_image_similarity = image_features[ind_query] @ spectra_features.T
cross_spectral_similarity = spectra_features[ind_query] @ image_features.T

crop = CenterCrop(96)

plt.figure(figsize=[15, 4])
plt.subplot(121)
plt.imshow(crop(torch.tensor(images[ind_query]).T).T)
plt.title("Queried Image")
plt.subplot(122)
plt.plot(l, source_spec[ind_query], color="red", alpha=0.5)
plt.title("Queried Spectrum")
# plt.ylim(-0, 20)
# %% Query with spectral similarity
inds = np.argsort(spectral_similarity)[::-1]
print("Query with spectral similarity:")
for i in range(4):
    plt.figure(figsize=[15, 4])
    plt.subplot(121)
    plt.imshow(crop(torch.tensor(images[inds[i]]).T).T)
    if i == 0:
        plt.title("Retrieved Image")
    plt.subplot(122)
    plt.plot(l, source_spec[inds[i]], color="red", alpha=0.5, label="Retrieved")
    # plt.ylim(-0, 20)
    plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5, label="Query")

    if i == 0:
        plt.title("Retrieved Spectrum")

    plt.legend()

# %%  Query with image similarity
print("Query with image similarity:")
inds = np.argsort(image_similarity)[::-1]
for i in range(4):
    plt.figure(figsize=[15, 4])
    plt.subplot(121)
    plt.imshow(crop(torch.tensor(images[inds[i]]).T).T)
    if i == 0:
        plt.title("Retrieved Image")
    plt.subplot(122)
    plt.plot(l, source_spec[inds[i]], color="red", alpha=0.5, label="Retrieved")
    # plt.ylim(-0, 20)
    plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5, label="Query")

    if i == 0:
        plt.title("Retrieved Spectrum")

    plt.legend()

# %% Query with cross similarity
print("Query with image->spectrum similarity:")
inds = np.argsort(cross_image_similarity)[::-1]
for i in range(4):
    plt.figure(figsize=[15, 4])
    plt.subplot(121)
    plt.imshow(crop(torch.tensor(images[inds[i]]).T).T)
    if i == 0:
        plt.title("Retrieved Image")
    plt.subplot(122)
    plt.plot(l, source_spec[inds[i]], color="red", alpha=0.5, label="Retrieved")
    # plt.ylim(-0, 20)
    plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5, label="Query")

    if i == 0:
        plt.title("Retrieved Spectrum")

    plt.legend()

# %% Query with cross similarity
print("Query with spectrum->image similarity:")
inds = np.argsort(cross_spectral_similarity)[::-1]
for i in range(4):
    plt.figure(figsize=[15, 4])
    plt.subplot(121)
    plt.imshow(crop(torch.tensor(images[inds[i]]).T).T)
    if i == 0:
        plt.title("Retrieved Image")
    plt.subplot(122)
    plt.plot(l, source_spec[inds[i]], color="red", alpha=0.5, label="Retrieved")
    # plt.ylim(-0, 20)
    plt.plot(l, source_spec[ind_query], color="grey", alpha=0.5, label="Query")

    if i == 0:
        plt.title("Retrieved Spectrum")

    plt.legend()

# %%
