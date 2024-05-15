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

print("imports done")
# %%
CACHE_DIR = "C:\datasets_astroclip"
dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
dataset.set_format(type="torch", columns=["image", "spectrum", "redshift"])

testdata = DataLoader(dataset["test"], batch_size=512, shuffle=False, num_workers=0)

embeddings = []
images = []
spectra = []
source_spec = []

CLIP = AstroCLIP.load_from_checkpoint(
    "model_checkpoints/hpc-15-05-epoch=49-step=15450.cpkt"
)
