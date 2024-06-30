


## Environment Installation
Because of OS-specific dependencies, we provide three different environment files: 
- `win_env.yml` was used on a Windows OS for all downstream tasks and testing local training.
- `hpc_env.yml` expects a linux OS, and was used for training the model on the HPC cluster.
- `no_builds_env.yml` is not OS-dependent (this should be used on a MAC OS), but was found to lead into potential package dependecy issues.

To install the necessary Conda environment from these dependencies, run eg:
```bash
conda env create -f win_env.yml
```

Once the environment is created, activate it using:

```bash
conda activate astroclip
```

## Script Contents
Here is quick description of each script and folder in `src/`:

| Folder/File               | Description          |
|---------------------------|----------------------|
| `datasets_files/`            | Contains the scripts needed to download the datasets (prepared by the original work's authors)                     |
| `downstream_notebooks/`      | Jypiter notebooks containing the downstream tasks and figures used in the report (for ease of use)                     |
| `run_outputs/`               | Contains the shell scripts used in training on HPC                     |
| `downstream.py`             |   Downstream task script for query search, zero-shot predictions                   |
| `loss.py`                   |  Contains the InfoNCE loss function for contrastive learning                    |
| `models.py`                 | Contains classes for our pre-trained embedders and the unified AstroCLIP model                     |
| `plot_training.py`          |  Plots scripts for plotting the training loss, validation loss and learning rate during HPC training                    |
| `train_hpc.py`              |   Script for training AstroCLIP on HPC cluster                   |
| `train_local.py`            |   Script for training AstroCLIP locally (used for testing, needs ~ 20 hours on NVIDIA 16GB RTX3060 GPU)                   |
| `umap_DBSCAN.py`            | Extension script: UMAP projection and DBSCAN clustering                     |
| `umap_kmeans.py`            | Extension script: UMAP projection and KMeans clustering                     |
| `utils.py`                  | Contains 2 utility functions borrowed from Stein et. al (2021) (as does the original work) to ensure proper reproducibility.                      |



## Astroclip Checkpoints and Embeddings
Some of our model checkpoints can be found in the `model_checkpoints` folder. The checkpoint used in obtaining the report figures is `model_checkpoints/epoch=49-[256, 128, 128, 128], lr=0.0005, hpc-alt-1.ckpt`. If you prefer to download our embeddings directly, we host them as HuggingFace Dataset in `.npz` format, which can be easily downloaded from https://huggingface.co/datasets/AVrikkis/astroclip_reproduction/tree/main. They are 12.3 GB in size. The scripts expect the embeddins to be blaced in `data/` directory.

## Data Access
We authors of the original work provide the dataset as a HuggingFace dataset,  which can be accessed directly with the `src/dataset_files` in this directory. It can be downloaded onto the `data/` directory using:

```python
from datasets import load_dataset

CACHE_DIR = "data/"
dataset = load_dataset("src/datasets_files/legacy_survey.py", cache_dir=CACHE_DIR)
```
This downloads about 60 GB of data. Both `src/train_local` and `src/train_hpc` contain this relevant code which will initiate the downloading process if the dataset is not in the repository.

To get the PRObabilistic Value-Added Bright Galaxy Survey (PROVABGS) Catalogue data used in the downstream tasks, you can follow the instruction on their website https://data.desi.lbl.gov/doc/releases/edr/vac/provabgs/ . The necessary file used in the script is `BGS_ANY_full.provabgs.sv3.v0.hdf5`


## Docker Instructions
All the packages used and their versions were included in the file hpc_env.yml. This file can be used to replicate the Conda environment used during HPC training of this project.

To run the Python scripts inside Docker, first build the image
```
docker build -t astroclip .
```

This would generate an image called `astroclip`. To deploy and run the container, run the following command:

```
docker run --rm --gpus all -ti astroclip
```
This would start the process inside the container. `--gpus` all tag is needed to enable GPU access inside the container, which is necessary for running of most scripts. To enable GPU access inside containers, NVIDIA Container Toolkit needs to be installed. On devices with GPU's, run the following commands to install NVIDIA Container Toolkit if it isn't already available:

1) Get the `.gpg` key set up:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
2) Install the toolkit:
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
After running the above commands, NVIDIA Container Toolkit should be availble for use.

It is important to ensure there is sufficient storage before installing. For the same reason, we also advise that the dataset should be downloaded locally first, then mounted into the Docker container with appropriate storage settings instead of downloading it inside the container. For testing purposes, it's sufficient to only mount a subset of all data into the container.