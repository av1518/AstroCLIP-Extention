FROM continuumio/miniconda3

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY hpc_env.yml .
RUN conda env create -f hpc_env.yml
RUN echo "conda activate astroclip_3" >> ~/.bashrc

SHELL ["conda", "run", "-n", "astroclip_3", "/bin/bash", "-c"]
