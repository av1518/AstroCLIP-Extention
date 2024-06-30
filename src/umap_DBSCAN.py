# %%
import numpy as np
from astropy.table import Table, join
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage import gaussian_filter1d as smooth
from tqdm import tqdm

# Path to the PROVABGS table (edit this path as needed)
PROVABGS_PATH = "C:\datasets_astroclip\BGS_ANY_full.provabgs.sv3.v0.hdf5"


print("imports done")
# %%
# Load the embeddings
print("loading embeddings")
emb = np.load("data/embeddings-main.npz")
source_images = emb["images"]
im_embeddings = emb["im_embeddings"]
spectra = emb["spectra"]
redshifts = emb["redshifts"]
source_spec = emb["source_spec"]
targetid = emb["targetid"]
print("embeddings loaded")
# %%

print("loading PROVABGS table")
provabgs = Table.read(PROVABGS_PATH)

# join the provabgs table using the targetid
embedding_table = Table(
    {
        "targetid": targetid,
        "image_embedding": im_embeddings,
        "spectrum_embedding": spectra,
        "source_image": source_images,
        "source_spec": source_spec,
    }
)
provabgs = join(provabgs, embedding_table, keys_left="TARGETID", keys_right="targetid")

# remove invalid values
provabgs = provabgs[
    (provabgs["PROVABGS_LOGMSTAR_BF"] > 0)
    * (provabgs["MAG_G"] > 0)
    * (provabgs["MAG_R"] > 0)
    * (provabgs["MAG_Z"] > 0)
]

print("PROVABGS table loaded and filtered, joined with embeddings")
# set random seed
np.random.seed(25101999)
# randomise the order
provabgs = provabgs[np.random.permutation(len(provabgs))]
# %%
# Create a UMAP of the spectra embeddings
umap_sp = umap.UMAP()
emb_map_sp = umap_sp.fit_transform(provabgs["spectrum_embedding"])

# Plot the UMAP
plt.figure(figsize=(20, 8), dpi=500)

# Plot the UMAP of spectrum embeddings with redshift as the color
plt.subplot(1, 2, 1)
sc1 = plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=provabgs["Z_HP"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

# turn off values on the x and y axis
plt.xticks([])
plt.yticks([])
# plt.title("UMAP of Spectrum Embeddings with Redshift")
cbar1 = plt.colorbar(sc1, ax=plt.gca(), orientation="vertical", aspect=30)
cbar1.set_label(r"Redshift $Z_{HP}$", fontsize=14)
cbar1.ax.tick_params(labelsize=15)

plt.subplot(1, 2, 2)
sc2 = plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="magma",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

plt.xticks([])
plt.yticks([])

# plt.title("UMAP of Spectrum Embeddings with logMstar")
cbar2 = plt.colorbar(sc2, ax=plt.gca(), orientation="vertical", aspect=30)
cbar2.set_label(r"Stellar Mass $\log M_{*}$", fontsize=14)
cbar2.ax.tick_params(labelsize=15)

plt.tick_params(axis="both", which="major", labelsize=15)
plt.tight_layout()

# plt.savefig("figures/umap_spectra_heatmap.png", dpi=500, bbox_inches="tight")

plt.show()

# %% UMAP of the image embeddings
umap_im = umap.UMAP()
emb_map_im = umap_im.fit_transform(provabgs["image_embedding"])

# Plot the UMAP
plt.figure(figsize=(20, 8), dpi=500)

# Plot the UMAP of image embeddings with redshift as the color
plt.subplot(1, 2, 1)
sc1 = plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=provabgs["Z_HP"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

# turn off values on the x and y axis
plt.xticks([])
plt.yticks([])
cbar1 = plt.colorbar(sc1, ax=plt.gca(), orientation="vertical", aspect=30)
cbar1.set_label(r"Redshift $Z_{HP}$", fontsize=14)
cbar1.ax.tick_params(labelsize=15)

# Plot the UMAP of image embeddings with log(M*) as the color
plt.subplot(1, 2, 2)
sc2 = plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="magma",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

plt.xticks([])
plt.yticks([])

cbar2 = plt.colorbar(sc2, ax=plt.gca(), orientation="vertical", aspect=30)
cbar2.set_label(r"Stellar Mass $\log M_{*}$", fontsize=14)
cbar2.ax.tick_params(labelsize=15)

plt.tick_params(axis="both", which="major", labelsize=15)
plt.tight_layout()

# plt.savefig("figures/umap_image_heatmap.png", dpi=500, bbox_inches="tight")

plt.show()


# %% UMAP of both image and spectra embeddings
umap_both = umap.UMAP()
emb_map_both = umap_both.fit_transform(
    np.concatenate(
        [provabgs["image_embedding"], provabgs["spectrum_embedding"]], axis=1
    )
)

# Plot the UMAP
plt.figure(figsize=(20, 8), dpi=500)

# Plot the UMAP of the image and spectra embeddings with redshift as the color
plt.subplot(1, 2, 1)
sc1 = plt.scatter(
    emb_map_both[:, 0],
    emb_map_both[:, 1],
    c=provabgs["Z_HP"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

# turn off values on the x and y axis
plt.xticks([])
plt.yticks([])
cbar1 = plt.colorbar(sc1, ax=plt.gca(), orientation="vertical", aspect=30)
cbar1.set_label(r"Redshift $Z_{HP}$", fontsize=14)
cbar1.ax.tick_params(labelsize=15)

# Plot the UMAP of the image and spectra embeddings with log(M*) as the color
plt.subplot(1, 2, 2)
sc2 = plt.scatter(
    emb_map_both[:, 0],
    emb_map_both[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="magma",
    s=2,
    alpha=0.5,
)
plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)

plt.xticks([])
plt.yticks([])

cbar2 = plt.colorbar(sc2, ax=plt.gca(), orientation="vertical", aspect=30)
cbar2.set_label(r"Stellar Mass $\log M_{*}$", fontsize=14)
cbar2.ax.tick_params(labelsize=15)

plt.tick_params(axis="both", which="major", labelsize=15)
plt.tight_layout()

# plt.savefig("figures/umap_both_heatmap.png", dpi=500, bbox_inches="tight")

plt.show()


# %% Clustering the UMAP of image embeddings


# Determine the optimal number of clusters using silhouette score
sil_scores = []
k_values = range(2, 10)  # Evaluate K from 2 to 10

for k in tqdm(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(emb_map_im)
    sil_score = silhouette_score(emb_map_im, cluster_labels)
    sil_scores.append(sil_score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, sil_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for K-Means Clustering")
plt.show()

# Choose the optimal number of clusters based on the silhouette score
optimal_k = k_values[np.argmax(sil_scores)]
print(f"Optimal number of clusters: {optimal_k}")
# %%
n_clusters = 6

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(emb_map_im)

# Plot the UMAP with cluster assignments
plt.figure(figsize=(20, 8))

# Plot the UMAP of spectrum embeddings with clusters
plt.subplot(1, 2, 1)
plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=cluster_labels,
    cmap="tab10",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Spectrum Embeddings with Clusters")

# Plot the UMAP of image embeddings with clusters
plt.subplot(1, 2, 2)
plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=cluster_labels,
    cmap="tab10",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Image Embeddings with Clusters")

plt.tight_layout()
plt.show()
# %%
from sklearn.cluster import DBSCAN

# Apply DBSCAN on the UMAP-transformed spectrum embeddings
dbscan_sp = DBSCAN(eps=0.22, min_samples=5)  # Adjust eps and min_samples as needed
dbscan_labels_sp = dbscan_sp.fit_predict(emb_map_sp)

# Apply DBSCAN on the UMAP-transformed image embeddings
dbscan_im = DBSCAN(eps=0.22, min_samples=5)  # Adjust eps and min_samples as needed
dbscan_labels_im = dbscan_im.fit_predict(emb_map_im)

# Print the number of clusters found (excluding noise points labeled as -1)
n_clusters_sp = len(set(dbscan_labels_sp)) - (1 if -1 in dbscan_labels_sp else 0)
n_clusters_im = len(set(dbscan_labels_im)) - (1 if -1 in dbscan_labels_im else 0)
print(f"Number of clusters found in spectrum embeddings: {n_clusters_sp}")
print(f"Number of clusters found in image embeddings: {n_clusters_im}")


# %%
# Plot the UMAP with DBSCAN cluster assignments
plt.figure(figsize=(15, 8))

unique_labels_sp = set(dbscan_labels_sp)
colors_sp = [plt.cm.tab10(each) for each in np.linspace(0, 1, len(unique_labels_sp))]

plt.subplot(1, 2, 1)
for k, col in zip(unique_labels_sp, colors_sp):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = dbscan_labels_sp == k
    xy = emb_map_sp[class_member_mask]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=2,
        color=tuple(col),
        label=f"Cluster {k + 1}" if k != -1 else "Noise",
    )

plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)
plt.xticks([])
plt.yticks([])
# plt.title("UMAP of Spectrum Embeddings with DBSCAN Clusters", fontsize=18)
plt.legend(markerscale=8, loc="best", fontsize=15)

unique_labels_im = set(dbscan_labels_im)
colors_im = [plt.cm.tab10(each) for each in np.linspace(0, 1, len(unique_labels_im))]

plt.subplot(1, 2, 2)
for k, col in zip(unique_labels_im, colors_im):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = dbscan_labels_im == k
    xy = emb_map_im[class_member_mask]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=2,
        color=tuple(col),
        label=f"Cluster {k + 1}" if k != -1 else "Noise",
    )

plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)
plt.xticks([])
plt.yticks([])
# plt.title("UMAP of Image Embeddings with DBSCAN Clusters", fontsize=18)
plt.legend(markerscale=8, loc="best", fontsize=15)

plt.tight_layout()
# #plt.savefig("figures/umap_dbscan_clusters.png", dpi=500, bbox_inches="tight")
plt.show()


# %%


def get_cluster_entries(cluster_label, dbscan_labels, provabgs):
    """
    Returns the entries of a specified cluster from the DBSCAN results.

    Parameters:
    - cluster_label: The label of the cluster to extract.
    - dbscan_labels: The array of cluster labels from DBSCAN.
    - provabgs: The astropy table containing the data.

    Returns:
    - cluster_entries: The subset of the table corresponding to the specified cluster.
    """
    cluster_indices = np.where(dbscan_labels == cluster_label)[0]
    return cluster_indices


def plot_cluster_entries(cluster_indices, provabgs, num_samples=5):
    """
    Plots the images and spectra of the selected cluster entries.

    Parameters:
    - cluster_indices: The indices of the entries in the specified cluster.
    - provabgs: The astropy table containing the data.
    - num_samples: The number of samples to plot.
    """
    num_samples = min(num_samples, len(cluster_indices))
    selected_indices = np.random.choice(cluster_indices, num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))

    l = np.linspace(3586.7408577, 10372.89543574, source_spec.shape[1])

    for i, idx in enumerate(selected_indices):
        entry = provabgs[idx]
        image = entry["source_image"]
        spectrum = entry["source_spec"][:, 0]

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].axis("off")

        # Plot the spectrum
        axes[i, 1].plot(l, spectrum, alpha=0.5, label="Raw Spectrum", color="gray")
        axes[i, 1].plot(l, smooth(spectrum, 2))
        axes[i, 1].set_xlabel(r"$\lambda [\AA]$")
        axes[i, 1].set_ylabel("Flux")
        axes[i, 1].legend()

    # set the cluster label as the title
    fig.suptitle(f"Cluster {cluster_label}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_cluster_images(cluster_label, dbscan_labels, provabgs, num_images=8):
    """
    Plots the images of the selected cluster entries in rows of 4.

    Parameters:
    - cluster_label: The label of the cluster to extract.
    - dbscan_labels: The array of cluster labels from DBSCAN.
    - provabgs: The astropy table containing the data.
    - num_images: The number of images to plot.
    """
    cluster_indices = np.where(dbscan_labels == cluster_label)[0]

    # Ensure we don't try to plot more images than we have
    num_images = min(num_images, len(cluster_indices))
    selected_indices = np.random.choice(cluster_indices, num_images, replace=False)

    # Determine the number of rows needed to display 4 images per row
    num_rows = (num_images + 3) // 4  # Adding 3 ensures rounding up for any remainder

    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))

    for i, idx in enumerate(selected_indices):
        entry = provabgs[idx]
        image = entry["source_image"]

        row = i // 4
        col = i % 4

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].axis("off")

    for j in range(i + 1, num_rows * 4):
        row = j // 4
        col = j % 4
        axes[row, col].axis("off")

    # fig.suptitle(f"DBScan Cluster {cluster_label + 1}", fontsize=16)
    plt.tight_layout()
    plt.show()


# %%
# Inspect islands in image UMAP
cluster_label = 2
cluster_indices = get_cluster_entries(cluster_label - 1, dbscan_labels_im, provabgs)
plot_cluster_entries(cluster_indices + 1, provabgs, num_samples=5)
plot_cluster_images(cluster_label - 1, dbscan_labels_im, provabgs, num_images=16)
print(f"Number of entries in cluster {cluster_label}: {len(cluster_indices)}")
# %%
# Inspect islands in spectrum UMAP
# Inspect islands in image UMAP
cluster_label = 3  # Specify the cluster you want to inspect
cluster_indices = get_cluster_entries(cluster_label - 1, dbscan_labels_sp, provabgs)
plot_cluster_entries(cluster_indices, provabgs, num_samples=12)
plot_cluster_images(cluster_label - 1, dbscan_labels_sp, provabgs, num_images=16)
print(f"Number of entries in cluster {cluster_label}: {len(cluster_indices)}")

# %%
# Plot the UMAP with DBSCAN cluster assignments for image embeddings
plt.figure(figsize=(15, 15))

unique_labels_im = set(dbscan_labels_im)
colors_im = [plt.cm.tab10(each) for each in np.linspace(0, 1, len(unique_labels_im))]

for k, col in zip(unique_labels_im, colors_im):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = dbscan_labels_im == k
    xy = emb_map_im[class_member_mask]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=8,
        color=tuple(col),
        label=f"Cluster {k + 1}" if k != -1 else "Noise",
    )

plt.xlabel("UMAP 1", fontsize=18)
plt.ylabel("UMAP 2", fontsize=18)
plt.xticks([])
plt.yticks([])
# plt.title("UMAP of Image Embeddings with DBSCAN Clusters", fontsize=18)
plt.legend(markerscale=8, loc="best", fontsize=15)

plt.tight_layout()
# plt.savefig("figures/umap_of_images_dbscan_alone.png", dpi=500, bbox_inches="tight")
plt.show()

# %%


def plot_noise_images(dbscan_labels, provabgs, num_images=8):
    """
    Plots the images of the noise entries detected by DBSCAN in rows of 4.

    Parameters:
    - dbscan_labels: The array of cluster labels from DBSCAN.
    - provabgs: The astropy table containing the data.
    - num_images: The number of images to plot.
    """
    noise_indices = np.where(dbscan_labels == -1)[0]

    # Ensure we don't try to plot more images than we have
    num_images = min(num_images, len(noise_indices))
    selected_indices = np.random.choice(noise_indices, num_images, replace=False)

    num_rows = (num_images + 3) // 4  # Adding 3 ensures rounding up for any remainder

    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))

    for i, idx in enumerate(selected_indices):
        entry = provabgs[idx]
        image = entry["source_image"]

        row = i // 4
        col = i % 4

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].axis("off")

    for j in range(i + 1, num_rows * 4):
        row = j // 4
        col = j % 4
        axes[row, col].axis("off")

    fig.suptitle("Noise Images", fontsize=16)
    plt.tight_layout()
    plt.show()


# Example usage
plot_noise_images(dbscan_labels_im, provabgs, num_images=16)
