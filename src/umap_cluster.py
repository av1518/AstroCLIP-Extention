# %%
import numpy as np
from astropy.table import Table, join
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
provabgs = Table.read("C:\datasets_astroclip\BGS_ANY_full.provabgs.sv3.v0.hdf5")

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


# set random seed
np.random.seed(25101999)
# randomise the order
provabgs = provabgs[np.random.permutation(len(provabgs))]
# %%
print(len(provabgs))
# %%
# Create a UMAP of the spectra embeddings
umap_sp = umap.UMAP()
emb_map_sp = umap_sp.fit_transform(provabgs["spectrum_embedding"])
# %%
# plot the umap
plt.figure(figsize=(20, 8))

# Plot the UMAP of spectrum embeddings
plt.subplot(1, 2, 1)
plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.colorbar(label=r"$logM*$")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the spectrum embeddings")

# Plot the UMAP of spectrum embeddings with redshift as the color
plt.subplot(1, 2, 2)
plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=provabgs["Z_HP"],
    cmap="plasma",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="Redshift [units]")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the spectrum embeddings")

plt.tight_layout()
plt.show()


# %% Umap of the image embeddings
umap_im = umap.UMAP()
emb_map_im = umap_im.fit_transform(provabgs["image_embedding"])

# plot the umap
plt.figure(figsize=(20, 8))

# Plot the UMAP of image embeddings with log(M*) as the color
plt.subplot(1, 2, 1)
plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="log(M*)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the image embeddings")

# Plot the UMAP of image embeddings with redshift as the color
plt.subplot(1, 2, 2)
plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=provabgs["Z_HP"],
    cmap="plasma",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="redshift")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the image embeddings")

plt.tight_layout()
plt.show()


# %% Umap of both image and spectra embeddings
umap_both = umap.UMAP()
emb_map_both = umap_both.fit_transform(
    np.concatenate(
        [provabgs["image_embedding"], provabgs["spectrum_embedding"]], axis=1
    )
)
# plot the umap
plt.figure(figsize=(20, 8))

# Plot the UMAP of the image and spectra embeddings with log(M*) as the color
plt.subplot(1, 2, 1)
plt.scatter(
    emb_map_both[:, 0],
    emb_map_both[:, 1],
    c=provabgs["PROVABGS_LOGMSTAR_BF"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="log(M*)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of both image and spectrum embeddings")

# Plot the UMAP of the image and spectra embeddings with redshift as the color
plt.subplot(1, 2, 2)
plt.scatter(
    emb_map_both[:, 0],
    emb_map_both[:, 1],
    c=provabgs["Z_HP"],
    cmap="plasma",
    s=2,
    alpha=0.5,
)
plt.colorbar(label="redshift")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the image and spectra embeddings")

plt.tight_layout()
plt.show()


# %%
def silhouette_analysis(embeddings, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(
            f"For n_clusters = {n_clusters}, the average silhouette_score is {silhouette_avg}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
    plt.title("Silhouette Analysis")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

    return silhouette_scores


max_clusters = 10
silhouette_scores = silhouette_analysis(provabgs["spectrum_embedding"], max_clusters)

# %% Cluster the spectra embeddings using KMeans

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=25101999)
kmeans_clusters_sp = kmeans.fit_predict(provabgs["spectrum_embedding"])

# # Add the cluster labels to table
provabgs["kmeans_cluster_sp"] = kmeans_clusters_sp

plt.figure(figsize=(10, 8))
plt.scatter(
    emb_map_sp[:, 0],
    emb_map_sp[:, 1],
    c=provabgs["kmeans_cluster_sp"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)

# Extract unique labels
unique_labels = np.unique(provabgs["kmeans_cluster_sp"])

# Correct way to map cluster labels to colors using normalization
norm = plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
unique_colors = [plt.cm.viridis(norm(label)) for label in unique_labels]

for i, color in zip(unique_labels, unique_colors):
    plt.scatter([], [], color=color, label=f"Cluster {i}")


plt.legend()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("K-Means Clusters in UMAP Space (Spectra Embeddings)")
plt.show()

# Get the silhouette score for the clustering
silhouette_score(provabgs["spectrum_embedding"], kmeans_clusters_sp)


# %%
max_clusters = 5
silhouette_scores = silhouette_analysis(provabgs["image_embedding"], max_clusters)

# plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
plt.title("Silhouette Analysis for Image Embeddings")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()


# %% Cluster the image embeddings using KMeans
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=25101999)
kmeans_clusters_im = kmeans.fit_predict(provabgs["image_embedding"])

# # Add the cluster labels to table
provabgs["kmeans_clusters_im"] = kmeans_clusters_im

# Create the scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    emb_map_im[:, 0],
    emb_map_im[:, 1],
    c=provabgs["kmeans_clusters_im"],
    cmap="viridis",
    s=2,
    alpha=0.5,
)


unique_labels = np.unique(kmeans_clusters_im)
unique_colors = [scatter.cmap(scat) for scat in scatter.norm(unique_labels)]
for i, color in zip(unique_labels, unique_colors):
    plt.scatter([], [], color=color, label=f"Cluster {i}")

plt.legend()
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("K-Means Clusters in UMAP Space (Images)")
plt.show()

# Get the silhouette score for the clustering
silhouette_score(provabgs["image_embedding"], kmeans_clusters_im)
# %% For spectra clustering
# Get the unique cluster labels and their counts
unique_clusters, counts = np.unique(provabgs["kmeans_cluster_sp"], return_counts=True)

for cluster, count in zip(unique_clusters, counts):
    print(f"Spectrum Cluster {cluster}: {count} entries")

unique_clusters, counts = np.unique(provabgs["kmeans_clusters_im"], return_counts=True)

for cluster, count in zip(unique_clusters, counts):
    print(f"Image Cluster {cluster}: {count} entries")


# %%
# Parameters
cluster_number = 0
num_images_to_display = 30  # Set how many images you want to display
cluster_type = "kmeans_clusters_im"  # Set the cluster type

# Step 1: Filter to get entries in the specified cluster
cluster_indices = np.where(provabgs[cluster_type] == cluster_number)[0]

# Select a subset of indices (you can use random choice here if preferred)
selected_indices = (
    cluster_indices[:num_images_to_display]
    if len(cluster_indices) > num_images_to_display
    else cluster_indices
)

# Extract images using selected indices
cluster_images = provabgs["source_image"][selected_indices]

# Step 2: Determine how many images you want to display and set up the plot accordingly
cols = 5
rows = (len(cluster_images) + cols - 1) // cols

# Step 3: Create a figure and axes for plotting the images
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 3 * rows))
ax = ax.flatten()

# Step 4: Plot each image
for i, img in enumerate(cluster_images):
    ax[i].imshow(img)
    ax[i].axis("off")

# Turn off axes for any unused subplots
for j in range(i + 1, len(ax)):
    ax[j].axis("off")

fig.suptitle(f"Images from Cluster {cluster_number}", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# %%
def plot_histograms(provabgs, indices, bin_count=30):
    """
    Plots histograms for the stellar mass and redshift of the selected entries.

    Parameters:
        provabgs (Table): The main data table with all galaxy information.
        indices (array-like): Indices of the galaxies to consider for the histograms.
        bin_count (int): Number of bins to use in the histograms.
    """
    # Extract the stellar masses and redshifts for the selected indices
    selected_masses = provabgs["PROVABGS_LOGMSTAR_BF"][indices]
    selected_redshifts = provabgs["Z_HP"][indices]

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of stellar masses
    ax[0].hist(selected_masses, bins=bin_count, color="skyblue", edgecolor="black")
    ax[0].set_title("Histogram of Stellar Masses")
    ax[0].set_xlabel("$\log(M_*/M_\odot)$")
    ax[0].set_ylabel("Frequency")

    # Histogram of redshifts
    ax[1].hist(selected_redshifts, bins=bin_count, color="salmon", edgecolor="black")
    ax[1].set_title("Histogram of Redshifts")
    ax[1].set_xlabel("Redshift (z)")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


cluster_number = 4
indices = np.where(provabgs["kmeans_cluster_sp"] == cluster_number)[0]

plot_histograms(provabgs, indices)


# %%
def plot_histograms(provabgs, cluster_indices_dict, bin_count=50):
    """
    Plots histograms for the stellar mass and redshift of multiple clusters on the same plots.

    Parameters:
        provabgs (Table): The main data table with all galaxy information.
        cluster_indices_dict (dict): Dictionary where keys are cluster identifiers and values are indices of galaxies in each cluster.
        bin_count (int): Number of bins to use in the histograms.
    """
    # Generate colors for each cluster
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_indices_dict)))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    for index, (cluster_label, indices) in enumerate(cluster_indices_dict.items()):
        # Use index to access color
        color = colors[index]

        # Extract the stellar masses and redshifts for the selected indices
        selected_masses = provabgs["PROVABGS_LOGMSTAR_BF"][indices]
        selected_redshifts = provabgs["Z_HP"][indices]

        # Histogram of stellar masses
        ax[0].hist(
            selected_masses,
            bins=bin_count,
            color=color,
            alpha=0.5,
            edgecolor="black",
            label=f"Cluster {cluster_label}",
        )

        # Histogram of redshifts
        ax[1].hist(
            selected_redshifts,
            bins=bin_count,
            color=color,
            alpha=0.5,
            edgecolor="black",
            label=f"Cluster {cluster_label}",
        )

    # Set titles, labels and legends
    ax[0].set_title("Histogram of Stellar Masses")
    ax[0].set_xlabel("$\log(M_*/M_\odot)$")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()

    ax[1].set_title("Histogram of Redshifts")
    ax[1].set_xlabel("Redshift (z)")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


cluster_numbers = [0, 1, 2, 3, 4, 5]
cluster_indices_dict = {
    num: np.where(provabgs["kmeans_cluster_sp"] == num)[0] for num in cluster_numbers
}

# Call the function with the dictionary of cluster indices
plot_histograms(provabgs, cluster_indices_dict)
# %%


def plot_histograms_by_cluster(provabgs, cluster_indices_dict, bin_count=30):
    """
    Plots histograms for the stellar mass and redshift of each cluster on separate subplots with uniform axis limits.

    Parameters:
        provabgs (Table): The main data table with all galaxy information.
        cluster_indices_dict (dict): Dictionary where keys are cluster identifiers and values are indices of galaxies in each cluster.
        bin_count (int): Number of bins to use in the histograms.
    """
    num_clusters = len(cluster_indices_dict)
    fig, axes = plt.subplots(
        num_clusters, 2, figsize=(12, 6 * num_clusters)
    )  # Each cluster gets a row with 2 columns
    colors = plt.cm.viridis(
        np.linspace(0, 1, num_clusters)
    )  # Generate colors for each cluster

    # Initialize lists to find global min and max
    all_masses = []
    all_redshifts = []

    for indices in cluster_indices_dict.values():
        all_masses.append(provabgs["PROVABGS_LOGMSTAR_BF"][indices])
        all_redshifts.append(provabgs["Z_HP"][indices])

    # Determine global min and max for consistent axis scales
    min_mass = np.min(np.hstack(all_masses))
    max_mass = np.max(np.hstack(all_masses))
    min_redshift = np.min(np.hstack(all_redshifts))
    max_redshift = np.max(np.hstack(all_redshifts))

    if num_clusters == 1:  # Handle the case of a single cluster for proper indexing
        axes = np.array([axes])

    for index, ((cluster_label, indices), color) in enumerate(
        zip(cluster_indices_dict.items(), colors)
    ):
        # Extract the stellar masses and redshifts for the selected indices
        selected_masses = provabgs["PROVABGS_LOGMSTAR_BF"][indices]
        selected_redshifts = provabgs["Z_HP"][indices]

        # Plotting the histogram of stellar masses
        ax_mass = axes[index, 0]
        ax_mass.hist(
            selected_masses, bins=bin_count, color=color, alpha=0.5, edgecolor="black"
        )
        ax_mass.set_title(f"Cluster {cluster_label} - Stellar Masses")
        ax_mass.set_xlabel("$\log(M_*/M_\odot)$")
        ax_mass.set_ylabel("Frequency")
        ax_mass.set_xlim(min_mass, max_mass)

        # Plotting the histogram of redshifts
        ax_z = axes[index, 1]
        ax_z.hist(
            selected_redshifts,
            bins=bin_count,
            color=color,
            alpha=0.5,
            edgecolor="black",
        )
        ax_z.set_title(f"Cluster {cluster_label} - Redshifts")
        ax_z.set_xlabel("Redshift (z)")
        ax_z.set_ylabel("Frequency")
        ax_z.set_xlim(min_redshift, max_redshift)

    plt.tight_layout()
    plt.show()


cluster_numbers = [0, 1, 2, 4, 5]
cluster_indices_dict = {
    num: np.where(provabgs["kmeans_cluster_sp"] == num)[0] for num in cluster_numbers
}

plot_histograms_by_cluster(provabgs, cluster_indices_dict)

# %%


def plot_boxplots_by_cluster(provabgs, cluster_indices_dict):
    """
    Plots box plots for the stellar mass and redshift of each cluster.

    Parameters:
        provabgs (Table): The main data table with all galaxy information.
        cluster_indices_dict (dict): Dictionary where keys are cluster identifiers and values are indices of galaxies in each cluster.
    """
    num_clusters = len(cluster_indices_dict)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6)
    )  # One row with 2 columns for mass and redshift

    # Prepare data for box plots
    mass_data = [
        provabgs["PROVABGS_LOGMSTAR_BF"][indices]
        for indices in cluster_indices_dict.values()
    ]
    redshift_data = [
        provabgs["Z_HP"][indices] for indices in cluster_indices_dict.values()
    ]

    # Box plot for stellar masses
    axes[0].boxplot(mass_data)
    axes[0].set_title("Box Plot of Stellar Masses")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("$\log(M_*/M_\odot)$")
    axes[0].set_xticklabels(
        [f"Cluster {k}" for k in cluster_indices_dict.keys()], rotation=45
    )

    # Box plot for redshifts
    axes[1].boxplot(redshift_data)
    axes[1].set_title("Box Plot of Redshifts")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Redshift (z)")
    axes[1].set_xticklabels(
        [f"Cluster {k}" for k in cluster_indices_dict.keys()], rotation=45
    )

    plt.tight_layout()
    plt.show()


cluster_numbers = [0, 1, 2, 3, 4, 5]
cluster_indices_dict = {
    num: np.where(provabgs["kmeans_cluster_sp"] == num)[0] for num in cluster_numbers
}

# Call the function with the dictionary of cluster indices
print("boxplot by cluster for spectrum embeddings")
plot_boxplots_by_cluster(provabgs, cluster_indices_dict)

# %%
