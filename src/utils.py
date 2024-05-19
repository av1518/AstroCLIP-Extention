import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d


# Function borrowed from https://github.com/PolymathicAI/AstroCLIP
def sdss_rgb(imgs, bands, scales=None, m=0.02):
    rgbscales = {
        "u": (2, 1.5),  # 1.0,
        "g": (2, 2.5),
        "r": (1, 1.5),
        "i": (0, 1.0),
        "z": (0, 0.4),  # 0.3
    }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.0) * 1e-6
    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I

    rgb = np.clip(rgb, 0, 1)
    return rgb


def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )


# Code borrowed from https://github.com/georgestein/ssl-legacysurvey
def scatter_plot_as_images(
    z_emb, images, nx=8, ny=8, npix_show=96, iseed=13579, display_image=True
):
    """Sample points from scatter plot and display as their original galaxy image

    Parameters
    ----------
    DDL : class instance
        DecalsDataLoader class instance
    z_emb: array
        (N, 2) array of the galaxies location in some compressed space.
        If second axis has a dimensionality greater than 2 we only consider the leading two components.
    """
    z_emb = z_emb[:, :2]  # keep only first two dimensions

    nplt = nx * ny

    img_full = (
        np.zeros((ny * npix_show, nx * npix_show, 3)) + 255
    )  # , dtype=np.uint8) + 255

    xmin = z_emb[:, 0].min()
    xmax = z_emb[:, 0].max()
    ymin = z_emb[:, 1].min()
    ymax = z_emb[:, 1].max()

    dz_emb = 0.25
    dx_cent = z_emb[:, 0].mean()
    dy_cent = z_emb[:, 1].mean()

    dx_cent = 10.0
    dy_cent = 7.0

    # xmin = dx_cent - dz_emb
    # xmax = dx_cent + dz_emb
    # ymin = dy_cent - dz_emb
    # ymax = dy_cent + dz_emb

    binx = np.linspace(xmin, xmax, nx + 1)
    biny = np.linspace(ymin, ymax, ny + 1)

    ret = binned_statistic_2d(
        z_emb[:, 0],
        z_emb[:, 1],
        z_emb[:, 1],
        "count",
        bins=[binx, biny],
        expand_binnumbers=True,
    )
    z_emb_bins = ret.binnumber.T

    inds_used = []
    inds_lin = np.arange(z_emb.shape[0])

    # First get all indexes that will be used
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            inds = inds_lin[dm]

            np.random.seed(ix * nx + iy + iseed)
            if len(inds) > 0:
                ind_plt = np.random.choice(inds)
                inds_used.append(ind_plt)  # inds_use[ind_plt])

    # load in all images
    iimg = 0

    # Add each image as postage stamp in desired region
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            inds = inds_lin[dm]

            np.random.seed(ix * nx + iy + iseed)
            if len(inds) > 0:
                imi = images[inds[0]][28:-28, 28:-28]
                img_full[
                    iy * npix_show : (iy + 1) * npix_show,
                    ix * npix_show : (ix + 1) * npix_show,
                ] = imi

                iimg += 1

    if display_image:
        plt.figure(figsize=(nx, ny))
        plt.imshow(img_full, origin="lower")  # , interpolation='none')
        plt.axis("off")

    return img_full
