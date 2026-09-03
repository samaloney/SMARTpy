import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map

from smart.differential_rotation import diff_rotation
from smart.processing import smooth_los_threshold

__all__ = ["index_and_grow_mask", "plot_indexed_grown_mask"]


def prepare_magnetogram(mag: Map):
    r"""
    Prepare magnetogram for and extract some information

    Parameters
    ----------
    mag

    Returns
    -------

    """


def index_and_grow_mask(
    current_map: Map,
    previous_map: Map,
    thresh: u.Quantity[u.Gauss] = 100 * u.Gauss,
    sigma: u.Quantity[u.arcsec] = 10 * u.arcsec,
    dilation_radius: u.Quantity[u.arcsec] = 5 * u.arcsec,
    min_area: u.Quantity[u.arcsec**2] = 4500 * u.arcsec**2,
):
    """
    Performing Indexing and Growing of the Mask (hence the name IGM).

    Following Higgins et al. (2011): both magnetograms are smoothed, noise-thresholded and
    LOS-corrected (`~smart.processing.smooth_los_threshold`); the earlier one is then
    differentially rotated to time 't' and made binary, giving :math:`M_{t-\\Delta t}`,
    while the current one gives :math:`M_t`. Both masks are dilated and compared: a feature
    present in only one frame is transient and is removed from the *un-grown* :math:`M_t`,
    as are features smaller than ``min_area``. The survivors are grown by ``dilation_radius``
    to form :math:`M_{f,t}`, and each contiguous feature is given an ascending integer value
    in order of decreasing size.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed magnetogram map from time 't'.
    previous_map : `~sunpy.map.Map`
        Processed magnetogram map from time 't - delta_t' (not yet rotated; the rotation to
        time 't' is done here, after the smooth/threshold/LOS step, as in the paper).
    thresh : `~astropy.units.Quantity`, optional
        Noise threshold for the binary detection masks, passed to
        `~smart.processing.smooth_los_threshold` (default 100 G).
    sigma : `~astropy.units.Quantity`, optional
        Gaussian smoothing width for the binary detection masks, passed to
        `~smart.processing.smooth_los_threshold` (default 10 arcsec).
    dilation_radius : `~astropy.units.Quantity`, optional
        Radius of the disk for binary dilation (default is 5 arcsec).
    min_area : `~astropy.units.Quantity`, optional
        Minimum on-disk area for a feature to survive transient removal (default
        4500 arcsec**2).

    Returns
    -------
    sorted_labels : `~numpy.ndarray`
        Individual contiguous features are indexed by assigning ascending integer
        values (beginning with one) in order of decreasing feature size.

    """
    diff_in_days = (current_map.date - previous_map.date).to_value("day")
    if 0 < diff_in_days < 1:
        arcsec_to_pixel = ((current_map.scale[0] + current_map.scale[1]) / 2) ** (-1)
        footprint = disk((np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix))
        min_area_px = np.round(min_area * arcsec_to_pixel**2).to_value(u.pix**2)
        thresh_gauss = thresh.to_value(u.Gauss)

        # Current frame: smooth / threshold / LOS-correct -> binary M_t.
        current_binary = smooth_los_threshold(current_map, thresh=thresh, sigma=sigma, grow=False)[1]

        # Previous frame: smooth / threshold / LOS-correct -> rotate to time 't' -> binary.
        previous_smooth = smooth_los_threshold(previous_map, thresh=thresh, sigma=sigma, grow=False)[0]
        rotated_smooth = diff_rotation(current_map, previous_smooth)
        rotated_binary = np.abs(np.nan_to_num(rotated_smooth.data)) >= thresh_gauss

        # Transient removal: dilate both masks, keep grown M_t features that overlap the grown
        # M_{t-dt}, and map that decision back onto the un-grown M_t.
        grown_current = ski.morphology.dilation(current_binary, footprint)
        grown_rotated = ski.morphology.dilation(rotated_binary, footprint)
        grown_labels = ski.measure.label(grown_current)
        persistent = np.unique(grown_labels[grown_rotated & (grown_labels > 0)])
        non_transient = current_binary & np.isin(grown_labels, persistent)

        # Drop features smaller than min_area, then grow the survivors to form M_f,t.
        labels = ski.measure.label(non_transient)
        big_enough = np.flatnonzero(np.bincount(labels.ravel()) >= min_area_px)
        surviving = np.isin(labels, big_enough[big_enough > 0])

        final_labels = ski.measure.label(ski.morphology.dilation(surviving, footprint))

        regions = ski.measure.regionprops(final_labels)
        region_sizes = [(region.label, region.area) for region in regions]

        sorted_region_sizes = sorted(region_sizes, key=lambda x: x[1], reverse=True)
        sorted_labels = np.zeros_like(final_labels)
        for new_label, (old_label, _) in enumerate(sorted_region_sizes, start=1):
            sorted_labels[final_labels == old_label] = new_label

        return sorted_labels
    else:
        raise ValueError(
            f"Difference between current map and previous map: {diff_in_days} is negative or greater than 1 day."
        )


def plot_indexed_grown_mask(current_map: Map, sorted_labels, contours=True, labels=True, figtext=True):
    """
    Plotting the fully processed and segmented magnetogram with labels and AR contours optionally displayed.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed magnetogram map from time 't'.
    sorted_labels : `~sunpy.map.Map`
        Processed magnetogtam map from time 't - delta_t' differentially rotated to time t.
    contours : `bool`, optional
        If True, contours of the detected regions displayed on map (default is True).
    labels : `bool`, optional
        If True, labels with the region numbers will be overlaid on the regions (default is True).
    figtext : `bool`, optional
        If True, figtext with the total number of detected regions is displayed on the map (default is True).

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection=current_map)
    current_map.plot(axes=ax)

    unique_labels = np.unique(sorted_labels)
    unique_labels = unique_labels[unique_labels != 0]

    if contours:
        ax.contour(sorted_labels)

    if labels:
        regions = ski.measure.regionprops(sorted_labels)
        for label, region in zip(unique_labels, regions):
            centroid = region.centroid
            ax.text(
                centroid[1],
                centroid[0],
                str(label),
                color="red",
                fontsize=12,
                weight="bold",
                ha="center",
                va="center",
            )

    if figtext:
        plt.figtext(0.47, 0.2, f"Number of regions = {len(unique_labels)}", color="white")

    plt.show()
