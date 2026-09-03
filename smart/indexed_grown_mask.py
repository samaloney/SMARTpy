import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map

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
    rotated_map: Map,
    dilation_radius: u.Quantity[u.arcsec] = 5 * u.arcsec,
    min_area: u.Quantity[u.arcsec**2] = 4500 * u.arcsec**2,
):
    """
    Performing Indexing and Growing of the Mask (hence the name IGM).

    Following Higgins et al. (2011): the un-grown binary detection masks :math:`M_t` and
    (differentially rotated) :math:`M_{t-\\Delta t}` are compared. A feature in :math:`M_t`
    with no counterpart in the grown :math:`M_{t-\\Delta t}`, or smaller than ``min_area``,
    is dropped as transient. The survivors are then grown by ``dilation_radius`` to form
    :math:`M_{f,t}`, and each contiguous feature is given an ascending integer value in
    order of decreasing size.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed magnetogram map from time 't'.
    rotated_map : `~sunpy.map.Map`
        Processed magnetogtam map from time 't - delta_t' differentially rotated to time t.
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
    arcsec_to_pixel = ((current_map.scale[0] + current_map.scale[1]) / 2) ** (-1)
    footprint = disk((np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix))
    min_area_px = np.round(min_area * arcsec_to_pixel**2).to_value(u.pix**2)

    # Un-grown binary detection masks: M_t and the differentially rotated M_{t-dt}.
    current_binary = smooth_los_threshold(current_map, grow=False)[1]
    rotated_binary = smooth_los_threshold(rotated_map, grow=False)[1]

    # A feature in the un-grown current mask survives only if it overlaps the grown,
    # rotated previous-frame mask (so it is not a transient) and is not tiny.
    grown_rotated = ski.morphology.dilation(rotated_binary, footprint)
    current_labels = ski.measure.label(current_binary)
    overlapping = np.unique(current_labels[grown_rotated & (current_labels > 0)])
    big_enough = np.flatnonzero(np.bincount(current_labels.ravel()) >= min_area_px)
    surviving = np.isin(current_labels, np.intersect1d(overlapping, big_enough))

    # Grow the survivors to form M_f,t, then index by decreasing size.
    final_labels = ski.measure.label(ski.morphology.dilation(surviving, footprint))

    regions = ski.measure.regionprops(final_labels)
    region_sizes = [(region.label, region.area) for region in regions]

    sorted_region_sizes = sorted(region_sizes, key=lambda x: x[1], reverse=True)
    sorted_labels = np.zeros_like(final_labels)
    for new_label, (old_label, _) in enumerate(sorted_region_sizes, start=1):
        sorted_labels[final_labels == old_label] = new_label

    return sorted_labels


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
