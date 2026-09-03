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


def index_and_grow_mask(current_map: Map, rotated_map: Map, dilation_radius: u.Quantity[u.arcsec] = 5 * u.arcsec):
    """
    Performing Indexing and Growing of the Mask (hence the name IGM).

    Transient features are removed by comparing the detection mask at time 't' with the
    detection mask differentially rotated to time 't': features in the current mask with no
    counterpart in the rotated mask are dropped. The surviving detection is grown, and each
    contiguous feature is assigned an ascending integer value (starting from one) in order
    of decreasing size.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed magnetogram map from time 't'.
    rotated_map : `~sunpy.map.Map`
        Processed magnetogtam map from time 't - delta_t' differentially rotated to time t.
    dilation_radius : `~astropy.units.Quantity`, optional
        Radius of the disk for binary dilation (default is 5 arcsec).

    Returns
    -------
    sorted_labels : `~numpy.ndarray`
        Individual contiguous features are indexed by assigning ascending integer
        values (beginning with one) in order of decreasing feature size.

    """
    arcsec_to_pixel = ((current_map.scale[0] + current_map.scale[1]) / 2) ** (-1)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)

    current_mask = smooth_los_threshold(current_map)[1]
    rotated_mask = smooth_los_threshold(rotated_map)[1]

    # Grow the differentially rotated previous-frame mask so a feature that has shifted
    # slightly between the two times still overlaps its counterpart, then keep only the
    # current-frame features that overlap it -- everything else is a transient.
    grown_rotated_mask = ski.morphology.dilation(rotated_mask, disk(dilation_radius))
    current_labels = ski.measure.label(current_mask)
    persistent_labels = np.unique(current_labels[grown_rotated_mask & (current_labels > 0)])
    persistent_mask = np.isin(current_labels, persistent_labels)

    # Grow the surviving detection to form the final feature mask.
    final_mask = ski.morphology.dilation(persistent_mask, disk(dilation_radius))
    final_labels = ski.measure.label(final_mask)

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
