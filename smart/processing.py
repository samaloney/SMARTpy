import numpy as np
import skimage as ski
from matplotlib import colors
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

__all__ = [
    "remove_off_disk",
    "calculate_cosine_correction",
    "cosine_correct_data",
    "threshold_los",
    "smooth_los_threshold",
    "smart_prep",
]


def remove_off_disk(im_map):
    """
    Return a copy of ``im_map`` with off-disk pixels blanked.

    Off-limb pixels are set to NaN so that later processing only sees the solar
    disk. The returned map also carries display settings suited to a
    magnetogram (NaNs drawn black, a symmetric ``+/- 200`` colour scale). The
    input map is not modified.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Unprocessed magnetogram map.

    Returns
    -------
    `~sunpy.map.Map`
        Copy of ``im_map`` with off-disk pixels set to NaN.

    """
    disk_map = Map(im_map.data.copy(), im_map.meta.copy())
    off_disk = ~coordinate_is_on_solar_disk(all_coordinates_from_map(disk_map))
    disk_map.data[off_disk] = np.nan

    cmap = disk_map.cmap.copy()
    cmap.set_bad("k")
    disk_map.plot_settings["cmap"] = cmap
    disk_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return disk_map


@u.quantity_input
def smooth_los_threshold(
    im_map: Map,
    thresh: u.Quantity[u.Gauss] = 100 * u.Gauss,
    dilation_radius: u.Quantity[u.arcsec] = 5 * u.arcsec,
    sigma: u.Quantity[u.arcsec] = 10 * u.arcsec,
    min_area: u.Quantity[u.arcsec**2] = 4500 * u.arcsec**2,
    grow: bool = True,
):
    """
    Apply smoothing, a noise threshold and an LOS correction, in that order, then detect.

    Following Higgins et al. (2011): the raw magnetogram is Gaussian smoothed (which
    removes ephemeral regions), sub-threshold pixels are zeroed, and the surviving
    line-of-sight field is deprojected to radial. The result is made binary; with
    ``grow=True`` it is also dilated and small features are removed.

    Parameters
    ----------
    im_map : ``~sunpy.map.Map``
        Processed SunPy magnetogram map.
    thresh : `~astropy.units.Quantity`, optional
        Noise threshold; pixels below this in the smoothed map are zeroed (default 100 G).
    dilation_radius : `~astropy.units.Quantity`, optional
        Radius of the disk for binary dilation (default 5 arcsec).
    sigma : `~astropy.units.Quantity`, optional
        Standard deviation of the Gaussian smoothing kernel (default 10 arcsec).
    min_area : `~astropy.units.Quantity`, optional
        Minimum on-disk area for a region to be kept, as a solid angle that is
        converted to pixels for the map at hand (default 4500 arcsec**2). Higgins
        et al. (2011) use 50 pixels, but that is applied after a supergranule-scale
        smoothed detection that is not yet ported, so a larger backstop is used here.
    grow : `bool`, optional
        If `True` (default) the binary mask is dilated and regions smaller than
        ``min_area`` are dropped. If `False` the raw, un-grown binary mask
        :math:`M_t` is returned instead, for callers that do their own growing
        (e.g. `~smart.indexed_grown_mask.index_and_grow_mask`).

    Returns
    -------
    smooth_map : `~sunpy.map.Map`
        The smoothed, noise-thresholded, LOS-corrected magnetogram.
    filtered_labels : `numpy.ndarray`
        Boolean detection mask -- grown and area-filtered when ``grow=True``,
        the raw binary mask :math:`M_t` when ``grow=False``.
    mask_sizes : `~numpy.ndarray` or `None`
        Boolean array indicating which labelled regions exceed ``min_area``
        (`None` when ``grow=False``).

    """

    arcsec_to_pixel = ((im_map.scale[0] + im_map.scale[1]) / 2) ** (-1)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)
    sigma = (np.round(sigma * arcsec_to_pixel)).to_value(u.pix)
    min_area = np.round(min_area * arcsec_to_pixel**2).to_value(u.pix**2)

    # Smooth the raw magnetogram, then apply the noise threshold and LOS correction
    # (the paper's "TL Process").
    smoothed = Map(ski.filters.gaussian(np.nan_to_num(im_map.data), sigma), im_map.meta)
    smooth_map = threshold_los(smoothed, thresh=thresh)

    # Un-grown binary detection mask M_t.
    mask = np.abs(smooth_map.data) >= thresh.to_value(u.Gauss)
    if not grow:
        return smooth_map, mask, None

    # Grow the mask and drop small features.
    dilated_mask = ski.morphology.dilation(mask, disk(dilation_radius))

    labels = ski.measure.label(dilated_mask)
    label_areas = np.bincount(labels.ravel())
    mask_sizes = label_areas > min_area
    mask_sizes[0] = 0
    filtered_labels = mask_sizes[labels]
    return smooth_map, filtered_labels, mask_sizes


def calculate_cosine_correction(im_map: Map, limit: float = 0.99):
    r"""
    Find the cosine (:math:`1/\mu`) correction values for on-disk pixels.

    For each on-disk pixel the heliocentric angle :math:`\theta` is found from
    its angular distance from disk centre, and the line-of-sight to radial
    correction factor is :math:`1/\cos\theta`.  Off-disk pixels are set to 1.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    limit : `float`, optional
        Cap on :math:`\sin\theta`, so the correction cannot exceed
        ``1 / cos(arcsin(limit))``.  The default of 0.99 caps it at ~7.1
        (:math:`\theta \approx 82^\circ`), beyond which the deprojection is
        unreliable.

    Returns
    -------
    cos_correction : `~numpy.ndarray`
        Array of cosine correction factors for each pixel.

    """

    coordinates = all_coordinates_from_map(im_map)
    on_disk = coordinate_is_on_solar_disk(coordinates)

    cos_correction = np.ones_like(im_map.data)

    radial_angle = np.arccos(np.cos(coordinates.Tx[on_disk]) * np.cos(coordinates.Ty[on_disk]))
    sin_theta = (radial_angle / im_map.rsun_obs).decompose()
    sin_theta = np.clip(sin_theta, -limit, limit)

    cos_correction[on_disk] = 1 / np.cos(np.arcsin(sin_theta))

    return cos_correction


def cosine_correct_data(im_map: Map, cosmap=None, limit: float = 0.99):
    """
    Perform magnetic field cosine correction.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    cosmap : `numpy.ndarray`, optional
        An array of the cosine correction factors for each pixel.
        If not provided, computed using `calculate_cosine_correction`.
    limit : `float`, optional
        Passed to `calculate_cosine_correction`, and used to cap a
        ``cosmap`` that is supplied directly, at ``1 / cos(arcsin(limit))``.

    Returns
    -------
    corrected_data : `~numpy.ndarray`
        The magnetic field data after applying the cosine correction (units = Gauss).

    """
    if cosmap is None:
        cosmap = calculate_cosine_correction(im_map, limit=limit)

    cosmap = np.clip(cosmap, None, 1 / np.cos(np.arcsin(limit)))

    corrected_data = im_map.data * cosmap * u.Gauss
    return corrected_data


@u.quantity_input
def threshold_los(im_map: Map, thresh: u.Quantity[u.Gauss] = 70 * u.Gauss):
    """
    Noise-threshold and LOS-correct a magnetogram (the paper's "TL Process").

    Pixels with ``|B| < thresh`` are zeroed and the surviving line-of-sight field is
    deprojected to radial with `cosine_correct_data`. Unlike `smooth_los_threshold`
    there is no smoothing; this is the characterization-stage processing of Higgins
    et al. (2011), Section 2.2.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Magnetogram map (off-disk pixels should already be NaN).
    thresh : `~astropy.units.Quantity`, optional
        Noise threshold (default 70 G).

    Returns
    -------
    `~sunpy.map.Map`
        The noise-thresholded, LOS-corrected magnetogram (units of Gauss).

    """
    data = np.array(im_map.data, dtype=float)
    data[np.abs(data) < thresh.to_value(u.Gauss)] = 0
    corrected = cosine_correct_data(Map(data, im_map.meta))
    return Map(corrected.to_value(u.Gauss), im_map.meta)


def smart_prep(im_map, **kwargs):
    """
    Prepare map for use in segmentation and characterization processes.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Unprocessed SunPy magnetogram map.
    **kwargs
        Passed through to `smooth_los_threshold` (``thresh``, ``sigma``,
        ``dilation_radius``, ``min_area``).

    Returns
    -------
    disk_map : `~sunpy.map.Map`
        Copy of ``im_map`` with off-disk pixels blanked.
    cos_correction : `~numpy.ndarray`
        Array of cosine correction factors for each pixel.

    """
    disk_map = remove_off_disk(im_map)
    smooth_map, *_ = smooth_los_threshold(disk_map, **kwargs)
    cos_correction = calculate_cosine_correction(smooth_map)
    return disk_map, cos_correction
