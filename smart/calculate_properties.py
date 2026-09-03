import numpy as np

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

from smart.differential_rotation import diff_rotation
from smart.indexed_grown_mask import index_and_grow_mask
from smart.processing import calculate_cosine_correction, smart_prep

__all__ = ["cosine_weighted_area_map", "extract_features", "dB_dt", "get_properties"]


def cosine_weighted_area_map(im_map: Map):
    """
    Calculate the cosine-weighted area map for a feature and determine the feature's total area.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    feature_mask : `~numpy.ndarray`
        Binary mask where feature pixels = 1 and background pixels = 0.

    Returns
    -------
    total_area : `~astropy.units.quantity.Quantity`
        The total area of the feature in square metres.
    area_map : `~astropy.units.quantity.Quantity`
        Area map corrected for cosine projection.
    """
    cos_cor = calculate_cosine_correction(im_map)

    m_per_arcsec = im_map.rsun_meters / im_map.rsun_obs
    pixel_area = (im_map.scale[0] * m_per_arcsec) * (im_map.scale[1] * m_per_arcsec)
    pixel_area = pixel_area * u.pix**2

    area_map = pixel_area * cos_cor

    return area_map


def extract_features(sorted_labels):
    """
    Extract binary masks for each feature found in index_and_grow_mask's sorted_labels.

    Parameters
    ----------
    sorted_labels : `~numpy.ndarray`
        An array where each unique label corresponds to a different feature on the solar disk.

    Returns
    -------
    feature_masks : `~numpy.ndarray`
        An array containing a binary mask for each identified feature.
    """
    unique_labels = np.unique(sorted_labels)
    unique_labels = unique_labels[unique_labels != 0]

    feature_masks = []
    for label_value in unique_labels:
        feature_mask = (sorted_labels == label_value).astype(int)
        feature_masks.append(feature_mask)

    return feature_masks


def dB_dt(current_map: Map, previous_map: Map):
    """
    A magnetogram differentially rotated to time 't' is subtracted from a processed magnetogram from time 't', and the resultant map is divided
    by their time separation, yielding a map of the temporal change in field strength.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map from time 't'.
    previous_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map from time 't - delta_t'.

    Returns
    -------
    dB_dt : `~sunpy.map.Map`
        Map showcasing the change in magnetic field strength over time.
    dB : `astropy.units.quantity.Quantity`
        The change in magnetic field strength.
    dt : `astropy.units.quantity.Quantity`
        The time interval over which the change in magnetic field strength was measured.
    """
    diff_map = diff_rotation(current_map, previous_map)

    dB = (current_map.data - diff_map.data) * u.Gauss
    dt = (current_map.date - previous_map.date).to(u.s)

    dB_dt = Map(dB / dt, current_map.meta)
    dB_dt.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(dB_dt))] = np.nan
    dB_dt.cmap.set_bad("k")
    return dB_dt, dt


def get_properties(im_map, dB_dt, dt, sorted_labels):
    """
    Calculate various properties for each detected feature.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    dB_dt : `~sunpy.map.Map`
        Map showcasing the change in magnetic field strength over time.
    dt : `astropy.units.quantity.Quantity`
        The time interval over which the change in magnetic field strength was measured.
    sorted_labels : `~numpy.ndarray`
        An array where each unique label corresponds to a different feature on the solar disk.

    Returns
    -------
    properties : `~list`
        A list containing properties related to each individual feature.
    """

    feature_masks = extract_features(sorted_labels)

    area_map = cosine_weighted_area_map(im_map)

    properties = []
    for i, feature_mask in enumerate(feature_masks, start=1):
        region_area_map = area_map * feature_mask
        dBdt_data = dB_dt.data * u.G

        total_area = np.sum(region_area_map).to(u.km**2)
        millionths = ((2 * np.pi * im_map.rsun_meters**2) / (1 * 10**6)).to(u.km**2)
        total_area_mh = total_area / millionths

        magnetic_flux = np.nansum(dBdt_data * region_area_map)
        flux_emergence_rate = magnetic_flux / dt

        B = im_map.data * feature_mask * u.G
        B_mean = np.nanmean(B)
        B_std = np.nanstd(B)
        B_min = np.nanmin(B)
        B_max = np.nanmax(B)

        flux_pos = np.nansum(B[B > 0] * area_map[B > 0])
        flux_neg = np.nansum(B[B < 0] * area_map[B < 0])
        flux_uns = np.nansum(np.abs(B) * area_map)
        flux_imb = (flux_pos - flux_neg) / (flux_pos + flux_neg)

        properties.append(
            {
                "feature label": i,
                "flux emergence rate": flux_emergence_rate.to(u.Wb / u.s),
                "mean B": B_mean.to(u.G),
                "std B": B_std.to(u.G),
                "minimum B": B_min.to(u.G),
                "maximum B": B_max.to(u.G),
                "positive flux": flux_pos.to(u.Wb),
                "negative flux": flux_neg.to(u.Wb),
                "unsigned flux": flux_uns.to(u.Wb),
                "flux imbalance": flux_imb,
                "total area (millionths)": total_area_mh,
            }
        )

    return properties


def smart_indentify_and_characterize(im_map, previous_map, **kwargs):
    """
    Identifies and characterises solar features from magnetogram maps.

    This function prepares two magnetograms, one from our desired time, 't', and one taken before this, and then calculates properties of the desired magnetogram,
    such as total area and flux emergence rate. To do this, the magnetograms are processed, features are detected, and a binary mask is created for the map. Then,
    the previous map is rotated to match the reference magnetogram. This rotated map has a binary mask created and then compared to the other binary mask, so we
    can remove transient features due to solar rotation. Individual masks are made for each feature, and finally each of these features has their properties calculated.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Map from which features are to be detected and properties extracted.
    previous_map : `sunpy.map.Map`
        Map to be rotated and compared to primary map.

    Returns
    -------
    properties : `~list`
        A list containing properties related to each individual feature.
    """
    threshold_map, *_ = smart_prep(im_map)
    threshold_map_prev, *_ = smart_prep(previous_map)

    sorted_labels = index_and_grow_mask(threshold_map, threshold_map_prev, **kwargs)

    dBdt, dt = dB_dt(threshold_map, threshold_map_prev)

    properties = get_properties(threshold_map, dBdt, dt, sorted_labels)
    return properties
