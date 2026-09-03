import numpy as np
from scipy import stats

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

from smart.differential_rotation import diff_rotation
from smart.indexed_grown_mask import index_and_grow_mask
from smart.processing import calculate_cosine_correction, remove_off_disk, threshold_los

__all__ = ["cosine_weighted_area_map", "extract_features", "dB_dt", "get_properties"]


def cosine_weighted_area_map(im_map: Map, limit: float = 0.999):
    """
    Per-pixel area on the solar surface (plane-of-sky pixel area deprojected by 1 / cos(theta)).

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    limit : `float`, optional
        Cap on ``sin(theta)`` for the deprojection, passed to
        `~smart.processing.calculate_cosine_correction`. Looser than the field
        correction's default (0.99) because Table 1's ``A_cos`` is not capped;
        0.999 limits the area factor to ~22 rather than ~7.

    Returns
    -------
    area_map : `~astropy.units.Quantity`
        Deprojected pixel area for every pixel, in square metres.
    """
    cos_cor = calculate_cosine_correction(im_map, limit=limit)

    m_per_arcsec = im_map.rsun_meters / im_map.rsun_obs
    pixel_area = (im_map.scale[0] * m_per_arcsec) * (im_map.scale[1] * m_per_arcsec) * u.pix**2

    return pixel_area * cos_cor


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
    Map of the temporal change in field strength, ``(B_t - B_{t-dt, rotated}) / dt``.

    The earlier magnetogram is differentially rotated to the time of ``current_map`` and
    subtracted from it, and the difference is divided by the time separation.

    Parameters
    ----------
    current_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map from time 't'.
    previous_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map from time 't - delta_t'.

    Returns
    -------
    dB_dt : `~sunpy.map.Map`
        Map of the change in magnetic field strength over time (units of Gauss / second).
    dt : `~astropy.units.Quantity`
        The time interval over which the change was measured.
    """
    diff_map = diff_rotation(current_map, previous_map)

    dB = (current_map.data - diff_map.data) * u.Gauss
    dt = (current_map.date - previous_map.date).to(u.s)

    dbdt_map = Map(dB / dt, current_map.meta)
    dbdt_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(dbdt_map))] = np.nan
    cmap = dbdt_map.cmap.copy()
    cmap.set_bad("k")
    dbdt_map.plot_settings["cmap"] = cmap
    return dbdt_map, dt


def get_properties(im_map, dbdt_map, dt, sorted_labels):
    """
    Magnetic properties of each detected feature (Higgins et al. 2011, Table 1).

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        The characterisation magnetogram at time 't': noise-thresholded and LOS-corrected
        (`~smart.processing.threshold_los`), *not* smoothed.
    dbdt_map : `~sunpy.map.Map`
        Output of `dB_dt` (units of Gauss / second).
    dt : `~astropy.units.Quantity`
        Time separation of the two magnetograms (kept for reference; not needed for the
        flux-emergence rate, which uses ``dbdt_map`` directly).
    sorted_labels : `~numpy.ndarray`
        Indexed grown mask from `~smart.indexed_grown_mask.index_and_grow_mask`.

    Returns
    -------
    properties : `list` of `dict`
        One entry per feature, with ``geometry`` / ``field`` / ``flux`` groups. Table 1
        identifiers: ``geometry`` -> ``HG_pos``, ``A_tot``; ``field`` -> ``B_min``,
        ``B_max``, ``B_tot``, ``B_totuns``, ``mu``, ``sigma^2``, ``gamma``, ``kappa``;
        ``flux`` -> ``Phi_+``, ``Phi_-``, ``Phi_uns``, ``Phi_imb``, ``dPhi/dt_net``.
    """
    feature_masks = extract_features(sorted_labels)

    area_map = cosine_weighted_area_map(im_map)
    phi_map = im_map.data * u.Gauss * area_map
    dbdt = np.nan_to_num(dbdt_map.data) * (u.Gauss / u.s)

    hgs = all_coordinates_from_map(im_map).heliographic_stonyhurst
    lon = hgs.lon.wrap_at(180 * u.deg)
    lat = hgs.lat

    hemisphere_millionths = (2 * np.pi * im_map.rsun_meters**2) / 1e6

    properties = []
    for i, feature_mask in enumerate(feature_masks, start=1):
        sel = feature_mask.astype(bool)

        b = im_map.data[sel] * u.Gauss
        b_val = b.to_value(u.Gauss)
        weights = np.abs(b_val)

        # Skew / kurtosis are undefined for a constant field (all sub-threshold or all
        # equal); report 0 in that degenerate case rather than dividing by zero.
        if np.nanvar(b_val) > 0:
            skewness = float(stats.skew(b_val, nan_policy="omit"))
            kurtosis = float(stats.kurtosis(b_val, nan_policy="omit"))
        else:
            skewness = kurtosis = 0.0

        area = np.nansum(area_map[sel]).to(u.Mm**2)

        phi = phi_map[sel]
        flux_pos = np.nansum(phi[phi > 0]).to(u.Wb)
        flux_neg = np.nansum(phi[phi < 0]).to(u.Wb)
        flux_uns = np.nansum(np.abs(phi)).to(u.Wb)
        flux_imb = (np.abs(flux_pos + flux_neg) / flux_uns).to_value(u.dimensionless_unscaled)

        emergence_rate = np.nansum(dbdt[sel] * area_map[sel]).to(u.Wb / u.s)

        if weights.sum() > 0:
            hg_position = (np.average(lon[sel], weights=weights), np.average(lat[sel], weights=weights))
        else:
            hg_position = (np.nanmean(lon[sel]), np.nanmean(lat[sel]))
        hg_centroid = (np.nanmean(lon[sel]), np.nanmean(lat[sel]))

        properties.append(
            {
                "label": i,
                "geometry": {
                    "hg_position": hg_position,
                    "hg_centroid": hg_centroid,
                    "area": area,
                    "area_millionths": (area / hemisphere_millionths).to_value(u.dimensionless_unscaled),
                },
                "field": {
                    "min": np.nanmin(b),
                    "max": np.nanmax(b),
                    "total": np.nansum(b),
                    "total_unsigned": np.nansum(np.abs(b)),
                    "mean": np.nanmean(b),
                    "variance": np.nanvar(b),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                },
                "flux": {
                    "positive": flux_pos,
                    "negative": flux_neg,
                    "unsigned": flux_uns,
                    "imbalance": flux_imb,
                    "emergence_rate": emergence_rate,
                },
            }
        )

    return properties


def smart_indentify_and_characterize(im_map, previous_map, **kwargs):
    """
    Identify and characterise solar features from two magnetograms.

    The two magnetograms are prepared, features are detected and transient-filtered
    (`~smart.indexed_grown_mask.index_and_grow_mask`), and each feature's magnetic
    properties are measured on the noise-thresholded, LOS-corrected field.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Magnetogram from time 't'.
    previous_map : `~sunpy.map.Map`
        Magnetogram from an earlier time, used for transient removal and the flux
        emergence rate.
    **kwargs
        Passed to `~smart.indexed_grown_mask.index_and_grow_mask`; ``thresh`` (default
        70 G) is also used for the characterisation `~smart.processing.threshold_los`.

    Returns
    -------
    properties : `list` of `dict`
        See `get_properties`.
    """
    disk_t = remove_off_disk(im_map)
    disk_prev = remove_off_disk(previous_map)

    sorted_labels = index_and_grow_mask(disk_t, disk_prev, **kwargs)

    thresh = kwargs.get("thresh", 70 * u.Gauss)
    tl_t = threshold_los(disk_t, thresh)
    tl_prev = threshold_los(disk_prev, thresh)

    dbdt_map, dt = dB_dt(tl_t, tl_prev)

    return get_properties(tl_t, dbdt_map, dt, sorted_labels)
