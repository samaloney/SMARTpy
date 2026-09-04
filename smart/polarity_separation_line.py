import numpy as np
import skimage as ski
from numpy.typing import NDArray
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map

__all__ = ["separate_polarities", "polarity_separation_line_mask", "psl_properties"]


def separate_polarities(feature_map, dilation_px: int = 4):
    """
    Split a feature magnetogram into dilated positive and negative polarity masks.

    Parameters
    ----------
    feature_map : `~numpy.ndarray`
        The feature's (masked) magnetic field, in Gauss.
    dilation_px : `int`, optional
        Radius, in pixels, of the disk used to dilate each polarity mask
        (Higgins et al. 2011 dilate by 4 pixels).

    Returns
    -------
    pos_dilated, neg_dilated : `~numpy.ndarray`
        The dilated positive and negative polarity masks (boolean), in that order.
    """
    footprint = disk(dilation_px)
    pos_dilated = ski.morphology.dilation(feature_map > 0, footprint)
    neg_dilated = ski.morphology.dilation(feature_map < 0, footprint)
    return pos_dilated, neg_dilated


def polarity_separation_line_mask(pos_dilated, neg_dilated):
    """
    Polarity separation line mask and its one-pixel-thinned version.

    The PSL is where the dilated positive and negative masks overlap
    (:math:`M_{PSL,t,i}`); thinning it to a single pixel gives
    :math:`M_{PSL,thin,t,i}`.

    Parameters
    ----------
    pos_dilated, neg_dilated : `~numpy.ndarray`
        Dilated positive / negative polarity masks from `separate_polarities`.

    Returns
    -------
    psl_mask, psl_thin_mask : `~numpy.ndarray`
        The PSL mask and its thinned skeleton (boolean).
    """
    psl_mask = np.asarray(pos_dilated, bool) & np.asarray(neg_dilated, bool)
    psl_thin_mask = ski.morphology.thin(psl_mask)
    return psl_mask, psl_thin_mask


@u.quantity_input
def psl_properties(
    im_map: Map,
    feature_mask: NDArray[bool],
    dilation_radius: u.Quantity[u.pix] = 4 * u.pix,
    gradient_threshold: u.Quantity[u.Gauss / u.Mm] = 50 * u.Gauss / u.Mm,
    r_star_fwhm: u.Quantity[u.pix] = 10 * u.pix,
    r_star_kernel: u.Quantity[u.pix] = 20 * u.pix,
):
    r"""
    Polarity-separation-line properties for one feature (Higgins et al. 2011, Table 2).

    The four explicitly-defined quantities are computed:

    * ``l_psl`` - PSL length, :math:`\sum M_{PSL,thin}` as a length on the solar surface.
    * ``l_sg`` - strong-gradient PSL length, summing only thinned PSL pixels where
      :math:`|\nabla B| > \text{gradient\_threshold}`.
    * ``r_star`` - :math:`\sum (M_{PSL} \circledast G_{2D}) \cdot |B_{t,i}|`, with a
      2-D Gaussian of ``r_star_fwhm`` FWHM over an ``r_star_kernel``-wide kernel. (Table 2
      writes :math:`\times B_{t,i}`, but a signed sum along a PSL cancels; :math:`|B|` is
      used, matching Schrijver 2007.)
    * ``wlsg_star`` -- :math:`\sum M_{PSL,thin} \cdot \nabla B_{t,i}` (gradient weighted,
      no gradient threshold).

    Schrijver's ``R`` and Falconer's ``WL_sg`` (citation-only in the paper) are not
    computed. :math:`\nabla B` uses second-order central differences, which equal the
    paper's 3-point Lagrangian derivative on a uniform grid.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        The characterisation magnetogram (:math:`B_t`), noise-thresholded and
        LOS-corrected.
    feature_mask : `~numpy.ndarray`
        Binary mask of the feature (from `~smart.calculate_properties.extract_features`).
    dilation_radius : `~astropy.units.Quantity`, optional
        Polarity-mask dilation radius in pixels (default 4 px).
    gradient_threshold : `~astropy.units.Quantity`, optional
        Field-gradient threshold for ``l_sg`` (default 50 G / Mm).
    r_star_fwhm, r_star_kernel : `~astropy.units.Quantity`, optional
        FWHM and full width (pixels) of the Gaussian kernel used for ``r_star``
        (defaults 10 px and 20 px).

    Returns
    -------
    `dict`
        ``{"l_psl", "l_sg", "r_star", "wlsg_star"}``.
    """
    feature_map = np.where(feature_mask.astype(bool), np.nan_to_num(im_map.data), 0.0)

    scale = (im_map.scale[0] + im_map.scale[1]) / 2
    pix_mm = ((im_map.rsun_meters / im_map.rsun_obs) * scale).to(u.Mm / u.pix)

    pos_dilated, neg_dilated = separate_polarities(feature_map, round(dilation_radius.to_value(u.pix)))
    psl_mask, psl_thin_mask = polarity_separation_line_mask(pos_dilated, neg_dilated)

    gradient = (np.hypot(*np.gradient(feature_map)) * (u.Gauss / u.pix) / pix_mm).to(u.Gauss / u.Mm)

    l_psl = (psl_thin_mask.sum() * u.pix * pix_mm).to(u.Mm)
    l_sg = ((psl_thin_mask & (gradient > gradient_threshold)).sum() * u.pix * pix_mm).to(u.Mm)

    sigma_px = r_star_fwhm.to_value(u.pix) / (2 * np.sqrt(2 * np.log(2)))
    truncate = (r_star_kernel.to_value(u.pix) / 2) / sigma_px
    r_mask = ski.filters.gaussian(psl_mask.astype(float), sigma_px, truncate=truncate)
    r_star = np.nansum(r_mask * np.abs(feature_map)) * u.Gauss

    wlsg_star = np.nansum(psl_thin_mask * gradient.to_value(u.Gauss / u.Mm)) * (u.Gauss / u.Mm)

    return {"l_psl": l_psl, "l_sg": l_sg, "r_star": r_star, "wlsg_star": wlsg_star}
