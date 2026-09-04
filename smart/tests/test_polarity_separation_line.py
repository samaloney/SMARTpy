import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose

import sunpy.map
from sunpy.coordinates import get_earth

from smart.polarity_separation_line import polarity_separation_line_mask, psl_properties, separate_polarities


def _disk_centre_map(data):
    obstime = "2015-01-01T00:00:00"
    coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame="helioprojective", obstime=obstime, observer=get_earth(obstime))
    header = sunpy.map.make_fitswcs_header(data, coord, scale=[2, 2] * u.arcsec / u.pix, telescope="TEST")
    return sunpy.map.Map(data, header)


def _pix_mm(im_map):
    scale = (im_map.scale[0] + im_map.scale[1]) / 2
    return ((im_map.rsun_meters / im_map.rsun_obs) * scale).to(u.Mm / u.pix)


def _sharp_bipole(height=41):
    """A feature with a clean vertical PIL: +100 G left half, -100 G right half."""
    data = np.zeros((101, 101))
    top, bot = 50 - height // 2, 50 + height // 2 + 1
    data[top:bot, 35:50] = 100.0
    data[top:bot, 51:66] = -100.0
    mask = np.zeros((101, 101), dtype=int)
    mask[top:bot, 35:66] = 1
    return _disk_centre_map(data), mask, height


def test_separate_polarities_returns_pos_then_neg():
    fm = np.zeros((20, 20))
    fm[:10, :] = 100.0  # positive on top, nothing negative
    pos_dilated, neg_dilated = separate_polarities(fm, dilation_px=1)
    assert pos_dilated.any()
    assert not neg_dilated.any()


def test_l_psl_scales_with_pil_length():
    im_map, mask, height = _sharp_bipole(height=41)
    props = psl_properties(im_map, mask)
    # Thinned PIL runs the height of the feature (plus a little from the 4-px dilation).
    expected = height * _pix_mm(im_map) * u.pix
    assert props["l_psl"] > 0.6 * expected
    assert props["l_psl"] < 1.6 * expected


def test_l_sg_sharp_vs_smooth():
    # Sharp +/-100 G step -> gradient well above 50 G/Mm along the whole PIL.
    im_map, mask, _ = _sharp_bipole()
    sharp = psl_properties(im_map, mask)
    assert_quantity_allclose(sharp["l_sg"], sharp["l_psl"], rtol=0.2)

    # Smooth ramp from +100 to -100 over ~30 px -> gradient well below 50 G/Mm.
    data = np.zeros((101, 101))
    ramp = np.linspace(100, -100, 31)
    data[30:71, 35:66] = ramp[None, :]
    mask2 = np.zeros((101, 101), dtype=int)
    mask2[30:71, 35:66] = 1
    smooth = psl_properties(_disk_centre_map(data), mask2)
    assert smooth["l_sg"] == 0 * u.Mm


def test_r_star_and_wlsg_star_positive_finite():
    im_map, mask, _ = _sharp_bipole()
    props = psl_properties(im_map, mask)
    assert props["r_star"] > 0 * u.Gauss
    assert np.isfinite(props["r_star"].value)
    assert props["wlsg_star"] > 0 * u.Gauss / u.Mm
    assert np.isfinite(props["wlsg_star"].value)


def test_no_pil_gives_zero():
    data = np.zeros((60, 60))
    data[20:40, 20:40] = 100.0  # unipolar -> no polarity overlap
    mask = np.zeros((60, 60), dtype=int)
    mask[20:40, 20:40] = 1
    props = psl_properties(_disk_centre_map(data), mask)
    assert props["l_psl"] == 0 * u.Mm
    assert props["l_sg"] == 0 * u.Mm
    assert props["r_star"] == 0 * u.Gauss


def test_psl_mask_is_overlap_of_dilated_polarities():
    fm = np.zeros((40, 40))
    fm[:, :20] = 100.0
    fm[:, 20:] = -100.0
    pos_dilated, neg_dilated = separate_polarities(fm, dilation_px=3)
    psl_mask, psl_thin = polarity_separation_line_mask(pos_dilated, neg_dilated)
    assert np.array_equal(psl_mask, pos_dilated & neg_dilated)
    assert psl_thin.sum() <= psl_mask.sum()
