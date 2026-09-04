import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose

import sunpy.map
from sunpy.coordinates import get_earth

from smart.calculate_properties import get_properties, smart_indentify_and_characterize


def _disk_centre_map(data):
    """A minimal disk-centre magnetogram Map wrapping ``data`` (Gauss)."""
    obstime = "2015-01-01T00:00:00"
    coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame="helioprojective", obstime=obstime, observer=get_earth(obstime))
    header = sunpy.map.make_fitswcs_header(
        data,
        coord,
        scale=[2, 2] * u.arcsec / u.pix,
        telescope="TEST",
    )
    return sunpy.map.Map(data, header)


def _zero_dbdt(im_map):
    return sunpy.map.Map(np.zeros_like(im_map.data), im_map.meta), 1 * u.s


@pytest.fixture
def uniform_feature():
    data = np.zeros((101, 101))
    data[40:61, 40:61] = 100.0  # 21x21 patch of +100 G at disk centre
    labels = np.zeros((101, 101), dtype=int)
    labels[40:61, 40:61] = 1
    return _disk_centre_map(data), labels


@pytest.fixture
def bipole_feature():
    data = np.zeros((101, 101))
    data[40:61, 40:50] = 100.0
    data[40:61, 51:61] = -100.0
    labels = np.zeros((101, 101), dtype=int)
    labels[40:61, 40:61] = 1
    return _disk_centre_map(data), labels


def test_flux_imbalance_unipolar(uniform_feature):
    im_map, labels = uniform_feature
    dbdt_map, dt = _zero_dbdt(im_map)
    (props,) = get_properties(im_map, dbdt_map, dt, labels)

    assert_allclose(props["flux"]["imbalance"], 1.0, atol=1e-6)
    assert_quantity_allclose(props["field"]["mean"], 100 * u.Gauss, rtol=1e-6)
    assert_quantity_allclose(props["field"]["total_unsigned"], 21 * 21 * 100 * u.Gauss)
    assert_quantity_allclose(props["field"]["min"], 100 * u.Gauss)
    assert props["flux"]["positive"] > 0 * u.Wb
    assert_quantity_allclose(props["flux"]["negative"], 0 * u.Wb)


def test_flux_imbalance_balanced(bipole_feature):
    im_map, labels = bipole_feature
    dbdt_map, dt = _zero_dbdt(im_map)
    (props,) = get_properties(im_map, dbdt_map, dt, labels)

    assert_allclose(props["flux"]["imbalance"], 0.0, atol=1e-6)
    assert_quantity_allclose(props["field"]["total"], 0 * u.Gauss, atol=1e-6 * u.Gauss)


def test_emergence_rate_not_double_divided(uniform_feature):
    im_map, labels = uniform_feature
    # dB/dt constant at 1 G/s over the feature.
    dbdt_map = sunpy.map.Map(np.where(labels > 0, 1.0, 0.0), im_map.meta)
    dt = 3600 * u.s
    (props,) = get_properties(im_map, dbdt_map, dt, labels)

    expected = (1.0 * u.Gauss / u.s * props["geometry"]["area"]).to(u.Wb / u.s)
    assert_quantity_allclose(props["flux"]["emergence_rate"], expected, rtol=1e-3)


def test_hg_position_near_disk_centre(uniform_feature):
    im_map, labels = uniform_feature
    dbdt_map, dt = _zero_dbdt(im_map)
    (props,) = get_properties(im_map, dbdt_map, dt, labels)

    # The feature sits at the reference pixel (disk centre), so its Stonyhurst
    # position should match the sub-observer point: lon 0, lat = B0.
    sub_obs = im_map.observer_coordinate.heliographic_stonyhurst
    lon, lat = props["geometry"]["hg_position"]
    assert_quantity_allclose(lon, 0 * u.deg, atol=2 * u.deg)
    assert_quantity_allclose(lat, sub_obs.lat, atol=2 * u.deg)


def test_psl_group_present(bipole_feature):
    im_map, labels = bipole_feature
    dbdt_map, dt = _zero_dbdt(im_map)
    (props,) = get_properties(im_map, dbdt_map, dt, labels)

    psl = props["psl"]
    assert set(psl) == {"l_psl", "l_sg", "r_star", "wlsg_star"}
    assert psl["l_psl"] > 0 * u.Mm
    assert psl["l_sg"] <= psl["l_psl"]
    assert psl["r_star"].unit.is_equivalent(u.Gauss)
    assert psl["wlsg_star"].unit.is_equivalent(u.Gauss / u.Mm)


def test_smart():
    cur = sunpy.map.Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_000000_TAI.fits")
    prev = sunpy.map.Map("http://jsoc1.stanford.edu/data/hmi/fits/2024/06/05/hmi.M_720s.20240605_010000_TAI.fits")
    properties = smart_indentify_and_characterize(cur, prev)
    assert len(properties) > 0
    p = properties[0]
    assert set(p) == {"label", "geometry", "field", "flux", "psl"}
    assert p["flux"]["emergence_rate"].unit.is_equivalent(u.Wb / u.s)
    assert 0 <= p["flux"]["imbalance"] <= 1
    assert p["psl"]["l_psl"].unit.is_equivalent(u.Mm)
