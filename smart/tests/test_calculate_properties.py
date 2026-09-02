from sunpy.map import Map

from smart.calculate_properties import smart_indentify_and_characterize


def test_smart():
    cur_hmi_map = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_000000_TAI.fits")
    prev_hmi_map = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/05/hmi.M_720s.20240605_000000_TAI.fits")
    properties = smart_indentify_and_characterize(cur_hmi_map, prev_hmi_map)
    assert len(properties) > 0
