from copy import deepcopy

import numpy as np
import pytest

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.map.mapbase import GenericMap

from smart.processing import (
    calculate_cosine_correction,
    cosine_correct_data,
    remove_off_disk,
    smooth_los_threshold,
)


@pytest.fixture(scope="session")
def hmi_nrt():
    return "https://jsoc1.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_230000_TAI.fits"


@pytest.fixture(scope="session")
def mag_map_sample(hmi_nrt):
    return remove_off_disk(Map(hmi_nrt))


@pytest.fixture
def create_fake_map(value, shape=((4098, 4098))):
    fake_data = np.ones(shape) * value
    return Map(fake_data, mag_map_sample().meta)


def test_remove_off_disk(mag_map_sample):
    processed_map = remove_off_disk(mag_map_sample)

    assert isinstance(processed_map, GenericMap), "Result is not a SunPy Map."
    assert processed_map is not mag_map_sample, "Input map was not copied."

    coordinates = all_coordinates_from_map(processed_map)
    on_solar_disk = coordinate_is_on_solar_disk(coordinates)
    assert np.all(np.isnan(processed_map.data[~on_solar_disk])), "Off-disk NaN values not set correctly."


def test_get_cosine_correction_shape(mag_map_sample):
    cos_cor = calculate_cosine_correction(mag_map_sample)
    assert cos_cor.shape == mag_map_sample.data.shape, "cos_cor shape != hmi_nrt.data.shape"


def test_get_cosine_correction_limits(mag_map_sample):
    cos_cor = calculate_cosine_correction(mag_map_sample)

    edge = 0.99
    # coordinates = all_coordinates_from_map(mag_map_sample)
    # on_disk = coordinate_is_on_solar_disk(coordinates)
    # off_disk = ~on_disk

    assert np.all(cos_cor >= 0), "cos_cor lower limits incorrect"
    assert np.all(cos_cor <= 1 / np.cos(np.arcsin(edge))), "cos_cor upper limits incorrect"


def test_cosine_correction(mag_map_sample):
    coordinates = all_coordinates_from_map(mag_map_sample)

    los_radial = np.cos(coordinates.Tx.to(u.rad)) * np.cos(coordinates.Ty.to(u.rad))

    fake_map = Map(los_radial, mag_map_sample.meta)
    fake_cosmap = np.ones((len(los_radial), len(los_radial)))

    corrected_data = cosine_correct_data(fake_map, fake_cosmap)
    corrected_data_value = corrected_data.to_value(u.Gauss)
    assert np.allclose(corrected_data_value, 1, atol=1e-4), "cosine corrected data not behaving as expected"


def test_smooth_los_threshold(mag_map_sample):
    under_thresh = deepcopy(mag_map_sample)
    under_thresh.data[:, :] = 1
    over_thresh = deepcopy(mag_map_sample)
    over_thresh.data[:, :] = 1000

    smooth_under, fl_under, mask_under = smooth_los_threshold(under_thresh, thresh=500 * u.Gauss)
    smooth_over, fl_over, mask_over = smooth_los_threshold(over_thresh, thresh=500 * u.Gauss)

    assert isinstance(smooth_under, type(under_thresh)), "smooth_under is no longer a Map"
    assert isinstance(smooth_over, type(over_thresh)), "smooth_over is no longer a Map"

    assert np.sum(fl_under) == 0, "fl should all be False when all data is below threshold"
    assert np.sum(fl_over) == len(fl_over.flatten()), "fl should all be True when all data is above threshold "

    assert np.sum(mask_under) == 0, "no regions should have been detected in 'under_thresh'"
    assert np.sum(mask_over) > 0, "background region should have been detected in 'over thresh'"
