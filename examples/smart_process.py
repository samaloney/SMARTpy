"""
=========================
SMART processing example.
=========================

SMART processing example from the creation of the first map to the extraction of it's properties.
"""

from matplotlib import pyplot as plt

from sunpy.map import Map

from smart.calculate_properties import cosine_weighted_area_map, dB_dt, extract_features, get_properties
from smart.differential_rotation import diff_rotation
from smart.indexed_grown_mask import index_and_grow_mask, plot_indexed_grown_mask
from smart.processing import (
    calculate_cosine_correction,
    cosine_correct_data,
    remove_off_disk,
    smart_prep,
    smooth_los_threshold,
)

#####################################################
#
#
# Data Preparation
# -----------------
#
#
#
# We start by creating a sunpy.Map from our .fits file.

hmi_map = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_230000_TAI.fits")
# hmi_map = Map(
#     "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
# )

#####################################################
# We'll also plot this map to see how it looks before applying any of the SMART processes.

hmi_map.plot()

#####################################################
# We apply the `~smart.processing.remove_off_disk` function, which returns a copy of the map with the off-disk pixels set to NaN (drawn black). We'll plot this new map to see how it looks.

disk_map = remove_off_disk(hmi_map)
disk_map.plot()

#####################################################
# Next we'll use the `~smart.map_processing.smooth_los_threshold` function in order to get a smoothed version of our map. This will then be used with the
# `~smart.map_processing.calculate_cosine_correction` and `~smart.map_processing.cosine_correct_data` functions to get cosine corrected data for our map
#
# Once again we will use a plot to see how our corrected data looks.

smooth_map, filtered_labels, mask_sizes = smooth_los_threshold(disk_map)
cos_correction = calculate_cosine_correction(smooth_map)
corrected_data = cosine_correct_data(smooth_map, cos_correction)
plt.imshow(corrected_data.value)

#####################################################
# We now need to prepare our second map, taken from a time 'Δt' before the first.
#
# This time we'll simply use the `~smart.map_processing.smart_prep` function to create our thresholded map and calculate the corrected data. This function performs the
# `~smart.map_processing.smooth_los_threshold`, `~smart.map_processing.calculate_cosine_correction`, and `~smart.map_processing.cosine_correct_data` functions.

hmi_map_prev = Map("http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_000000_TAI.fits")
# hmi_map_prev = Map(
#     "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_000000_TAI.3.magnetogram.fits"
# )
thresholded_map_prev, cos_correction_prev = smart_prep(hmi_map_prev)

#####################################################
# We will now differentially rotate our second, earlier, map to the time, 't', of our original map. We will use the `~smart.differential_rotation.diff_rotation` function to do this. Later this rotated map will be useful for removing transient features from our masks.
#
# We plot the three maps. From left to right we have our map from time 't', our map from time 't - Δt', and finally our rotated map.

rotated_map = diff_rotation(hmi_map, hmi_map_prev)
diff_map = Map((hmi_map.data - rotated_map.data, rotated_map.meta))
fig, ax = plt.subplot_mosaic(
    [["cur", "prev", "diff"]],
    figsize=(9, 6),
    per_subplot_kw={
        "cur": {"projection": hmi_map},
        "prev": {"projection": hmi_map_prev},
        "diff": {"projection": rotated_map},
    },
)
hmi_map.plot(axes=ax["cur"])
hmi_map_prev.plot(axes=ax["prev"])
diff_map.plot(axes=ax["diff"])

#####################################################
#
# Region Detection
# -----------------

#####################################################
# Now that we have our map from time 't' and the map rotated to time 't', we can subtract them in order to remove any transient features due to solar rotation.
#
# The `~smart.indexed_grown_mask.index_and_grow_mask` function performs this operation, and also orders the detected active regions in order of descending area, and assigns an ascending integer value to
# these regions, starting with 1.

sorted_labels = index_and_grow_mask(hmi_map, hmi_map_prev)

#####################################################
# Using the `~smart.indexed_grown_mask.plot_indexed_grown_mask` function we can easily see how the map now looks with it's contours and region labels.

plot_indexed_grown_mask(hmi_map, sorted_labels)

#####################################################
#
# Characterization
# -----------------

#####################################################
# The `~smart.calculate_properties.extract_features` function returns an array of individual feature masks for each detected region.
# The `~smart.calculate_properties.cosine_weighted_area_map` is used to return a cosine corrected area map of the disk, which is later used in combination with the aforementioned feature masks in order to
# extract properties from individual regions.

feature_masks = extract_features(sorted_labels)
region1_feature_mask = feature_masks[:1]
region1_area_map = cosine_weighted_area_map(hmi_map)

#####################################################
# We will call the `~smart.calculate_properties.dB_dt` function and plot the result in order to see a map showing the temporal change of the magnetic field on the solar disk.

dBdt, dt = dB_dt(hmi_map, hmi_map_prev)
dBdt.plot()

#####################################################
# Finally, we use our `~smart.calculate_properties.get_properties` function to extract information relating to the magnetic field strength, flux, and area of each identified active region.

properties = get_properties(hmi_map, dBdt, dt, sorted_labels)

for i in range(len(properties)):
    for prop, value in properties[i].items():
        print(prop, ":", value)
    print()

#####################################################
#
