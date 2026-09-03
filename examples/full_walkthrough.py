"""
================
Full Walkthrough
================

Here we see the functions needed to quickly extract properties from a magnetogram.
"""

from pprint import pprint

from sunpy.map import Map

from smart.calculate_properties import dB_dt, get_properties, smart_indentify_and_characterize
from smart.indexed_grown_mask import index_and_grow_mask
from smart.processing import remove_off_disk, threshold_los

#####################################################
#

hmi_map = Map(
    "http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_230000_TAI.fits"
    # "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"
)
hmi_map_prev = Map(
    "http://jsoc.stanford.edu/data/hmi/fits/2024/06/06/hmi.M_720s.20240606_000000_TAI.fits"
    # "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_000000_TAI.3.magnetogram.fits"
)


disk_map = remove_off_disk(hmi_map)
disk_map_prev = remove_off_disk(hmi_map_prev)

sorted_labels = index_and_grow_mask(disk_map, disk_map_prev)

tl_map = threshold_los(disk_map)
tl_map_prev = threshold_los(disk_map_prev)
dBdt, dt = dB_dt(tl_map, tl_map_prev)

properties = get_properties(tl_map, dBdt, dt, sorted_labels)

for props in properties:
    pprint(props)
    print()

#####################################################
# We can also use the `~smart.calculate_properties.smart_identify_and_characterize` function to quickly and easily get these properties

smart_properties = smart_indentify_and_characterize(hmi_map, hmi_map_prev)

for i in range(len(smart_properties)):
    for prop, value in smart_properties[i].items():
        print(prop, ":", value)
    print()
