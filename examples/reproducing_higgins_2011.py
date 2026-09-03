"""
=============================================
Reproducing figures from Higgins et al. (2011)
=============================================

``smart`` is a port of the SMART algorithm described in `Higgins et al. (2011)
<https://doi.org/10.1016/j.asr.2010.06.024>`__.  This example recreates three
figures from that paper with the ported code, using the same SOHO/MDI
line-of-sight magnetograms and dates as the originals:

* **Figure 3** -- the per-feature processing steps, from a calibrated
  magnetogram to the indexed grown feature mask.
* **Figure 5** -- the magnetic field distribution of an active region compared
  with nearby quiet Sun, which motivates the +/- 70 G noise threshold.
* **Figure 11** -- detections for three cases where SMART groups flux
  differently from the NOAA active-region catalogue.

.. note::

    SMART's feature *tracking* (YAFTA) is not ported yet, so the Figure 11
    panels here are a detection-only approximation: each panel is an
    independent single-magnetogram detection, the contours are not linked
    across columns, and there is no persistent per-feature identity.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize

import sunpy.map
from sunpy.coordinates import HeliographicStonyhurst, propagate_with_solar_surface
from sunpy.net import Fido
from sunpy.net import attrs as a

from smart.differential_rotation import diff_rotation
from smart.indexed_grown_mask import index_and_grow_mask
from smart.processing import remove_off_disk, smooth_los_threshold

#####################################################
# Helpers
# -------
#
# The MDI magnetograms come from `JSOC <http://jsoc.stanford.edu>`__ (series
# ``mdi.fd_M_96m_lev182``, the 96-minute line-of-sight magnetograms).
# ``fetch_mdi`` searches a time window and returns the frames as
# `~sunpy.map.Map` objects.  ``noaa_center`` and ``central_ar`` look up an
# active-region position from the `HEK <https://www.lmsal.com/hek/>`__ (the
# NOAA Solar Region Summary); ``project`` drifts such a position with solar
# rotation onto a given magnetogram, so it can be used as the centre of a
# cutout.  ``paper_parameters`` returns the segmentation parameters quoted in
# Higgins et al. (2011) -- which are given in MDI pixels -- as angular
# quantities for a given map.

# Any e-mail address registered with JSOC at
# http://jsoc.stanford.edu/ajax/register_email.html works here.
MDI_SERIES = a.jsoc.Series("mdi.fd_M_96m_lev182")
JSOC_NOTIFY = a.jsoc.Notify("maloneys@tcd.ie")


def paper_parameters(smap):
    """Higgins et al. (2011) segmentation parameters, in angular units for ``smap``.

    The paper convolves with a 10x10 pixel kernel of FWHM 5 pixels, thresholds
    at +/- 70 G, dilates by 10 pixels and drops features smaller than 50 pixels.
    """
    scale = (smap.scale[0] + smap.scale[1]) / 2
    return dict(
        thresh=70 * u.Gauss,
        sigma=(5 * u.pix * scale) / (2 * np.sqrt(2 * np.log(2))),
        dilation_radius=10 * u.pix * scale,
        min_area=50 * u.pix**2 * scale**2,
    )


def fetch_mdi(start, end, sample=None):
    """Search JSOC for MDI LOS magnetograms in ``[start, end]`` and load them."""
    search_attrs = [a.Time(start, end), MDI_SERIES, JSOC_NOTIFY]
    if sample is not None:
        search_attrs.append(a.Sample(sample))
    files = sorted(f for f in Fido.fetch(Fido.search(*search_attrs)) if str(f).endswith(".fits"))
    maps = sunpy.map.Map(files)
    return maps if isinstance(maps, list) else [maps]


def _hek_stonyhurst(row):
    return SkyCoord(
        row["hgs_x"] * u.deg,
        row["hgs_y"] * u.deg,
        frame=HeliographicStonyhurst,
        obstime=row["event_starttime"],
        observer="earth",
    )


def _hek_ars(day):
    """All HEK active-region records within a day of ``day`` that carry a NOAA number."""
    day = Time(day)
    hek = Fido.search(a.Time(day - 12 * u.hour, day + 36 * u.hour), a.hek.EventType("AR"))["hek"]
    return [row for row in hek if row["ar_noaanum"]]


def noaa_center(noaa, day):
    """Stonyhurst position of a named NOAA active region, from the HEK.

    The NOAA Solar Region Summary lists each region once per day at 00:00 UT;
    we take the entry for ``noaa`` closest in time to ``day``.
    """
    rows = [row for row in _hek_ars(day) if int(row["ar_noaanum"]) == noaa]
    row = min(rows, key=lambda r: abs((Time(r["event_starttime"]) - Time(day)).to_value("s")))
    return _hek_stonyhurst(row)


def central_ar(day):
    """Stonyhurst position of the HEK active region closest to disk centre on ``day``."""
    row = min(_hek_ars(day), key=lambda r: r["hgs_x"] ** 2 + r["hgs_y"] ** 2)
    return _hek_stonyhurst(row)


def project(coord, ref_map):
    """Drift ``coord`` with solar rotation onto ``ref_map`` (helioprojective)."""
    with propagate_with_solar_surface():
        return coord.transform_to(ref_map.coordinate_frame)


def crop(smap, center, fov):
    """Square cutout of ``smap`` of width ``fov`` centred on ``center``."""
    half = fov / 2
    bottom_left = SkyCoord(center.Tx - half, center.Ty - half, frame=smap.coordinate_frame)
    top_right = SkyCoord(center.Tx + half, center.Ty + half, frame=smap.coordinate_frame)
    return smap.submap(bottom_left, top_right=top_right)


def copy_map(smap):
    return sunpy.map.Map(smap.data.copy(), smap.meta.copy())


#####################################################
# Figure 3 -- processing steps for one feature extraction
# ------------------------------------------------------
#
# Higgins et al. (2011), Figure 3, follows one feature extraction on
# 25 November 2003 around NOAA 10507:
#
# * **(A)** the calibrated magnetogram :math:`B_t`;
# * **(B)** :math:`B_t` after Gaussian smoothing, noise thresholding and the
#   line-of-sight correction (`~smart.processing.smooth_los_threshold`);
# * **(C)** the un-grown binary detection mask :math:`M_t` (``grow=False``),
#   i.e. before the 10-pixel dilation;
# * **(D)** the final indexed grown feature mask :math:`IGM_{t,i}`
#   (`~smart.indexed_grown_mask.index_and_grow_mask`) -- :math:`M_t` with
#   small and transient features removed, dilated, and indexed by size.
#   Transients are found against a magnetogram one time step (96 minutes)
#   earlier.
#
# Both functions are given the paper's parameters via ``paper_parameters``.

b_prev, b_t = fetch_mdi("2003-11-25 01:00", "2003-11-25 04:00")[-2:]

disk_b_t = remove_off_disk(b_t)
paper = paper_parameters(b_t)
rotated = diff_rotation(disk_b_t, remove_off_disk(b_prev))

smooth_map, *_ = smooth_los_threshold(disk_b_t, **paper)
binary_mask = smooth_los_threshold(disk_b_t, **paper, grow=False)[1]
igm = index_and_grow_mask(disk_b_t, rotated, **paper)

#####################################################
# .. note::
#
#     The detection is busier than the published Figure 3: NOAA 10507 sits in
#     an extended activity band, and the 10-pixel dilation bridges the
#     surrounding network into the region.  The paper's tighter crop and the
#     supergranule-scale detection of the later IDL library (not ported) both
#     suppress this.

center = project(noaa_center(10507, "2003-11-25"), b_t)
fov = 600 * u.arcsec
clip = ImageNormalize(vmin=-1000, vmax=1000)

panels = [
    (crop(copy_map(b_t), center, fov), "(A) calibrated $B_t$", dict(cmap="gray", norm=clip)),
    (
        crop(sunpy.map.Map(smooth_map.data, b_t.meta), center, fov),
        "(B) smoothed + thresholded + LOS",
        dict(cmap="gray", norm=clip),
    ),
    (
        crop(sunpy.map.Map(binary_mask.astype(float), b_t.meta), center, fov),
        "(C) binary mask $M_t$ (un-grown)",
        dict(cmap="binary"),
    ),
    (
        crop(sunpy.map.Map((igm > 0).astype(float), b_t.meta), center, fov),
        "(D) indexed grown mask $IGM_{t,i}$",
        dict(cmap="binary"),
    ),
]

fig = plt.figure(figsize=(9, 9))
for n, (smap, title, plot_kw) in enumerate(panels, start=1):
    ax = fig.add_subplot(2, 2, n, projection=smap)
    smap.plot(axes=ax, **plot_kw)
    ax.set_title(title)
fig.suptitle(f"Higgins et al. (2011) Figure 3 -- {b_t.date.iso[:19]}")
fig.tight_layout()

#####################################################
# Figure 5 -- the +/- 70 G noise threshold
# ---------------------------------------
#
# Higgins et al. (2011), Figure 5, compares an active region with a nearby
# patch of quiet Sun in an MDI magnetogram from 17 September 1997 (NOAA 8086).
# Both cutouts are smoothed with a 5-pixel-FWHM Gaussian, and the distributions
# of :math:`|B|` are compared: subtracting the quiet-Sun distribution from the
# active-region one (thick red line) shows that thresholding near 70 G
# (dot-dashed line) removes the quiet-Sun background while keeping the
# active-region field.

mdi_97 = fetch_mdi("1997-09-17 11:00", "1997-09-17 13:00")[0]

ar_center_97 = project(noaa_center(8086, "1997-09-17"), mdi_97)
qs_center_97 = SkyCoord(ar_center_97.Tx, ar_center_97.Ty - 350 * u.arcsec, frame=mdi_97.coordinate_frame)
fov_97 = 280 * u.arcsec

ar_cut = crop(mdi_97, ar_center_97, fov_97)
qs_cut = crop(mdi_97, qs_center_97, fov_97)

fwhm = 5
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
ar_smooth = gaussian(np.nan_to_num(ar_cut.data), sigma)
qs_smooth = gaussian(np.nan_to_num(qs_cut.data), sigma)

# The paper uses constant-width (linear) bins, shown on log-log axes -- hence
# the few wide steps at low |B| and the dense, jagged tail above ~100 G.
bins = np.arange(0, 1000 + 5, 5)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
ar_hist, _ = np.histogram(np.abs(ar_smooth), bins=bins)
qs_hist, _ = np.histogram(np.abs(qs_smooth), bins=bins)
floor = 0.1  # so empty / negative bins still plot on the log axis

fig, axes = plt.subplot_mosaic([["ar", "qs"], ["hist", "hist"]], figsize=(9, 8))
for key, image, title in [("ar", ar_smooth, "NOAA 8086"), ("qs", qs_smooth, "nearby quiet Sun")]:
    axes[key].imshow(image, origin="lower", cmap="gray", vmin=-300, vmax=300)
    axes[key].set_title(title)
    axes[key].set_xticks([])
    axes[key].set_yticks([])

axes["hist"].step(bin_centers, np.clip(qs_hist, floor, None), where="mid", color="tab:blue", ls="-.", label="QS")
axes["hist"].step(bin_centers, np.clip(ar_hist, floor, None), where="mid", color="k", lw=1, label="AR")
axes["hist"].step(
    bin_centers, np.clip(ar_hist - qs_hist, floor, None), where="mid", color="firebrick", lw=2, label="AR $-$ QS"
)
axes["hist"].axvline(70, ls="-.", color="green")
axes["hist"].set_xscale("log")
axes["hist"].set_yscale("log")
axes["hist"].set_xlim(1, 1000)
axes["hist"].set_ylim(floor, None)
axes["hist"].set_xlabel("$|B|$ [G]")
axes["hist"].set_ylabel("number of pixels")
axes["hist"].legend()
fig.suptitle(f"Higgins et al. (2011) Figure 5 -- {mdi_97.date.iso[:19]}")
fig.tight_layout()

#####################################################
# Figure 11 -- detections that diverge from the NOAA catalogue
# ----------------------------------------------------------
#
# Higgins et al. (2011), Figure 11, shows three cases where SMART groups flux
# into features differently from the NOAA active-region catalogue:
#
# * **(A)** two bipolar regions join and later fragment (3-9 October 2003);
# * **(B)** several small bipolar regions merge into one complex (2-7 May 2002);
# * **(C)** a bipole is first detected as two unipolar features, then as one
#   bipolar region (2-5 August 2004).
#
# Each panel is an independent single-magnetogram detection
# (`~smart.processing.smooth_los_threshold` with the paper's parameters),
# cropped around the HEK active region nearest disk centre for that case and
# followed with solar rotation.  The detected feature contours are drawn in
# green; without tracking they carry no identity from one column to the next
# -- see the note at the top of this example.

cases = {
    "(A) join then fragment": (10471, ["2003-10-03", "2003-10-05", "2003-10-09"]),
    "(B) small bipoles merge": (9936, ["2002-05-02", "2002-05-04", "2002-05-07"]),
    "(C) two unipolar to one bipole": (10655, ["2004-08-02", "2004-08-03", "2004-08-05"]),
}
fov_11 = 500 * u.arcsec

fig, axes = plt.subplots(3, 3, figsize=(11, 11), squeeze=False)
for row, (label, (noaa, days)) in enumerate(cases.items()):
    daily = fetch_mdi(f"{days[0]} 00:00", f"{days[-1]} 18:00", sample=1 * u.day)
    maps: u.s = [min(daily, key=lambda m: abs((m.date - Time(day)).to_value("s"))) for day in days]
    ar = noaa_center(noaa, days[1]) if noaa else central_ar(days[1])
    for col, mdi in enumerate(maps):
        mask = smooth_los_threshold(remove_off_disk(mdi), **paper_parameters(mdi))[1]
        center_11 = project(ar, mdi)
        mdi_cut = crop(mdi, center_11, fov_11)
        mask_cut = crop(sunpy.map.Map(mask.astype(float), mdi.meta), center_11, fov_11)

        ax = axes[row, col]
        ax.imshow(mdi_cut.data, origin="lower", cmap="gray", vmin=-1000, vmax=1000)
        ax.contour(mask_cut.data, levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(mdi.date.iso[:10], fontsize=9)
    axes[row, 0].set_ylabel(label, fontsize=10)
fig.suptitle("Higgins et al. (2011) Figure 11 -- detection-only approximation")
fig.tight_layout()

plt.show()
