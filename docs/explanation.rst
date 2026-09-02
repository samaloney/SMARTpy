.. _explanation:

****************
The SMART method
****************

SMART (the Solar Monitor Active Region Tracker) turns a line-of-sight
photospheric magnetogram into a catalogue of solar active regions, each with a
set of magnetic properties, and follows those regions from one magnetogram to
the next.  It was introduced by `Higgins et al. (2011)
<https://doi.org/10.1007/s11207-010-9660-y>`__ and implemented in the IDL
`smart_library <https://github.com/pohuigin/smart_library>`__ (with shared
utilities in `gen_library <https://github.com/pohuigin/gen_library>`__).  This
package is a Python port of that library.

Pipeline overview
=================

A SMART run has four stages.

1. Magnetogram processing
-------------------------

The raw magnetogram is prepared for detection: off-limb pixels are removed, the
line-of-sight field is deprojected to the radial field with a cosine
correction (valid away from the limb, where the correction diverges and is
capped), a noise threshold is applied, and the data is smoothed.

Ported in :mod:`smart.processing` (``map_threshold``,
``calculate_cosine_correction``, ``cosine_correct_data``,
``smooth_los_threshold``, ``smart_prep``).

2. Detection
------------

In the IDL library the primary detection (``ar_detect``) is two-stage:

- a **smoothed detection** — the processed magnetogram is Gaussian-smoothed with
  a kernel the size of a supergranule (``smoothphys`` = 16 Mm) and thresholded
  at a low value (``smooththresh``; 15 G for HMI).  This picks out the coherent,
  active-region-scale field.
- a **fragment mask and region grow** — a separately processed, *un*-deprojected
  magnetogram is thresholded at a high value (``magthresh``; 350 G for HMI) to
  find compact flux fragments, which are dilated and then region-grown outward
  from the smoothed detections.  This reconnects fragments and bridges bipoles
  split by a wide polarity separation.

Detections are then labelled and ordered by area, largest first
(``ar_order_mask``).

A separate pass (``ar_detect_core``) finds the strong-gradient **core** of each
region around its polarity inversion line, for flare-relevant analysis.

In the Python port this is currently a single threshold on the deprojected data
followed by morphological dilation and a size filter
(:func:`smart.processing.smooth_los_threshold`).  The supergranule-scale
smoothed detection, the high-threshold fragment/region-grow step, and the core
detection are **not yet ported**.

3. Transient removal via differential rotation
---------------------------------------------------

A magnetogram from an earlier time is differentially rotated forward to the
current time and its detection mask is compared with the current one.  Features
that are not present in both are treated as transient and dropped; surviving
regions keep a consistent index.

Ported in :mod:`smart.differential_rotation` and
:func:`smart.indexed_grown_mask.index_and_grow_mask`.

4. Characterisation
-------------------

Each detected region is measured: cosine-corrected area (also in millionths of a
solar hemisphere), positive / negative / unsigned magnetic flux and flux
imbalance, field-strength statistics, and the flux emergence rate from the
change in field between two times.  Polarity-separation-line length and
strong-gradient PSL length are computed from the polarity masks.

Ported in :mod:`smart.calculate_properties` and
:mod:`smart.polarity_separation_line`.  Not yet ported: heliographic position,
bipole separation, the Schrijver *R* value, the Falconer WLSG, magnetic moments,
and chain-code region boundaries.

5. Tracking
-----------

The IDL library tracks regions across a time series with YAFTA
(``ar_track_yafta``) so that a region keeps the same identifier for its
lifetime, and records merge / split events.  Tracking and the HEK/JSON export
(``ar_smart2hek``) are **not yet ported**.

Port status
===========

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Stage
     - IDL routines
     - Port status
   * - Magnetogram processing
     - ``ar_processmag``, ``ar_cosmap``, ``ar_coscorlim``
     - Mostly ported
   * - Detection
     - ``ar_detect``, ``ar_grow``, ``ar_order_mask``
     - Simplified (single threshold)
   * - Core detection
     - ``ar_detect_core``, ``ar_pslmask``, ``ar_ridgemask``
     - Not started
   * - Transient removal
     - differential rotation in ``ar_processmag``
     - Ported
   * - Characterisation
     - ``ar_detstr2arstr``, ``ar_losgrad``, ``ar_bipolesep``
     - Partial
   * - Tracking / output
     - ``ar_track_yafta``, ``ar_smart2hek``
     - Not started
