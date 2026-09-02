Run the full SMART pipeline in one call
=======================================

Goal: go straight from two magnetograms to a list of active-region properties,
without driving each stage yourself.

.. code-block:: python

    from sunpy.map import Map

    from smart.calculate_properties import smart_indentify_and_characterize

    current = Map("hmi.M_720s.20240606_230000_TAI.fits")
    earlier = Map("hmi.M_720s.20240606_000000_TAI.fits")

    properties = smart_indentify_and_characterize(current, earlier)

    for region in properties:
        for name, value in region.items():
            print(f"{name}: {value}")
        print()

``smart_indentify_and_characterize`` runs, in order:

#. :func:`smart.processing.smart_prep` on both maps (threshold, smooth,
   cosine-correct);
#. :func:`smart.differential_rotation.diff_rotation` to bring the earlier map to
   the current time;
#. :func:`smart.indexed_grown_mask.index_and_grow_mask` to build and index the
   detection mask and drop transient features;
#. :func:`smart.calculate_properties.dB_dt` for the field-change map;
#. :func:`smart.calculate_properties.get_properties` for the per-region table.

To customise a stage (a different threshold, your own detection mask), call the
steps individually instead — see the :doc:`example gallery
</generated/gallery/index>`.

The two magnetograms should be far enough apart in time for real evolution to
show above the noise; a few hours to a day is typical.
