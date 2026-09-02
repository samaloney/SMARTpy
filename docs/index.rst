*******************
SMART documentation
*******************

``smart`` is a Python implementation of the **Solar Monitor Active Region
Tracker** (SMART): it detects, characterises, and tracks solar active regions in
line-of-sight photospheric magnetograms.  It is a port of the original IDL
`smart_library <https://github.com/pohuigin/smart_library>`__, following the
method of `Higgins et al. (2011)
<https://doi.org/10.1007/s11207-010-9660-y>`__.

.. warning::

   This package is under active development and the API is not yet stable.

The documentation is organised along the four `Diátaxis
<https://diataxis.fr>`__ axes.

Tutorials
=========

Start here if you are new to ``smart``.  The tutorials walk through a complete
SMART run from a raw magnetogram to a table of active-region properties,
explaining each stage as it goes.

.. toctree::
   :maxdepth: 1

   generated/gallery/index

How-to guides
=============
Task-focused recipes for people who already know the basics.

.. toctree::
   :maxdepth: 1

   how_to/index

Explanation
===========

Background on what SMART does and why: the processing, detection, and
characterisation stages, and how the port relates to the paper and the original
IDL library.

.. toctree::
   :maxdepth: 1

   explanation

Reference
=========

The full API.

.. toctree::
   :maxdepth: 1

   reference/index
   whatsnew/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
