FitSurfacePatch
==================
Copyright 2021 Cadence Design Systems, Inc. All rights reserved worldwide.

A Pointwise Glyph script which creates b-spline surfaces from a set of connected bounding curves
and is fit to a shell (faceted) database.

Usage
~~~~~
This script can only be run interactively.

Select one or more shell database surfaces and one or more sets of connected bounding curves. The script will invoke the surface fitter mode which will create as many surfaces as possible, and fit them using the shell database entities as a point cloud. It is recommended to project or draw the bounding curves onto the shell database entities of interest. The script can be modified to run the surface fitting routine for as many iterations as desired.

.. image:: https://raw.github.com/pointwise/FitSurfacePatch/master/FitSurfacePatchExample.png

Disclaimer
~~~~~~~~~~
This file is licensed under the Cadence Public License Version 1.0 (the "License"), a copy of which is found in the LICENSE file, and is distributed "AS IS." 
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE. 
Please see the License for the full text of applicable terms.
