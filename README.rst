ConformalModelMesher
====================
Copyright 2024 Cadence Design Systems, Inc. All rights reserved worldwide.

A Pointwise Glyph script for generating curvature-resolved, conformal surface
meshes on models using the *ConformalMesher* mode. 

Usage
~~~~~

This script can be run either in GUI or batch mode.

**GUI Mode**

+----------+----------------------------------------------------------+
| **Step** | **Notes**                                                |
+==========+==========================================================+
| (*Optional*) Select one or more database models.                    |
+---------------------------------------------------------------------+
| *Script*, *Execute...*, ConformalModelMesher.glf                    |
+----------+----------------------------------------------------------+
|          | If no models are selected, all visible models will be    |
|          | meshed.                                                  |
+----------+----------------------------------------------------------+
| Set meshing parameters in the dialog presented.                     |
+----------+----------------------------------------------------------+
|          | ``MaximumAbsoluteDeviation``: also referred to as "sag"  |
|          | tolerance, this is the maximum absolute deviation, to    |
|          | resolve the geometric curvature based on distance from   |
|          | quadratic face stencil points to the underlying surface. |
+----------+----------------------------------------------------------+
|          | ``MinimumEdgeLength``: this is the minimum desired       |
|          | triangle edge length.                                    |
+----------+----------------------------------------------------------+
|          | ``MaximumEdgeLength``: this is the maximum desired       |
|          | triangle edge length.                                    |
+----------+----------------------------------------------------------+
|          | ``MaximumAngleDeviation``: maximum angular deviation, to |
|          | resolve the geometric curvature based on normal          |
|          | deviation of mesh faces.                                 |
+----------+----------------------------------------------------------+
| Press *Create Mesh*                                                 |
+----------+----------------------------------------------------------+

.. image:: https://raw.github.com/pointwise/ConformalModelMesher/master/ConformalModelMesherUI.PNG

**Batch Mode**

+------------------+-----------------------------------------------------------------------------------------+
| **Invocation**                                                                                             |
+------------------+-----------------------------------------------------------------------------------------+
| **Platform**     | **Command Line**                                                                        |
+==================+=========================================================================================+
| **Linux/Mac OS** | ``${poinwtise_install_path}/pointwise -b ConformalModelMesher.glf ?script_args?``       |
+------------------+-----------------------------------------------------------------------------------------+
| **Windows**      | ``%poinwtise_install_path%/win64/bin/tclsh.exe ConformalModelMesher.glf ?script_args?`` |
+------------------+-----------------------------------------------------------------------------------------+



+---------------------------+-------------------------------------------------+
| **Script Arguments**                                                        |
+---------------------------+-------------------------------------------------+
| **Argument**              | **Description**                                 |
+===========================+=================================================+
| ``-f filename``           | (Required) path to input Pointwise (.pw) file.  |
+---------------------------+-------------------------------------------------+
| ``-o outfilename``        | (Optional) path to output Pointwise (.pw) file. |
|                           | If not specified, the input file will be        |
|                           | overwritten.                                    |
+---------------------------+-------------------------------------------------+
| ``-maxDeviation <float>`` | (Optional) Maximum absolute deviation.          |
+---------------------------+-------------------------------------------------+
| ``-minEdge <float>``      | (Optional) Minimum allowable edge length.       |
+---------------------------+-------------------------------------------------+
| ``-maxEdge <float>``      | (Optional) Maximum allowable edge length.       |
+---------------------------+-------------------------------------------------+
| ``-maxAngle <float>``     | (Optional) Maximum angle deviation.             |
+---------------------------+-------------------------------------------------+
| ``-h`` or ``-help``       | (Optional) Prints usage message and exits.      |
+---------------------------+-------------------------------------------------+

Disclaimer
~~~~~~~~~~
This file is licensed under the Cadence Public License Version 1.0 (the
"License"), a copy of which is found in the LICENSE file, and is distributed
"AS IS." 

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, CADENCE DISCLAIMS ALL
WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO ANY PARTY FOR ANY DAMAGES ARISING
OUT OF OR RELATING TO USE OF THIS FILE.  Please see the License for the full
text of applicable terms.
