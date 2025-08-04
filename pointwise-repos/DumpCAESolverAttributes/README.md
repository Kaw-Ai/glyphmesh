# DumpCAESolverAttributes
Copyright 2021 Cadence Design Systems, Inc. All rights reserved worldwide.

A script to dump a table of all attributes for all CAE exporters.

The script demonstrates how to use Glyph and [Glyph Client for Python][GlyphClient] to access CAE exporter attributes.


## Usage
If not in batch mode, this script can only be run on an empty project.

Use the command `tclsh DumpCAESolverAttributes.glf` to run the Glyph version of the script in batch mode.

Use the command `python DumpCAESolverAttributes.py` to run the Python version of the script in batch mode.


### Python Version Usage
The Python version of this script requires [Glyph Client for Python][GlyphClient] and [Pointwise][PW] V18.2 or higher.

The command `python -m pip install pointwise-glyph-client` can be used to install the [Glyph Client for Python][GlyphClient] package.

[GlyphClient]: https://github.com/pointwise/GlyphClientPython
[PW]: https://www.pointwise.com


## Disclaimer
This file is licensed under the Cadence Public License Version 1.0 (the "License"), a copy of which is found in the LICENSE file, and is distributed "AS IS." 
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE. 
Please see the License for the full text of applicable terms.
