MyDistribution
===============
Copyright 2021 Cadence Design Systems, Inc. All rights reserved worldwide.

This script demonstrates how you can use the setPoint command to create your own connector grid point distribution function. It's not all that useful right out of the box, but it provides the basis on which you can build a distribution function that better meets your requirements.

The script includes several sample functions that are based on a linear parameter u along the connector's length. There is a simple linear function, square, hyperbolic tangent, and a cubic (shown in the illustration).

As illustrated here, by programming your own distribution function you can cluster points toward the middle of the connector. You can only do this with Pointwise's GUI by inserting break points.

In the image below, the script uses a cubic function to cluster points on the connectors' interior (top) relative to its original state (bottom).

![ScriptImage](https://raw.github.com/pointwise/MyDistribution/master/ScriptImage.png)

Disclaimer
----------
This file is licensed under the Cadence Public License Version 1.0 (the "License"), a copy of which is found in the LICENSE file, and is distributed "AS IS." 
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE. 
Please see the License for the full text of applicable terms.
