#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

###############################################################################
##
## CalcAngleFromPoints.glf
##
## Script to determine the angle between three points
##
###############################################################################

package require PWI_Glyph

if { ! [pw::Application isInteractive] } {
  puts "This script can only be run interactively."
  exit 1
}

set ::Rad2Deg [expr 180.0 / (atan(1) * 4.0)]

# Use a mode for temporary db curve
set mode [pw::Application begin Create]

# Use a thick database line to display the selected points
set curve [pw::Curve create]
$curve setRenderAttribute LineWidth 5
$curve setRenderAttribute ColorMode Entity
$curve setColor "#FFFF00"

set seg [pw::SegmentSpline create]
$curve addSegment $seg

catch {
  set xyz1 [pw::Display selectPoint -description "Select the first end point"]
  $seg addPoint $xyz1
  set xyz2 [pw::Display selectPoint -description "Select the apex point"]
  $seg addPoint $xyz2
  pw::Display update
  set xyz3 [pw::Display selectPoint -description "Select the last end point"]
  $seg addPoint $xyz3
  pw::Display update
  after 500

  # Calculate and print the angle between the three points
  set leg1 [pwu::Vector3 normalize [pwu::Vector3 subtract $xyz1 $xyz2]]
  set leg2 [pwu::Vector3 normalize [pwu::Vector3 subtract $xyz3 $xyz2]]
  set dot [pwu::Vector3 dot $leg1 $leg2]
  puts [format "Angle: %g degrees" [expr acos($dot) * $::Rad2Deg]]
}

$mode abort

#############################################################################
#
# This file is licensed under the Cadence Public License Version 1.0 (the
# "License"), a copy of which is found in the included file named "LICENSE",
# and is distributed "AS IS." TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE
# LAW, CADENCE DISCLAIMS ALL WARRANTIES AND IN NO EVENT SHALL BE LIABLE TO
# ANY PARTY FOR ANY DAMAGES ARISING OUT OF OR RELATING TO USE OF THIS FILE.
# Please see the License for the full text of applicable terms.
#
#############################################################################
