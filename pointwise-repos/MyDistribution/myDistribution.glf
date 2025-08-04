#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

# Distributes points along a connector according to a user-defined function.

package require PWI_Glyph 2.4

# Make sure there's at least one connector.
set numCons [pw::Grid getCount -type pw::Connector]
if { $numCons < 1 } {
   puts "There are no connectors."
   exit
}

if { $numCons > 1 } {
   # Have the user select one dimensioned connector.
   set sm [pw::Display createSelectionMask -requireConnector { Dimensioned }]
   set desc "Select one dimensioned connector."
   set picked [pw::Display selectEntities -description $desc \
                                          -selectionmask $sm \
                                          -single results]
   set Con $results(Connectors)
} else {
   # Use the only connector.
   set Con [lindex [pw::Grid getAll -type pw::Connector] 0]
}

# Get the connector's dimensions.
set Imax [$Con getDimension]
puts "Imax=$Imax"
# Dimension the connector if it's undimensioned.
if { $Imax == 0 } {
   $Con setDimension 21
}

# Compute new grid point locations based on a function of arclength.
for { set i 2 } { $i < $Imax } { incr i } {
   # compute a linear parameter u
   set u [expr 1.0 * ($i - 1) / ($Imax - 1)]
   # compute a function of u
   # set f $u
   # set f [expr $u * $u]
   # set f [expr (1.0 + tanh(4.0 * $u - 2.0 )) / 2.0]
   set f [expr 0.5 + (2.0 * $u - 1.0)**3 / 2.0]
   # evaluate the connector at f
   set P($i) [$Con getXYZ -arc $f]
   # puts "u=$u, P=$P($i)"
}

# Set new grid points on connector.
for { set i 2 } { $i < $Imax } { incr i } {
   $Con setPoint $i $P($i)
}

exit

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
