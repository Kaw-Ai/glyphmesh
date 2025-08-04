#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Description:
#
# FitSurfacePatch
#
# This Glyph script demonstrates how to use the pw::SurfaceFitter mode to
# create a b-spline database surface fitted to a point cloud defined by an
# underlying database shell.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# --------------------------------------------------------------------------
# Initialization

# Load Pointwise Glyph package
package require PWI_Glyph 2

set FitIterations 1


# --------------------------------------------------------------------------
# Select entities of a given entity type
#
proc selectEntities { entityType prompt } {

  # Define the selection mask
  set selectionMask [pw::Display createSelectionMask -requireDatabase $entityType \
        -requireDatabaseBoundary {}]

  # Interactive selection
  pw::Display selectEntities -description $prompt -selectionmask $selectionMask \
        selectionArray

  set selection [list]
  if {$entityType eq "Curves"} {
    if {[llength $selectionArray(Databases)] > 0} {
      lappend selection $selectionArray(Databases)
    }
    if {[llength $selectionArray(Boundaries)] > 0} {
      lappend selection $selectionArray(Boundaries)
    }
  } elseif {$entityType eq "Shells"} {
    if {[llength $selectionArray(Databases)] > 0} {
      lappend selection $selectionArray(Databases)
    }
  }
  return [join $selection]
}


# --------------------------------------------------------------------------
# Main script body

# Select the database shell used to define the point cloud for fitting purposes
set prompt "Select the underlying database shell(s)...\n"
set shells [selectEntities Shells $prompt]
if {[llength $shells] == 0} {
  puts "ERROR: No database shell entities selected!"
  exit 1
}

# Select the database curves/boundaries bounding the new surface to be created
set prompt "Select bounding database curves and/or boundaries...\n"
set boundingEnts [selectEntities Curves $prompt]
if {[llength $boundingEnts] == 0} {
  puts "ERROR: No bounding database curves/boundaries selected!"
  exit 1
}

# Begin the SurfaceFit mode
set FitterMode [pw::Application begin SurfaceFit]

  # Create the new surface(s)
  if { [catch { $FitterMode createSurfacesFromCurves -fitThreshold 0.5 $boundingEnts } newSurfaces] } {
    puts "ERROR: Could not create database surface from bounding curves\n$newSurfaces"
    $FitterMode abort
    exit 1
  } elseif { [llength $newSurfaces] == 0 } {
    puts "ERROR: Could not create any database surface from bounding curves"
    $FitterMode abort
    exit 1
  }

  # Set the underlying database shell(s) that will be used for fitting purposes
  foreach surface $newSurfaces {
    if { [catch { $FitterMode setFitEntities $surface $shells } msg] } {
      puts "ERROR: Could not set fit entities\n$msg"
      $FitterMode abort
      exit 1
    }
  }

  # Run the surface fitter for the specified number of iterations
  puts [format "Running surface fitter for %d iteration%s..." $FitIterations \
    [expr $FitIterations == 1 ? "{}" : "{s}"]]
  if { [catch { $FitterMode run $FitIterations } msg] } {
    puts "ERROR: Could not fit surface:\n$msg"
    $FitterMode abort
    exit 1
  }

  set maxRMS 0.0
  foreach surface $newSurfaces {
    set rms [$FitterMode getRMSDistance $surface]
    set maxRMS [expr max($maxRMS,$rms)]
  }

  puts [format "Maximum RMS distance is %g" $maxRMS]

$FitterMode end

# Print out a link to the new surface that was created

set newSurfacesCount [llength $newSurfaces]
set linkMsg [format "%d surface%s" $newSurfacesCount [expr $newSurfacesCount == 1 ? "{}" : "{s}"]]
set link [pw::Script createEntityLink -title "Click to select" $linkMsg $newSurfaces]
puts "Created $link"

exit 0

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
