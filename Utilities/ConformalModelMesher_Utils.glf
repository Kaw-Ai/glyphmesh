#############################################################################
#
# (C) 2024 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

# ===========================================================================
# ConformalModelMesher: Utility Script
# ===========================================================================
# Written by Chris Fouts & Kristen Karman-Shoemake

# Script containing all utility functions required for running
# ConformalModelMesher.glf.

package require PWI_Glyph 8.24.1

# proc to get models from the selection for GUI mode. If no models are
# selected, all visible models are returned.
proc getModels { } {
  # Set the mask for database models and get the entities.
  set mask [pw::Display createSelectionMask -requireDatabase { Models }]
  pw::Display getSelectedEntities -selectionmask $mask results
  set models $results(Databases)

  # Check that models were selected. If not, default to using all visible
  # models.
  if {0 == [llength $models]} {
    set models [pw::Database getAll -type pw::Model -visibleOnly]
    set msg "No models selected. Defaulting to visible models"
  } else {
    set msg "Selected"
  }

  # Check the number of models again and exit if none were found.
  # Otherwise, format a print statement and print to screen.
  set numModels [llength $models]
  if {0 == $numModels} {
    puts "No models are present."
    exit
  } elseif {1 == $numModels} {
    puts "$msg: $numModels model."
  } else {
    puts "$msg: $numModels models."
  }

  # Return the models
  return $models
}

# proc to start the ConformalMesher mode with the given models and attributes
proc startMode { models vals } {
  upvar $vals values
  # Check that models is not empty and return an error if it is.
  if {0 < [llength $models]} {
    # Start the ConformalMesher mode
    set mesher [pw::Application begin ConformalMesher $models]
      # Add each attribute contained in values to the mode
      foreach key [array names values] {
        if {[catch {$mesher set${key} $values($key)}]} {
          error "$key is not a recognized attribute."
        }
      }

      # Update the display
      pw::Display update
  } else {
    error "No models were specified"
  }

  # Return the mode
  return $mesher
}

# proc to generate the surface mesh with the given ConformalMesher mode
proc modeGenerateMesh { mesher } {
  # Print attributes to screen
  puts "maxAng = [$mesher getMaximumAngleDeviation]"
  puts "minLen = [$mesher getMinimumEdgeLength]"
  puts "maxLen = [$mesher getMaximumEdgeLength]"
  puts "maxDev = [$mesher getMaximumAbsoluteDeviation]"

  # Initialize empty return variable, failed entity variable, and start timer
  set doms [list]
  set failQuilts [list]
  set et [pwu::Time now]
  set models [$mesher getEntities]

  # Attempt to create the grid entities. If it fails, print errors to screen.
  if {[catch {$mesher createGridEntities} msg]} {
    puts "$msg"
    foreach model $models {
      set modelResult [$mesher getModelError $model]
      if {0 < [string length $modelResult]} {
        puts "[$model getName]: $modelResult"
      }
    }
  } else {
    set doms $msg
  }

  # Obtain the unmeshed quilts and if any exist, print message to screen with hyperlinks
  set failQuilts [$mesher getUnmeshedQuilts]
  if {0 != [llength $failQuilts]} {
    puts [pw::Script createEntityLink "Unmeshed quilts" $failQuilts]
  }

  # Print timing information to screen
  if {1 == [llength $models]} {
    puts "[[lindex $models 0] getName] meshed in [pwu::Time elapsed $et] seconds"
  } else {
    puts "[llength $models] models meshed in [pwu::Time elapsed $et] seconds"
  }

  # Return the created domains
  return $doms
}

# proc to render the domains using a Shaded style and the models using Wireframe.
proc applyDomainStyle { doms models } {
  if {0 < [llength $doms]} {
    # create a collection for the domains and set the FillMode=Shaded,
    # ColorMode=Entity, and Color= #bbbbbb (gray)
    set col [pw::Collection create]
    $col add $doms
    $col do setRenderAttribute FillMode Shaded
    $col do setRenderAttribute ColorMode Entity
    $col do setColor 0x00bbbbbb
    $col delete

    #create a collection for the models and set the FillMode=None
    set col [pw::Collection create]
    $col add $models
    $col do setRenderAttribute FillMode None
    $col delete
  }
}

# proc to generate the mesh.
proc generateMesh { models vals } {
  # initialize variables
  upvar $vals values
  set doms [list]

  # Check that there are models, if not print message.
  if {0 < [llength $models]} {
    # Start the ConformalMesher mode
    set mesher [startMode $models values]

    # Generate the mesh and end the mode.
    set doms [modeGenerateMesh $mesher]
    $mesher end

    # Update the display of domains and models
    applyDomainStyle $doms $models
  } else {
    puts "No models present."
  }

  # Return the domains
  return $doms
}
