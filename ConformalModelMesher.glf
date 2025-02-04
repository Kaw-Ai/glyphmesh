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
# ConformalModelMesher: Main Script
# ===========================================================================
# Written by Kristen Karman-Shoemake

# Script for generating surface meshes using the ConformalMesher mode.
# This script can be run from the PW GUI or batch on the command line.
#
# To use GUI mode:
#   1) Optional: Select one or more models.
#   2) Run ConformalModelMesher.glf. If no models are selected, all visible
#      models will be meshed.
#   3) Optional: Edit the following attributes in the pop-up window.
#      - MinimumEdgeLength: minimum edge length to use when creating triangles.
#      - MaximumEdgeLength: maximum edge length to use when creating triangles.
#      - MaximumAngleDeviation: maximum angle deviation to use when creating
#           triangles.
#      - MaximumAbsoluteDeviation: maximum absolute deviation of the surface
#           mesh to use when creating triangles. This attribute is sometimes
#           referred to as sag tolerance.
#   4) Press Create Mesh.
#
# Batch mode usage:
#   Linux/MacOS:
#     $pointwise_home_path/pointwise -b ConformalModelMesher.glf ?script_args?
#   Windows:
#     $pointwise_home_path/win64/bin/tclsh.exe ConformalModelMesher.glf ?script_args?
#
#   script_args:
#     -f  fileName      | (Required) Path to input Pointwise file.
#     -o  outFileName   | (Optional) Path to output Pointwise file.
#     -minEdge val      | (Optional) Minimum edge length.
#     -maxEdge val      | (Optional) Maximum edge length.
#     -maxAngle val     | (Optional) Maximum angle deviation.
#     -maxDeviation val | (Optional) Maximum absolute deviation.
#     -h or -help       | Prints this message.

package require PWI_Glyph 8.24.1

# define global variables
global fileName
global outputName
global guiMode
global models
global params
global paramEntries
global advOption
global advFrame

# initialize global variables
set fileName ""
set outputName ""
set models []
array set params {}
array set paramEntries {}

# load ConformalUtils.glf
set utilityScript [file join [file dirname [info script]] Utilities ConformalModelMesher_Utils.glf]
if { [file exists $utilityScript] != 1 } {
  puts "Utility file $utilityScript not found."
  exit
}
source $utilityScript

########################################
# Procs
# ######################################

# proc to parse command line arguments
proc parseCommandLineArgs { argv } {
  # set up the string for the help print statement
  set helpString "\nBatch mode usage: \
            \nLinux/MacOS: \$pointwise_home_path/pointwise -b ConformalModelMesher.glf ?script_args? \
            \nWindows: \$pointwise_home_path/win64/bin/tclsh.exe ConformalModelMesher.glf ?script_args??
            \nscript_args:
            \n\t-f  fileName      | (Required) Path to input Pointwise file.
            \n\t-o  outFileName   | (Optional) Path to output Pointwise file.
            \n\t-minEdge val      | (Optional) Minimum edge length.
            \n\t-maxEdge val      | (Optional) Maximum edge length.
            \n\t-maxAngle val     | (Optional) Maximum angle deviation.
            \n\t-maxDeviation val | (Optional) Maximum absolute deviation.
            \n\t-h or -help       | Prints this message."

  for {set index 0 } {$index < [llength $argv]} {incr index} {
    set arg [lindex $argv $index]
    switch -exact $arg {
      -f {
        # Input File Name: check that the next argument is a valid file name with a .pw
        # extension
        set val [lindex $argv [incr index]]
        if {[file exists $val] != 1} {
          puts "Error: Pointwise file not found: $val"
          exit
        }
        # check extension
        if {[file extension $val] != ".pw"} {
          puts "Error: Incorrect file extension for input Pointwise file: $val"
          exit
        }
        set ::fileName $val
      }
      -o {
        # Output File Name: check that the next argument is a valid path for an output file and
        # that the extension is .pw
        set val [lindex $argv [incr index]]
        set dirName [file dirname $val]
        if {[file exists $dirName] != 1} {
          puts "Error: Output file directory does not exist: $dirName"
          exit
        }
        # check extension
        if {[file extension $val] != ".pw"} {
          puts "Error: Incorrect file extension for output Pointwise file: $val"
          exit
        }
        set ::outputName $val
      }
      -minEdge {
        # CMM Attribute - Minimum Edge Length: check that the next argument is a double
        set val [lindex $argv [incr index]]
        if {![string is double $val]} {
          puts "Error: Incorrect value type for MinimumEdgeLength. value = $val"
          exit
        }
        set ::params(MinimumEdgeLength) $val
      }
      -maxEdge {
        # CMM Attribute - Maximum Edge Length: check that the next argument is a double
        set val [lindex $argv [incr index]]
        if {![string is double $val]} {
          puts "Error: Incorrect value type for MaximumEdgeLength. value = $val"
          exit
        }
        set ::params(MaximumEdgeLength) $val
      }
      -maxAngle {
        # CMM Attribute - MaximumAngleDeviation: check that the next argument is a double
        set val [lindex $argv [incr index]]
        if {![string is double $val]} {
          puts "Error: Incorrect value type for MaximumAngleDeviation. value = $val"
          exit
        }
        set ::params(MaximumAngleDeviation) $val
      }
      -maxDeviation {
        # CMM Attribute - MaximumAbsoluteDeviation: check that the next argument is a double
        set val [lindex $argv [incr index]]
        if {![string is double $val]} {
          puts "Error: Incorrect value type for MaximumAbsoluteDeviation. value = $val"
          exit
        }
        set ::params(MaximumAbsoluteDeviation) $val
      }
      -help {
        # Print help statement
        puts $helpString
        exit
      }
      -h {
        # Print help statement
        puts $helpString
        exit
      }
      default {
        # Default switch statement: Exit with error message if argument is unknown
        puts "Unrecognized argument: $arg"
        puts $helpString
        exit
      }
    }
  }
}

# proc to validate provided values
proc validateEntries { } {
  # validate the provided values
  set valid true

  # validate the minimum edge length, if provided
  if {[info exists ::params(MinimumEdgeLength)]} {
    # check that the type is a number by converting to scientific notation
    if {[catch {set ::params(MinimumEdgeLength) [format %e $::params(MinimumEdgeLength)]}]} {
      puts "Invalid entry for Minimum Edge Length. Value must be of type double."
      set valid false

      # if the mode is GUI, update the entry background color
      if {$::guiMode} {
        $::paramEntries(MinimumEdgeLength) configure -background salmon
      }
    } else {
      # check that the value is in the range (0.0, inf)
      if { $::params(MinimumEdgeLength) <= 0.0 } {
        puts "Invalid entry for Minimum Edge Length. Value must be in range of (0.0, \u221E)."
        set valid false

        # if the mode is GUI, update the entry background color
        if {$::guiMode} {
          $::paramEntries(MinimumEdgeLength) configure -background salmon
        }
      } else {
        # entry is valid. ensure entry has no error highlighting
        if {$::guiMode} {
          $::paramEntries(MinimumEdgeLength) configure -background white
        }
      }
    }
  }

  # validate the maximum edge length, if provided
  if {[info exists ::params(MaximumEdgeLength)]} {
    # check that the type is a number by converting to scientific notation
    if {[catch {set ::params(MaximumEdgeLength) [format %e $::params(MaximumEdgeLength)]}]} {
      puts "Invalid entry for Maximum Edge Length. Value must be of type double."
      set valid false

      # if the mode is GUI, update the entry background color
      if {$::guiMode} {
        $::paramEntries(MaximumEdgeLength) configure -background salmon
      }
    } else {
      # check that the value is in the range (0.0, inf)
      if { $::params(MaximumEdgeLength) < 0.0 } {
        puts "Invalid entry for Maximum Edge Length. Value must be in range of \[0.0, \u221E)."
        set valid false

        # if the mode is GUI, update the entry background color
        if {$::guiMode} {
          $::paramEntries(MaximumEdgeLength) configure -background salmon
        }
      } else {
        # entry is valid. ensure entry has no error highlighting
        if {$::guiMode} {
          $::paramEntries(MaximumEdgeLength) configure -background white
        }
      }
    }
  }

  # validate the maximum angle deviation, if provided
  if {[info exists ::params(MaximumAngleDeviation)]} {
    if { !([string is integer $::params(MaximumAngleDeviation)] || \
           [string is double $::params(MaximumAngleDeviation)]) || \
         $::params(MaximumAngleDeviation) < 0.0 || \
         $::params(MaximumAngleDeviation) >= 180.0 } {
      puts "Invalid entry for Maximum Angle Deviation. Value must be double in range of \[0.0, 180.0)."

      # if the mode is GUI, update the entry background color
      if {$::guiMode} {
        $::paramEntries(MaximumAngleDeviation) configure -background salmon
      }
      set valid false
    } else {
      # entry is valid. ensure entry has no error highlighting
      if {$::guiMode} {
        $::paramEntries(MaximumAngleDeviation) configure -background white
      }
    }
  }

  # validate the minimum edge length, if provided
  if {[info exists ::params(MaximumAbsoluteDeviation)]} {
    # check that the type is a number by converting to scientific notation
    if {[catch {set ::params(MaximumAbsoluteDeviation) [format %e $::params(MaximumAbsoluteDeviation)]}]} {
      puts "Invalid entry for Maximum Absolute Deviation. Value must be of type double."
      set valid false

      # if the mode is GUI, update the entry background color
      if {$::guiMode} {
        $::paramEntries(MaximumAbsoluteDeviation) configure -background salmon
      }
    } else {
      # check that the value is in the range (0.0, inf)
      if { $::params(MaximumAbsoluteDeviation) < 0.0 } {
        puts "Invalid entry for Maximum Absolute Deviation. Value must be in range of \[0.0, \u221E)."
        set valid false

        # if the mode is GUI, update the entry background color
        if {$::guiMode} {
          $::paramEntries(MaximumAbsoluteDeviation) configure -background salmon
        }
      } else {
        # entry is valid. ensure entry has no error highlighting
        if {$::guiMode} {
          $::paramEntries(MaximumAbsoluteDeviation) configure -background white
        }
      }
    }
  }

  return $valid
}

# proc for batch mode to create the surface mesh using conformal model meshing (CMM)
proc meshBatch {} {
  # validate the input parameters. If invalid values are detected,
  # exit with error message.
  if { ![validateEntries] } {
    puts "Error: Invalid entries were detected. Re-enter values and try again."
    exit
  }

  # mesh the models using CMM
  set doms [generateMesh $::models ::params]
}

# proc for GUI mode to create surface mesh using conformal model meshing (CMM)
proc meshGUI { } {
  # validate the input parameters. If invalid values are detected,
  # return control to the window to re-enter values
  if { ![validateEntries] } {
    puts "Invalid entries were detected. Re-enter values and try again."
    return
  }

  # hide the pop up window for gui mode
  wm iconify $::t1

  # mesh the models using CMM
  set doms [generateMesh $::models ::params]

  # exit the script here for gui mode
  exit
}

proc processAdvanced { } {
  global advOption
  global advFrame
  if {$advOption} {
    pack $advFrame
  } else {
    pack forget $advFrame
  }
}


################################
# Main
# ##############################

# check for batch operation
set guiMode [pw::Application isInteractive]
if { !($guiMode) } {
  ################################
  # Batch mode

  # parse command line arguments
  parseCommandLineArgs $argv
  puts "\n****************************** \
        \nSummary of Input Options:"

  # check that a pw file was provided
  if { $fileName == "" } {
    puts "Error: No Pointwise file provided. Please try again using the -f <fileName> option."
    exit
  }
  puts "\nPointwise file: $fileName"

  # check outputName. If none was provided, set to default value based on fileName
  if {$outputName == "" } {
    set baseName [file rootname $fileName]
    set outputName "${baseName}_CMM.pw"
  }
  puts "Output File Name: $outputName"

  # print parameters to screen
  puts "\nOptional Parameters:"
  parray params

  puts "\n****************************** \
        \nLoading file $fileName ..."
  set projectLoader [pw::Application begin ProjectLoader]
  $projectLoader initialize $fileName
  $projectLoader load
  $projectLoader end
  unset projectLoader

  # get models
  puts "\nGetting models for meshing ..."
  set models [pw::Database getAll -type pw::Model -visibleOnly]
  if { 0 == [llength $models] } {
    puts "No models are present."
    exit
  }

  # create mesh
  puts "\nCreating mesh ..."
  meshBatch

  # save mesh
  puts "\bSaving file $outputName"
  pw::Application save $outputName

} else {

  ############################
  # GUI mode

  # load Tk and create GUI
  pw::Script loadTk

  # initialize parameters with default values
  set defMinEL [expr {[pw::Database getModelSize]*1.0e-8}]
  set params(MinimumEdgeLength)        $defMinEL
  set params(MaximumEdgeLength)        0.0
  set params(MaximumAngleDeviation)    0.0
  set params(MaximumAbsoluteDeviation) 0.0

  # get all models
  set models [getModels]

  ##################################################
  # create window for getting user defined parameters
  set t1 .
  wm title . "Conformal Model Meshing"
  wm withdraw $t1

  wm attributes . -topmost yes ; # keep window on top

  # add two frames: f1 to house the parameters; f2 to house the buttons
  set f1 [frame .f1 -borderwidth 1 -relief sunken]
  pack $f1 -padx 5 -pady 5
  set f2 [frame .f2]
  pack $f2 -padx 5 -pady 5

  # create label, entry, and range label for Maximum Absolute Deviation
  set maxAbsDev_Label [label $f1.maxAbsDev_label -text "Maximum Absolute Deviation:" -anchor e]
  set paramEntries(MaximumAbsoluteDeviation) [entry $f1.maxAbsDev_entry \
    -textvariable params(MaximumAbsoluteDeviation) ]
  set maxAbsDev_Range [label $f1.maxAbsDev_range \
    -text "Range \u2208 \[0.0, \u221E), 0 disables limit" -anchor w]

  set advOption 0
  set adv [checkbutton $f1.adv -text Advanced -command processAdvanced -variable advOption]
  set f11 [labelframe $f1.f1 -labelwidget $adv]
  set filler [frame $f11.filler -pady 0]
  pack $filler -side top -expand 0
  set advFrame [frame $f11.advFrame -padx 0 -pady 0]
  if {$advOption} {
    processAdvanced
  }
  # create label, entry, and range label for Minimum Edge Length
  set minEL_Label [label $advFrame.minEL_label -text "Minimum Edge Length:" -anchor e]
  set paramEntries(MinimumEdgeLength) [entry $advFrame.minEL_entry \
    -textvariable params(MinimumEdgeLength) ]
  set minEL_Range [label $advFrame.minEL_range \
    -text "Range \u2208 (0.0, \u221E)" -anchor w]

  # create label, entry, and range label for Maximum Edge Length
  set maxEL_Label [label $advFrame.maxEL_label -text "Maximum Edge Length:" -anchor e]
  set paramEntries(MaximumEdgeLength) [entry $advFrame.maxEL_entry \
    -textvariable params(MaximumEdgeLength) ]
  set maxEL_Range [label $advFrame.maxEL_range \
    -text "Range \u2208 \[0.0, \u221E), 0 disables limit" -anchor w]

  # create label, entry, and range label for Maximum Angle Deviation
  set maxAngleDev_Label [label $advFrame.maxAngleDev_label -text "Maximum Angle Deviation:" -anchor e]
  set paramEntries(MaximumAngleDeviation) [entry $advFrame.maxAngleDev_entry \
    -textvariable params(MaximumAngleDeviation) ]
  set maxAngleDev_Range [label $advFrame.maxAngleDev_range \
    -text "Range \u2208 \[0.0, 180.0), 0 disables limit" -anchor w]


  # create two buttons:
  #   OK <Return> - runs command meshGUI with provided parameters
  #   Cancel <Esc> - exits script
  set okayButton [button $f2.okay -padx 3 -text "Create Mesh" -command meshGUI]
  set exitButton [button $f2.exit -padx 3 -text "Cancel" -command exit]
  bind . <Return> {meshGUI}
  bind . <Escape> {exit}

  # place all labels and entry fields in the grid
  grid $maxAbsDev_Label -column 0 -row 0 -sticky nse -pady 5 -padx 2
  grid $paramEntries(MaximumAbsoluteDeviation) -column 1 -row 0 -sticky nsew -pady 5 -padx 2
  grid $maxAbsDev_Range -column 2 -row 0 -sticky nsw -pady 5 -padx 2

  grid $f11 -column 0 -columnspan 3 -row 1 -sticky nsew -pady 0 -padx 0

  grid $minEL_Label -column 0 -row 0 -sticky nse -pady 5 -padx 2
  grid $paramEntries(MinimumEdgeLength) -column 1 -row 0 -sticky nsew -pady 5 -padx 2
  grid $minEL_Range -column 2 -row 0 -sticky nsw -pady 5 -padx 2

  grid $maxEL_Label -column 0 -row 1 -sticky nse -pady 5 -padx 2
  grid $paramEntries(MaximumEdgeLength) -column 1 -row 1 -sticky nsew -pady 5 -padx 2
  grid $maxEL_Range -column 2 -row 1 -sticky nsw -pady 5 -padx 2

  grid $maxAngleDev_Label -column 0 -row 2 -sticky nse -pady 5 -padx 2
  grid $paramEntries(MaximumAngleDeviation) -column 1 -row 2 -sticky nsew -pady 5 -padx 2
  grid $maxAngleDev_Range -column 2 -row 2 -sticky nsw -pady 5 -padx 2

  # place both buttons in the grid
  grid $okayButton -column 0 -row 0 -padx 10 -pady 10
  grid $exitButton -column 1 -row 0 -padx 10 -pady 10

  wm deiconify $t1
}
