#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

#############################################################################
##
## CAE Exporter Attribute Dump
##
## Dumps a table of all attributes for all CAE exporters.
##
#############################################################################

package require PWI_Glyph 2.17.1

set hr  "--------------------------------------------------"


# Return display format
proc buildFmt { attrNames } {
    set maxLen 20 ;# min col width
    foreach name $attrNames {
        set len [string length $name]
        if { $len > $maxLen } {
            set maxLen $len
        }
    }
    set baseFmt "| %%20.20s | %%-%d.%ds | %%s"
    return [format $baseFmt $maxLen $maxLen]
}


# Display attributes of given solver and dimension
proc dumpSolver { solverName dim } {
    global hr
    if { ![catch "pw::Application setCAESolver \"$solverName\" $dim"] } {
        set attrNames [pw::Application getCAESolverAttributeNames -fullname -group All]
        if { 0 != [llength $attrNames] } {
            set fmt [buildFmt $attrNames]
            puts "${dim}D Attributes:"
            puts [format $fmt "Value" "Name" "Definition (type access desc value range)"]
            puts [format $fmt $hr $hr $hr]
            foreach name [lsort $attrNames] {
                set val [pw::Application getCAESolverAttribute $name]
                set defn [pw::Application getCAESolverAttributeDefinition $name]
                puts [format $fmt $val $name [lrange $defn 2 end]]
            }
        } else {
            puts "${dim}D Attributes: NONE"
        }
    }
}


# Main
# Check for existing entities
if { ! [pw::Application isEmpty] } {
    puts "This script can only be run on an empty project"
    exit 1
}

set currentSolver [pw::Application getCAESolver]
set solverNames [pw::Application getCAESolverNames]
foreach solverName $solverNames {
    puts "$solverName"
    puts "============================================"
    dumpSolver $solverName 3
    dumpSolver $solverName 2
    puts ""
}
pw::Application setCAESolver $currentSolver


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
