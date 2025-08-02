#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample source code is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

#########################################################
# Draws a 2D Fibonacci Spiral: 0, 1, 1, 2, 3, 5, 8, 13...
#########################################################

# I think this version requirement is too strict
package require PWI_Glyph 2

# initialization stuff from the original journal file
pw::Application reset
pw::Application clearModified
# don't show the body axes or nodes
pw::Display setShowBodyAxes 0
pw::Display setShowNodes 0
# how many increments to use for animated drawing of arcs
set arcIncrements 8
# notesize as a fraction of the level 
set noteSizeFraction 0.25
# milliseconds to pause between drawing
set pauseBetweenBoxes 1000
set pauseBetweenArcs 0

###########################################
# Make a line DB entity between two points.
# Note: Arguments are 2D. Add z=0 here.
proc makeDBLine { a b } {
   set seg [pw::SegmentSpline create]
   $seg addPoint [lappend a "0"]
   $seg addPoint [lappend b "0"]
   set line [pw::Curve create]
   $line addSegment $seg
   return line
}

################################################
# Make a circle connector given 2 pts and center
# Note: Arguments are 2D. Add z=0 here.
proc makeConCircle { a b c } {
   set seg [pw::SegmentCircle create]
   $seg addPoint [lappend a "0"]
   $seg addPoint [lappend b "0"]
   $seg setCenterPoint [lappend c "0"]
   set circle [pw::Connector create]
   $circle addSegment $seg
   return circle
}

#################################################################
# Animate making a circle connector given start point and center.
# Draws a 90 arc total.
# Note: Arguments are 2D. Add z=0 here.
proc makeConCircleAnimated { a c l } {
   global arcIncrements
   set dAngle [expr 90.0 / ($arcIncrements*$l-1)]
   # make first arc segment
   set seg [pw::SegmentCircle create]
   $seg addPoint [lappend a "0"]
   $seg addPoint [lappend c "0"]
   $seg setEndAngle $dAngle
   set circle [pw::Connector create]
   $circle addSegment $seg
   # modify arc incrementally
   for {set i 2} {$i < $arcIncrements*$l} {incr i} {
      set mode [pw::Application begin Modify $circle]
         set seg [$circle getSegment -copy 1]
         $seg removeAlternatePoint
         $seg setEndAngle [expr $dAngle * $i]
         $circle replaceAllSegments $seg
      $mode end   
      pw::Display update
   }
   return circle
}

########################################
# Make box label.
proc makeBoxLabel { x y l } {
   global noteSizeFraction
   set ndigits [string length $l]
   set noteSize [expr $l * $noteSizeFraction]
   set M [list [expr $x + $l/2.0 - $ndigits*$noteSize/2.0 ] \
               [expr $y + $l/2.0 -          $noteSize/2.0 ] \
               0 ]
   set Mode [pw::Application begin Create]
      set Note [pw::Note create]
      $Note setText $l
      $Note setPosition $M
      $Note setSize $noteSize
   $Mode end
}

################################
# PART 1 - Draw and label boxes.
################################

set level 1
set x 15
set y  9
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 1
set x 15
set y  8
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 2
set x 13
set y  8
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 3
set x 13
set y 10
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 5
set x 16
set y  8
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 8
set x 13
set y  0
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 13
set x  0
set y  0
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 21
set x  0
set y 13
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

set level 34
set x 21
set y  0
set A [list       $x                $y            ]
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set D [list       $x           [expr $y + $level] ]
set Line1 [makeDBLine $A $B]
set Line2 [makeDBLine $B $C]
set Line3 [makeDBLine $C $D]
set Line4 [makeDBLine $D $A]
makeBoxLabel $x $y $level
pw::Display update
pw::Display resetView -Z
after $pauseBetweenBoxes

#######################
# PART 2 - Draw spiral.
#######################

set level 34
set x 21
set y  0
set A [list [expr $x + $level]       $y           ] 
set B [list       $x           [expr $y + $level] ]
set C [list       $x                 $y           ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 21
set x  0
set y 13
set A [list [expr $x + $level] [expr $y + $level] ] 
set B [list       $x                $y            ]
set C [list [expr $x + $level]       $y           ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 13
set x 0
set y 0
set A [list       $x           [expr $y + $level] ] 
set B [list [expr $x + $level]      $y            ]
set C [list [expr $x + $level] [expr $y + $level] ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 8
set x 13
set y  0
set A [list       $x                 $y           ]
set B [list [expr $x + $level] [expr $y + $level] ]
set C [list       $x           [expr $y + $level] ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 5
set x 16
set y  8
set A [list [expr $x + $level]       $y           ]
set B [list       $x           [expr $y + $level] ]
set C [list       $x                 $y           ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 3
set x 13
set y 10
set A [list [expr $x + $level] [expr $y + $level] ]
set B [list       $x                 $y           ]
set C [list [expr $x + $level]       $y           ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 2
set x 13
set y  8
set A [list       $x           [expr $y + $level] ]
set B [list [expr $x + $level]       $y           ]
set C [list [expr $x + $level] [expr $y + $level] ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 1
set x 15
set y  8
set A [list       $x                 $y           ]
set B [list [expr $x + $level] [expr $y + $level] ]
set C [list       $x           [expr $y + $level] ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set level 1
set x 15
set y  9
set A [list [expr $x + $level]       $y           ]
set B [list       $x           [expr $y + $level] ]
set C [list       $x                 $y           ]
set Circle [makeConCircleAnimated $A $C $level]
pw::Display update
after $pauseBetweenArcs

set Collection [pw::Collection create]
$Collection set [pw::Grid getAll -type pw::Connector]
$Collection do setRenderAttribute LineWidth 3

######
# DONE
######

# display a title
set Mode [pw::Application begin Create]
   set Note [pw::Note create]
   $Note setText "The Fibonacci Spiral"
   $Note setPosition "2 40 0"
   $Note setSize 4
$Mode end

pw::Display resetView -Z

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

