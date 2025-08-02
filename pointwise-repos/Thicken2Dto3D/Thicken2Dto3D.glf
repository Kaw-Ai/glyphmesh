#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################


# ===============================================
# THICKEN 2D GRID SCRIPT - POINTWISE
# ===============================================
# https://github.com/pointwise/Thicken2D
#
# Vnn: Release Date / Author
# v01: Nov 23, 2013 / David Garlisch
#
# ===============================================

if { ![namespace exists pw::Thicken2D] } {

package require PWI_Glyph


#####################################################################
#                       public namespace procs
#####################################################################
namespace eval pw::Thicken2D {
  namespace export setVerbose
  namespace export setExtDirection
  namespace export setExtDistance
  namespace export setExtSteps
  namespace export setMinSidewallBC
  namespace export setMaxSidewallBC
  namespace export setSidewallBC
  namespace export thicken
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setVerbose { val } {
  variable verbose
  set verbose $val
  traceMsg "Setting verbose = $verbose."
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setExtDirection { val } {
  if { 3 != [llength $val]} {
	set val {0 0 1}
  }
  variable extDirection
  set extDirection $val
  traceMsg "Setting extDirection = \{$val\}."
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setExtDistance { val } {
  if { 0.0 >= $val} {
	set val 1.0
  }
  variable extDistance
  set extDistance $val
  traceMsg "Setting extDistance = $val."
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setExtSteps { val } {
  if { 0 >= $val} {
	set val 1
  }
  variable extNumSteps
  set extNumSteps $val
  traceMsg "Setting extNumSteps = $val."
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setMinSidewallBC { solverName bcName bcType {bcId "null"} } {
  setSidewallBC $solverName $bcName $bcType $bcId "min"
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setMaxSidewallBC { solverName bcName bcType {bcId "null"} } {
  setSidewallBC $solverName $bcName $bcType $bcId "max"
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::setSidewallBC { solverName {bcName "Unspecified"} {bcType "Unspecified"} {bcId "null"} {minMax "both"} } {
  if { -1 == [lsearch -exact [pw::Application getCAESolverNames] $solverName] } {
	fatalMsg "Invalid solverName='$solverName' in setSidewallBC!"
  }
  switch $minMax {
  min -
  max {
    set key "$solverName,$minMax"
    set pattern $key
  }
  both {
    set key $solverName
    set pattern "$key*"
  }
  default {
	fatalMsg "Invalid minMax='$minMax' in setSidewallBC!"
  }
  }

  variable extSideWallBCInfo
  if { "Unspecified" == $bcName } {
    array unset extSideWallBCInfo $pattern
    traceMsg "Removing Side Wall BC Info for '$key'."
  } else {
    set extSideWallBCInfo($key) [list $bcName $bcType $bcId]
    traceMsg "Adding extSideWallBCInfo($key) = \{'$bcName' '$bcType' '$bcId'\}."
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::thicken { domsToThicken } {
  if { 2 != [pw::Application getCAESolverDimension] } {
	fatalMsg "This script requires a 2D grid."
  }

  init

  puts "**** Preprocessing 2D grid..."

  array set con2DomsMap {} ;# maps a con from 2D grid to its doms
  array set reg2BCMap {}   ;# maps a 2D register to its BC

  # Process the 2D grid's connectors
  foreach con [pw::Grid getAll -type pw::Connector] {
    set doms [pw::Domain getDomainsFromConnectors [list $con]]
    foreach dom $doms {
      set bc [pw::BoundaryCondition getByEntities [list [list $dom $con]]]
      if { [$bc getName] == "Unspecified" } {
	# skip registers without a named BC applied
	continue
      }
      traceMsg "\{[$dom getName] [$con getName]\} has bc [bcToString $bc]"
      set reg2BCMap($dom,$con) $bc
    }
    if { 0 != [llength [array get reg2BCMap "*,$con"]] } {
      # con had at least one BC. Save the $con to $doms mapping.
      set con2DomsMap($con) $doms
    }
  }

  # Capture the list of connectors that had BCs applied
  set bcCons [array names con2DomsMap]

  puts "**** Converting to a 3D grid..."

  # switch current solver to 3D mode
  variable solverName
  pw::Application setCAESolver $solverName 3
  traceMsg "Solver '$solverName' switched to 3D mode."

  # sort list of domains - needed for lsearch
  set domsToThicken [lsort $domsToThicken]

  foreach dom $domsToThicken {
	extrudeDomain $dom
  }

  puts "**** Transferring BCs to the extruded domains..."

  # Process original BC connectors and transfer the BC to the extruded domain.
  foreach bcCon $bcCons {
    # Get the one or two domains from the original 2D grid that were on either
    # side of $bcCon. These domains were extuded to blocks in the 3D grid.
    set bcDoms $con2DomsMap($bcCon)

    # Get the domain ($extrudedDom) that was created by the extrusion of $bcCon.
    if { [getExtrudedDom $domsToThicken $bcCon extrudedDom] } {
      # drop through
    } elseif { [isInternalCon $con $bcDoms] } {
      warningMsg "Skipping internal connector [$bcCon getName]!"
      continue
    } else {
      fatalMsg "Could not find extruded domain for [$bcCon getName]!"
    }
    traceMsg "Move BC from [$bcCon getName] to [$extrudedDom getName]"
    foreach bcDom $bcDoms {
      # Get the block ($extrudedBlk) that was created by the extrusion of $bcDom.
      if { ![getExtrudedBlk $bcDom extrudedBlk] } {
	    fatalMsg "Could not find extruded block for [$bcDom getName]!"
      }
      # Get the BC associated with the 2D register
      if { [getRegBC reg2BCMap $bcDom $bcCon bc] } {
	    # The BC on the 2D register {$bcDom $bcCon} must be transferred to the 3D
	    # register {$extrudedBlk $extrudedDom}.
	    $bc apply [list [list $extrudedBlk $extrudedDom]]
      }
    }
  }
}


#####################################################################
#               private namespace procs and variables
#####################################################################
namespace eval pw::Thicken2D {

  # Set to 0/1 to disable/enable TRACE messages
  variable verbose 0

  # Controls extrusion direction
  variable extDirection {0 0 1}

  # Controls extrusion distance
  variable extDistance 1

  # Controls extrusion number of steps
  variable extNumSteps 1

  # Controls which BCs are used for extrusion base/top domains
  #   * Use SolverName entry to specify same BC for both base and top.
  #   * Use SolverName,min entry to specify base BC.
  #   * Use SolverName,max entry to specify top BC.
  # BCs are applied to the side wall domains ONLY if the solver is found
  variable extSideWallBCInfo
  array set extSideWallBCInfo {} ;#{"BCName" "BCType" Id}

  # BC applied to min (base) doms in extruded blocks.
  variable bcExtrusionBase "null"

  # BC applied to max (top) doms in extruded blocks.
  variable bcExtrusionTop "null"

  # Active CAE solver name
  variable solverName ""
}

#----------------------------------------------------------------------------
proc pw::Thicken2D::init {} {
  variable solverName
  set solverName [pw::Application getCAESolver]

  puts "**** Initializing namespace pw::Thicken2D ..."

  variable extSideWallBCInfo
  variable bcExtrusionBase
  variable bcExtrusionTop
  if { "" != [array names extSideWallBCInfo -exact $solverName] } {
    traceMsg "Same BC used for both side walls."
    lassign $extSideWallBCInfo($solverName) bcName bcType bcId
    set bcExtrusionBase [getMinMaxBC $bcName $bcType $bcId]
    set bcExtrusionTop $bcExtrusionBase
  } else {
    if { "" != [array names extSideWallBCInfo -exact "$solverName,min"] } {
      traceMsg "Using min side wall BC."
      lassign $extSideWallBCInfo($solverName,min) bcName bcType bcId
      set bcExtrusionBase [getMinMaxBC $bcName $bcType $bcId]
    }
    if { "" != [array names extSideWallBCInfo -exact "$solverName,max"] } {
      traceMsg "Using max side wall BC."
      lassign $extSideWallBCInfo($solverName,max) bcName bcType bcId
      set bcExtrusionTop [getMinMaxBC $bcName $bcType $bcId]
    }
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::fatalMsg { msg {exitCode -1} } {
  puts "  ERROR: $msg"
  exit $exitCode
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::warningMsg { msg {exitCode -1} } {
  puts "  WARNING: $msg"
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::traceMsg { msg } {
  variable verbose
  if { $verbose } {
    puts "  TRACE: $msg"
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::getMinMaxBC { bcName {physType null} {id null} } {
  if { [catch {pw::BoundaryCondition getByName $bcName} bc] } {
    traceMsg "Creating new BC('$bcName' '$physType' $id)."
    set bc [pw::BoundaryCondition create]
    $bc setName $bcName
    if { "null" != $physType } {
      $bc setPhysicalType $physType
    }
    if { "null" != $id } {
      $bc setId $id
    }
  } else {
    traceMsg "Found existing BC '$bcName'."
  }
  return $bc
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::bcToString { bc } {
  return "\{'[$bc getName]' '[$bc getPhysicalType]' [$bc getId]\}"
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::edgeContainsCon { edge con } {
  set ret 0
  set cnt [$edge getConnectorCount]
  for {set ii 1} {$ii <= $cnt} {incr ii} {
    if { "$con" == "[$edge getConnector $ii]" } {
      set ret 1
      break
    }
  }
  return $ret
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::isInternalCon { con doms } {
  foreach dom $doms {
    set cnt [$dom getEdgeCount]
    # edge 1 is ALWAYS the outer edge so we can skip it
    for {set ii 2} {$ii <= $cnt} {incr ii} {
      if { [edgeContainsCon [$dom getEdge $ii] $con] } {
	return 1
      }
    }
  }
  return 0
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::traceBlockFace { blk faceId } {
  if { [catch {[$blk getFace $faceId] getDomains} doms] } {
    traceMsg "  Bad faceid = $faceId"
  } else {
    foreach dom $doms {
      traceMsg "  $faceId = '[$dom getName]'"
    }
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::traceBlockFaces { blk } {
  traceMsg "BLOCK '[$blk getName]'"
  set cnt [$blk getFaceCount]
  for {set ii 1} {$ii <= $cnt} {incr ii} {
    traceBlockFace $blk $ii
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::extrudeDomain { dom } {
  set createMode [pw::Application begin Create]
    if { [$dom isOfType pw::DomainStructured] } {
      set face0 [lindex [pw::FaceStructured createFromDomains [list $dom]] 0]
      set blk [pw::BlockStructured create]
      set topFaceId KMaximum
    } else {
      set face0 [lindex [pw::FaceUnstructured createFromDomains [list $dom]] 0]
      set blk [pw::BlockExtruded create]
      set topFaceId JMaximum
    }
    $blk addFace $face0
  $createMode end
  unset createMode

  variable extDirection
  variable extDistance
  variable extNumSteps

  set solverMode [pw::Application begin ExtrusionSolver [list $blk]]
    $solverMode setKeepFailingStep true
    $blk setExtrusionSolverAttribute Mode Translate
    $blk setExtrusionSolverAttribute TranslateDirection $extDirection
    $blk setExtrusionSolverAttribute TranslateDistance $extDistance
    $solverMode run $extNumSteps
  $solverMode end
  unset solverMode
  unset face0
  traceMsg "----"
  traceMsg "Domain '[$dom getName]' extruded into block '[$blk getName]'"

  # BUG WORKAROUND - extruded block JMaximum is returning wrong face
  # FIXED in 17.1R5
  if { ![$dom isOfType pw::DomainStructured] } {
    set topFaceId [$blk getFaceCount]
  }

  variable bcExtrusionBase
  variable bcExtrusionTop
  if { "null" != $bcExtrusionBase } {
    $bcExtrusionBase apply [list [list $blk $dom]]
    traceMsg "Applied base BC '[$bcExtrusionBase getName]' to '[$dom getName]'"
  }

  if { "null" != $bcExtrusionTop } {
    set topDoms [[$blk getFace $topFaceId] getDomains]
    foreach topDom $topDoms {
      $bcExtrusionTop apply [list [list $blk $topDom]]
      traceMsg "Applied base BC '[$bcExtrusionTop getName]' to '[$topDom getName]'"
    }
  }
  traceBlockFaces $blk
  return $blk
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::getExtrudedDom { domsToThicken fromCon domVarName } {
  upvar $domVarName dom

  # get all domains using the current connector
  set doms [pw::Domain getDomainsFromConnectors [list $fromCon]]
  set foundDom 0
  foreach dom $doms {
    if { -1 == [lsearch -sorted $domsToThicken $dom] } {
      # $dom was NOT in the original 2D grid, it MUST have been extruded from
      # the original 2D connector $fromCon.
      set foundDom 1
      break
    }
  }
  return $foundDom
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::getExtrudedBlk { fromDom blkVarName } {
  upvar $blkVarName blk
  set ret 0
  set blocks [pw::Block getBlocksFromDomains [list $fromDom]]
  if { 1 == [llength $blocks] } {
    set blk [lindex $blocks 0]
    set ret 1
  }
  return $ret
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::getRegBC { mapName dom con bcVarName } {
  upvar $mapName reg2BCMap
  upvar $bcVarName bc
  set ret 0
  set pairs [array get reg2BCMap "$dom,$con"]
  if { 2 == [llength $pairs] } {
    set bc [lindex $pairs 1]
    set ret 1
  }
  return $ret
}



#####################################################################
#                       private namespace GUI procs
#####################################################################
namespace eval pw::Thicken2D::gui {

  namespace import ::pw::Thicken2D::*

  variable bcNames
  set bcNames [pw::BoundaryCondition getNames]

  variable bcNamesSorted
  set bcNamesSorted [lsort $bcNames]

  variable bcTypes
  set bcTypes [pw::BoundaryCondition getPhysicalTypes]

  variable errors
  array set errors [list]

  variable caeSolver
  set caeSolver [pw::Application getCAESolver]

  variable isVerbose
  set isVerbose 0

  variable extSteps
  set extSteps 1

  variable extDistance
  set extDistance 1.0

  variable minBCName
  set minBCName [lindex $bcNames 0]

  variable minBCType
  set minBCType [lindex $bcTypes 0]

  variable minBCId
  set minBCId null

  variable maxBCName
  set maxBCName [lindex $bcNames 0]

  variable maxBCType
  set maxBCType [lindex $bcTypes 0]

  variable maxBCId
  set maxBCId null

  # widget hierarchy
  variable w
  set w(LabelTitle)          .title
  set w(FrameMain)           .main

    set w(StepsLabel)        $w(FrameMain).stepsLabel
    set w(StepsEntry)        $w(FrameMain).stepsEntry

    set w(DistanceLabel)     $w(FrameMain).distanceLabel
    set w(DistanceEntry)     $w(FrameMain).distanceEntry

    set w(BoundaryLabel)     $w(FrameMain).boundaryLabel
    set w(BCNameLabel)       $w(FrameMain).bcNameLabel
    set w(BCTypeLabel)       $w(FrameMain).bcTypeLabel
    set w(BCIdLabel)         $w(FrameMain).bcIdLabel

    set w(MinBCLabel)        $w(FrameMain).minBCLabel
    set w(MinBCNameCombo)    $w(FrameMain).minBCNameCombo
    set w(MinBCTypeCombo)    $w(FrameMain).minBCTypeCombo
    set w(MinBCIdEntry)      $w(FrameMain).minBCIdEntry

    set w(MaxBCLabel)        $w(FrameMain).maxBCLabel
    set w(MaxBCNameCombo)    $w(FrameMain).maxBCNameCombo
    set w(MaxBCTypeCombo)    $w(FrameMain).maxBCTypeCombo
    set w(MaxBCIdEntry)      $w(FrameMain).maxBCIdEntry

    set w(VerboseCheck)      $w(FrameMain).verboseCheck

  set w(FrameButtons)        .fbuttons
    set w(Logo)              $w(FrameButtons).logo
    set w(OkButton)          $w(FrameButtons).okButton
    set w(CancelButton)      $w(FrameButtons).cancelButton
} ;# namespace eval pw::Thicken2D::gui

#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::run { } {
  makeWindow
  tkwait window .
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::checkErrors { } {
  variable errors
  variable w
  if { 0 == [array size errors] } {
    set state normal
  } else {
    set state disabled
  }
  if { [catch {$w(OkButton) configure -state $state} err] } {
    #puts $err
  }
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::validateInput { val type key } {
  variable errors
  if { [string is $type -strict $val] } {
    array unset errors $key
  } else {
    set errors($key) 1
  }
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::validateInteger { val key } {
  validateInput $val integer $key
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::validateBCId { val key } {
  if { "null" == $val } {
    # make integer check happy
    set val 0
  }
  validateInteger $val $key
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::validateDouble { val key } {
  validateInput $val double $key
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::validateString { val key } {
  validateInput $val print $key
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::okAction { } {
  variable caeSolver
  variable isVerbose
  variable extDistance
  variable extSteps
  variable minBCName
  variable minBCType
  variable minBCId
  variable maxBCName
  variable maxBCType
  variable maxBCId

  setVerbose $isVerbose

  # Controls extrusion direction
  setExtDirection {0 0 1}

  # Controls extrusion distance
  setExtDistance $extDistance

  # Controls extrusion number of steps
  setExtSteps $extSteps

  # clear all BC setting for solver
  setSidewallBC $caeSolver
  setMinSidewallBC $caeSolver $minBCName $minBCType $minBCId
  setMaxSidewallBC $caeSolver $maxBCName $maxBCType $maxBCId

  # Capture a list of all the grid's domains
  set allDoms [pw::Grid getAll -type pw::Domain]
  # Only keep the visible and selectable domains
  set domsToThicken {}
  foreach dom $allDoms {
    if { ![pw::Display isLayerVisible [$dom getLayer]] } {
      continue
    } elseif { ![$dom getEnabled] } {
      continue
    } else {
      lappend domsToThicken $dom
    }
  }
  thicken $domsToThicken
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::stepsAction { action newVal oldVal } {
  if { -1 != $action } {
    validateInteger $newVal STEPS
    checkErrors
  }
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::distanceAction { action newVal oldVal } {
  variable extDistance
  if { -1 != $action } {
    validateDouble $newVal DISTANCE
    checkErrors
  }
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::bcNameAction { which action newVal oldVal } {
  set lwhich [string tolower $which]
  set bcTypeCombo ${which}BCTypeCombo
  set bcIdEntry ${which}BCIdEntry
  set bcTypeVar ${lwhich}BCType
  set bcIdVar ${lwhich}BCId

  variable w
  variable ${lwhich}BCName
  variable ${lwhich}BCType
  variable ${lwhich}BCId
  variable bcNamesSorted

  if { -1 == [lsearch -sorted $bcNamesSorted $newVal] } {
    # bc does not exist, allow type and id values
    $w($bcTypeCombo) configure -state readonly
    $w($bcIdEntry) configure -state normal
    set $bcIdVar null
  } else {
    # bc exists, disallow type and id values
    $w($bcTypeCombo) configure -state disabled
    $w($bcIdEntry) configure -state disabled
    set bc [pw::BoundaryCondition getByName $newVal]
    set $bcTypeVar [$bc getPhysicalType]
    if { "Unspecified" == $newVal } {
      set $bcIdVar ""
    } else {
      set $bcIdVar [$bc getId]
    }
  }
  validateString $newVal "${which}_NAME"
  checkErrors
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::minBCNameAction { action newVal oldVal } {
  bcNameAction Min $action $newVal $oldVal
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::maxBCNameAction { action newVal oldVal } {
  bcNameAction Max $action $newVal $oldVal
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::minBCIdAction { action newVal oldVal } {
  if { -1 != $action } {
    validateBCId $newVal MIN_ID
    checkErrors
  }
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::maxBCIdAction { action newVal oldVal } {
  if { -1 != $action } {
    validateBCId $newVal MAX_ID
    checkErrors
  }
  return 1
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::makeWindow { } {
  variable w
  variable caeSolver
  variable bcNames
  variable bcTypes
  variable minBCName
  variable maxBCName

  set disabledBgColor [ttk::style lookup TEntry -fieldbackground disabled]
  ttk::style map TCombobox -fieldbackground [list disabled $disabledBgColor]

  # create the widgets
  label $w(LabelTitle) -text "Thicken 2D Grid ($caeSolver)"
  setTitleFont $w(LabelTitle)

  frame $w(FrameMain) -padx 15

  label $w(StepsLabel) -text "Extrude Steps" -anchor w
  entry $w(StepsEntry) -textvariable pw::Thicken2D::gui::extSteps -width 4 \
    -validate key -validatecommand { pw::Thicken2D::gui::stepsAction %d %P %s }

  label $w(DistanceLabel) -text "Extrude Distance" -anchor w
  entry $w(DistanceEntry) -textvariable pw::Thicken2D::gui::extDistance -width 8 \
    -validate key -validatecommand { pw::Thicken2D::gui::distanceAction %d %P %s }

  label $w(BoundaryLabel) -text "Boundary" -anchor w
  label $w(BCNameLabel) -text "Name" -anchor w
  label $w(BCTypeLabel) -text "Type" -anchor w
  label $w(BCIdLabel) -text "Id" -anchor w

  label $w(MinBCLabel) -text "Min Side BC" -anchor w
  ttk::combobox $w(MinBCNameCombo) -values $bcNames -state normal \
    -textvariable pw::Thicken2D::gui::minBCName -validate key \
    -validatecommand { pw::Thicken2D::gui::minBCNameAction %d %P %s }
  bind $w(MinBCNameCombo) <<ComboboxSelected>> \
    {pw::Thicken2D::gui::minBCNameAction 9 $pw::Thicken2D::gui::minBCName \
      $pw::Thicken2D::gui::minBCName}
  ttk::combobox $w(MinBCTypeCombo) -values $bcTypes \
    -state readonly -textvariable pw::Thicken2D::gui::minBCType
  entry $w(MinBCIdEntry) -textvariable pw::Thicken2D::gui::minBCId -width 4 \
    -validate key -validatecommand { pw::Thicken2D::gui::minBCIdAction %d %P %s }

  label $w(MaxBCLabel) -text "Max Side BC" -anchor w
  ttk::combobox $w(MaxBCNameCombo) -values $bcNames \
    -state normal -textvariable pw::Thicken2D::gui::maxBCName -validate key \
    -validatecommand { pw::Thicken2D::gui::maxBCNameAction %d %P %s }
  bind $w(MaxBCNameCombo) <<ComboboxSelected>> \
    {pw::Thicken2D::gui::maxBCNameAction 9 $pw::Thicken2D::gui::maxBCName \
      $pw::Thicken2D::gui::maxBCName}
  ttk::combobox $w(MaxBCTypeCombo) -values $bcTypes \
    -state readonly -textvariable pw::Thicken2D::gui::maxBCType
  entry $w(MaxBCIdEntry) -textvariable pw::Thicken2D::gui::maxBCId -width 4 \
    -validate key -validatecommand { pw::Thicken2D::gui::maxBCIdAction %d %P %s }

  checkbutton $w(VerboseCheck) -text "Enable verbose output" \
    -variable pw::Thicken2D::gui::isVerbose -anchor w -padx 20 -state active

  frame $w(FrameButtons) -relief sunken -padx 15 -pady 5

  label $w(Logo) -image [cadenceLogo] -bd 0 -relief flat
  button $w(OkButton) -text "OK" -width 12 -bd 2 \
    -command { wm withdraw . ; pw::Thicken2D::gui::okAction ; exit }
  button $w(CancelButton) -text "Cancel" -width 12 -bd 2 \
    -command { exit }

  # lay out the form
  pack $w(LabelTitle) -side top -pady 5
  pack [frame .sp -bd 2 -height 2 -relief sunken] -pady 0 -side top -fill x
  pack $w(FrameMain) -side top -fill both -expand 1 -pady 10

  # lay out the form in a grid
  grid $w(StepsLabel)    -row 0 -column 0 -sticky we -pady 3 -padx 3
  grid $w(StepsEntry)    -row 0 -column 1 -sticky w -pady 3 -padx 3

  grid $w(DistanceLabel) -row 1 -column 0 -sticky we -pady 3 -padx 3
  grid $w(DistanceEntry) -row 1 -column 1 -sticky w -pady 3 -padx 3

  grid $w(BoundaryLabel) -row 2 -column 0 -sticky w -pady 3 -padx 3
  grid $w(BCNameLabel)   -row 2 -column 1 -sticky w -pady 3 -padx 3
  grid $w(BCTypeLabel)   -row 2 -column 2 -sticky w -pady 3 -padx 3
  grid $w(BCIdLabel)     -row 2 -column 3 -sticky w -pady 3 -padx 3

  grid $w(MinBCLabel)     -row 3 -column 0 -sticky we -pady 3 -padx 3
  grid $w(MinBCNameCombo) -row 3 -column 1 -sticky we -pady 3 -padx 3
  grid $w(MinBCTypeCombo) -row 3 -column 2 -sticky we -pady 3 -padx 3
  grid $w(MinBCIdEntry)   -row 3 -column 3 -sticky we -pady 3 -padx 3

  grid $w(MaxBCLabel)     -row 4 -column 0 -sticky we -pady 3 -padx 3
  grid $w(MaxBCNameCombo) -row 4 -column 1 -sticky we -pady 3 -padx 3
  grid $w(MaxBCTypeCombo) -row 4 -column 2 -sticky we -pady 3 -padx 3
  grid $w(MaxBCIdEntry)   -row 4 -column 3 -sticky we -pady 3 -padx 3

  grid $w(VerboseCheck)  -row 5 -columnspan 2 -sticky we -pady 3 -padx 3

  # lay out buttons
  pack $w(CancelButton) $w(OkButton) -pady 3 -padx 3 -side right
  pack $w(Logo) -side left -padx 5

  # give extra space to (only) column
  grid columnconfigure $w(FrameMain) 1 -weight 1

  pack $w(FrameButtons) -fill x -side bottom -padx 0 -pady 0 -anchor s

  # init GUI state for BC data
  minBCNameAction 8 $minBCName $minBCName
  maxBCNameAction 8 $maxBCName $maxBCName

  focus $w(VerboseCheck)
  raise .

  # don't allow window to resize
  wm resizable . 0 0
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::setTitleFont { widget {fontScale 1.5} } {
  # set the font for the input widget to be bold and 1.5 times larger than
  # the default font
  variable titleFont
  if { ! [info exists titleFont] } {
    set fontSize [font actual TkCaptionFont -size]
    set titleFont [font create -family [font actual TkCaptionFont -family] \
      -weight bold -size [expr {int($fontScale * $fontSize)}]]
  }
  $widget configure -font $titleFont
}


#----------------------------------------------------------------------------
proc pw::Thicken2D::gui::cadenceLogo {} {
  set logoData "
R0lGODlhgAAYAPQfAI6MjDEtLlFOT8jHx7e2tv39/RYSE/Pz8+Tj46qoqHl3d+vq62ZjY/n4+NT
T0+gXJ/BhbN3d3fzk5vrJzR4aG3Fubz88PVxZWp2cnIOBgiIeH769vtjX2MLBwSMfIP///yH5BA
EAAB8AIf8LeG1wIGRhdGF4bXD/P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIe
nJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtdGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1w
dGs9IkFkb2JlIFhNUCBDb3JlIDUuMC1jMDYxIDY0LjE0MDk0OSwgMjAxMC8xMi8wNy0xMDo1Nzo
wMSAgICAgICAgIj48cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudy5vcmcvMTk5OS8wMi
8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmY6YWJvdXQ9IiIg/3htbG5zO
nhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIiB4bWxuczpzdFJlZj0iaHR0
cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3NUcGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh
0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0idX
VpZDoxMEJEMkEwOThFODExMUREQTBBQzhBN0JCMEIxNUM4NyB4bXBNTTpEb2N1bWVudElEPSJ4b
XAuZGlkOkIxQjg3MzdFOEI4MTFFQjhEMv81ODVDQTZCRURDQzZBIiB4bXBNTTpJbnN0YW5jZUlE
PSJ4bXAuaWQ6QjFCODczNkZFOEI4MTFFQjhEMjU4NUNBNkJFRENDNkEiIHhtcDpDcmVhdG9yVG9
vbD0iQWRvYmUgSWxsdXN0cmF0b3IgQ0MgMjMuMSAoTWFjaW50b3NoKSI+IDx4bXBNTTpEZXJpZW
RGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6MGE1NjBhMzgtOTJiMi00MjdmLWE4ZmQtM
jQ0NjMzNmNjMWI0IiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOjBhNTYwYTM4LTkyYjItNDL/
N2YtYThkLTI0NDYzMzZjYzFiNCIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g
6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/PgH//v38+/r5+Pf29fTz8vHw7+7t7Ovp6Ofm5e
Tj4uHg397d3Nva2djX1tXU09LR0M/OzczLysnIx8bFxMPCwcC/vr28u7q5uLe2tbSzsrGwr66tr
KuqqainpqWko6KhoJ+enZybmpmYl5aVlJOSkZCPjo2Mi4qJiIeGhYSDgoGAf359fHt6eXh3dnV0
c3JxcG9ubWxramloZ2ZlZGNiYWBfXl1cW1pZWFdWVlVUU1JRUE9OTUxLSklIR0ZFRENCQUA/Pj0
8Ozo5ODc2NTQzMjEwLy4tLCsqKSgnJiUkIyIhIB8eHRwbGhkYFxYVFBMSERAPDg0MCwoJCAcGBQ
QDAgEAACwAAAAAgAAYAAAF/uAnjmQpTk+qqpLpvnAsz3RdFgOQHPa5/q1a4UAs9I7IZCmCISQwx
wlkSqUGaRsDxbBQer+zhKPSIYCVWQ33zG4PMINc+5j1rOf4ZCHRwSDyNXV3gIQ0BYcmBQ0NRjBD
CwuMhgcIPB0Gdl0xigcNMoegoT2KkpsNB40yDQkWGhoUES57Fga1FAyajhm1Bk2Ygy4RF1seCjw
vAwYBy8wBxjOzHq8OMA4CWwEAqS4LAVoUWwMul7wUah7HsheYrxQBHpkwWeAGagGeLg717eDE6S
4HaPUzYMYFBi211FzYRuJAAAp2AggwIM5ElgwJElyzowAGAUwQL7iCB4wEgnoU/hRgIJnhxUlpA
SxY8ADRQMsXDSxAdHetYIlkNDMAqJngxS47GESZ6DSiwDUNHvDd0KkhQJcIEOMlGkbhJlAK/0a8
NLDhUDdX914A+AWAkaJEOg0U/ZCgXgCGHxbAS4lXxketJcbO/aCgZi4SC34dK9CKoouxFT8cBNz
Q3K2+I/RVxXfAnIE/JTDUBC1k1S/SJATl+ltSxEcKAlJV2ALFBOTMp8f9ihVjLYUKTa8Z6GBCAF
rMN8Y8zPrZYL2oIy5RHrHr1qlOsw0AePwrsj47HFysrYpcBFcF1w8Mk2ti7wUaDRgg1EISNXVwF
lKpdsEAIj9zNAFnW3e4gecCV7Ft/qKTNP0A2Et7AUIj3ysARLDBaC7MRkF+I+x3wzA08SLiTYER
KMJ3BoR3wzUUvLdJAFBtIWIttZEQIwMzfEXNB2PZJ0J1HIrgIQkFILjBkUgSwFuJdnj3i4pEIlg
eY+Bc0AGSRxLg4zsblkcYODiK0KNzUEk1JAkaCkjDbSc+maE5d20i3HY0zDbdh1vQyWNuJkjXnJ
C/HDbCQeTVwOYHKEJJwmR/wlBYi16KMMBOHTnClZpjmpAYUh0GGoyJMxya6KcBlieIj7IsqB0ji
5iwyyu8ZboigKCd2RRVAUTQyBAugToqXDVhwKpUIxzgyoaacILMc5jQEtkIHLCjwQUMkxhnx5I/
seMBta3cKSk7BghQAQMeqMmkY20amA+zHtDiEwl10dRiBcPoacJr0qjx7Ai+yTjQvk31aws92JZ
Q1070mGsSQsS1uYWiJeDrCkGy+CZvnjFEUME7VaFaQAcXCCDyyBYA3NQGIY8ssgU7vqAxjB4EwA
DEIyxggQAsjxDBzRagKtbGaBXclAMMvNNuBaiGAAA7"

  return [image create photo -format GIF -data $logoData]
}

} ;# ![namespace exists pw::Thicken2D]


#####################################################################
#                           MAIN
#####################################################################
if { ![info exists disableAutoRun_Thicken2D] } {
  pw::Script loadTk
  pw::Thicken2D::gui::run
}

# END SCRIPT

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
