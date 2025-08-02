#############################################################################
#
# (C) 2021 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This sample script is not supported by Cadence Design Systems, Inc.
# It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#
#############################################################################

#
# This sample script was written and provided by a 3rd party outside of
# Cadence Design Systems's control.  It is not supported by Cadence Design
# Systems, Inc.  It is provided freely for demonstration purposes only.
# SEE THE WARRANTY DISCLAIMER AT THE BOTTOM OF THIS FILE.
#

###############################################################################
##
## Color.glf
##
## Set colors of connectors, domains, and databases
##
###############################################################################

package require PWI_Glyph 2.4
pw::Script loadTk

############################################################################
# makeTitle
############################################################################
proc makeTitle { t } {
  pack [frame $t.title] -side top -expand FALSE -fill x
  pack [label $t.title.label -text "Color Entities" -justify center] \
    -side top -fill y -pady 5
  pack [frame $t.title.hr -height 2 -relief sunken -borderwidth 1] \
    -side top -fill x

  set font [$t.title.label cget -font]
  set fontFamily [font actual $font -family]
  set fontSize [font actual $font -size]
  set bigLabelFont [font create -family $fontFamily -weight bold \
    -size [expr {int(1.5 * $fontSize)}]]
  $t.title.label configure -font $bigLabelFont

  wm title . "Color Entities"
}

############################################################################
# colorData
############################################################################
proc colorData { color } {
  set data [pw::Database getAll]
  
  if {$color == "black"} {
    foreach datas $data {
      set color_hex #000000
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "grey"} {
    foreach datas $data {
      set color_hex #999999
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "red"} {
    foreach datas $data {
      set color_hex #CC0033
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "orange"} {
    foreach datas $data {
      set color_hex #FF3300
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "pink"} {
    foreach datas $data {
      set color_hex #FF33CC
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "yellow"} {
    foreach datas $data {
      set color_hex #FFFF00
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "white"} {
    foreach datas $data {
      set color_hex #FFFFFF
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "green"} {
    foreach datas $data {
      set color_hex #FF7F3F
      $datas setRenderAttribute ColorMode Entity
      $datas setColor $color_hex
      pw::Display update 
    }
  } elseif { $color == "default"} {
    foreach datas $data {
      $datas setRenderAttribute ColorMode Automatic
      pw::Display update
    }
  }
}

############################################################################
# colorCon
############################################################################
proc colorCon { color } {
  set con [pw::Grid getAll -type pw::Connector]
  
  if {$color == "black"} {
    foreach cons $con {
      set color #000000 
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "grey"} {
    foreach cons $con {
      set color #999999
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "red"} {
    foreach cons $con {
      set color #CC0033
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "orange"} {
    foreach cons $con {
      set color #FF3300
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "pink"} {
    foreach cons $con {
      set color #FF33CC
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "yellow"} {
    foreach cons $con {
      set color #FFFF00
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "white"} {
    foreach cons $con {
      set color #FFFFFF
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "green"} {
    foreach cons $con {
      set color #FF7F3F
      $cons setRenderAttribute ColorMode Entity
      $cons setColor $color
      pw::Display update 
    }
  } elseif { $color == "default"} {
    foreach cons $con {
      $cons setRenderAttribute ColorMode Automatic
      pw::Display update
    }
  }
}

############################################################################
# colorDom
############################################################################
proc colorDom { color } {
  set dom [pw::Grid getAll -type pw::Domain]
  
  if {$color == "black"} {
    foreach doms $dom {
      set color #000000 
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "grey"} {
    foreach doms $dom {
      set color #999999
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "red"} {
    foreach doms $dom {
      set color #CC0033
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "orange"} {
    foreach doms $dom {
      set color #FF3300
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "pink"} {
    foreach doms $dom {
      set color #FF33CC
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "yellow"} {
    foreach doms $dom {
      set color #FFFF00
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "white"} {
    foreach doms $dom {
      set color #FFFFFF
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "green"} {
    foreach doms $dom {
      set color #FF7F3F
      $doms setRenderAttribute ColorMode Entity
      $doms setColor $color
      pw::Display update 
    }
  } elseif { $color == "default"} {
    foreach doms $dom {
      $doms setRenderAttribute ColorMode Automatic
      pw::Display update
    }
  }
}

############################################################################
# makeTK
############################################################################
proc makeTK { } {

  pack [frame .t]
  makeTitle .t
  pack [frame .t.hr -height 2 -relief sunken -borderwidth 1] -side top -fill x
  pack [frame .f]
  pack [label .f.one -text "Database Color"] -padx 50 -pady 10
  pack [frame .f.data]
  pack [button .f.data.black -background "#000000" -command {colorData black}] -padx 1 -side right -ipadx 10
  pack [button .f.data.grey -background "#999999" -command {colorData grey}] -padx 1 -side right -ipadx 10
  pack [button .f.data.red -background "#CC0033" -command {colorData red}] -padx 1 -side right -ipadx 10  
  pack [button .f.data.orange -background "#FF3300" -command {colorData orange}] -padx 1 -side right -ipadx 10
  pack [button .f.data.pink -background "#FF33CC" -command {colorData pink}] -padx 1 -side right -ipadx 10
  pack [button .f.data.yellow -background "#FFFF00" -command {colorData yellow}] -padx 1 -side right -ipadx 10
  pack [button .f.data.white -background "#FFFFFF" -command {colorData white}] -padx 1 -side right -ipadx 10
  pack [button .f.data.green -background "#FF7F3F" -command {colorData green}] -padx 1 -side right -ipadx 10
  pack [button .f.data.default -text "Default" -command {colorData default}] -padx 1 -side right -ipadx 10
  pack [label .f.two -text "Connector Color"] -padx 1 -padx 50 -pady 10 -ipadx 10
  pack [frame .f.con]
  pack [button .f.con.black -background "#000000" -command {colorCon black}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.grey -background "#999999" -command {colorCon grey}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.red -background "#CC0033" -command {colorCon red}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.orange -background "#FF3300" -command {colorCon orange}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.pink -background "#FF33CC" -command {colorCon pink}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.yellow -background "#FFFF00" -command {colorCon yellow}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.white -background "#FFFFFF" -command {colorCon white}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.green -background "#FF7F3F" -command {colorCon green}] -padx 1 -ipadx 10 -side right
  pack [button .f.con.default -text "Default" -command {colorCon default}] -padx 1 -side right -ipadx 10
  pack [label .f.three -text "Domain Color"] -padx 1 -padx 50 -pady 10
  pack [frame .f.dom]
  pack [button .f.dom.black -background "#000000" -command {colorDom black}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.grey -background "#999999" -command {colorDom grey}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.red -background "#CC0033" -command {colorDom red}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.orange -background "#FF3300" -command {colorDom orange}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.pink -background "#FF33CC" -command {colorDom pink}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.yellow -background "#FFFF00" -command {colorDom yellow}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.white -background "#FFFFFF" -command {colorDom white}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.green -background "#FF7F3F" -command {colorDom green}] -padx 1 -ipadx 10 -side right
  pack [button .f.dom.default -text "Default" -command {colorDom default}] -padx 1 -side right -ipadx 10
  pack [frame .g]
  pack [frame .g.hr -height 2 -relief sunken -borderwidth 1] -side top -fill x
  pack [frame .g.bf] -expand true -fill x -pady 10
  pack [frame .h]
  pack [button .h.cancel -text "OK" -command exit] -side right -padx 5

  ::tk::PlaceWindow . widget

}

makeTK
tkwait window .

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
