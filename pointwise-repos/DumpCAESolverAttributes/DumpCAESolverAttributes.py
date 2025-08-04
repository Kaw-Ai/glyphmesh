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


from pointwise import GlyphClient
from pointwise.glyphapi import *

try:
    # Port 0 indicates batch mode
    glf = GlyphClient(port=0)
    glf.connect()
except:
    print("Could not connect to specified port")
    exit()
# Use the Glyph API for Python
pw = glf.get_glyphapi()


# Return length of longest attribute name for display format
def buildFmt(attrNames):
    maxLen = 20 # min col width
    for name in attrNames:
        length = len(name)
        if length > maxLen:
            maxLen = length
    return maxLen


# Accumulate attributes of given solver and dimension to be displayed, store in global list
def dumpSolver(solvName, dim):
    try:
        # pw::Application setCAESolver \"$solverName\" $dim
        pw.Application.setCAESolver(solvName, dim)
        # set attrNames [pw::Application getCAESolverAttributeNames -fullname -group All]
        attrNames = pw.Application.getCAESolverAttributeNames(fullname=True, group="ALL")
        if len(attrNames) != 0:
            fmt = buildFmt(attrNames)
            allAttributes.append("%dD Attributes:" % dim)
            allAttributes.append("| %20.20s | %-*s | %s" % ("Value", fmt, "Name", \
                "Definition (type access desc value range)"))
            allAttributes.append("| %20.20s | %-*.*s | %s" % (hr, fmt, fmt, hr, hr))
            attrNames.sort(reverse=False)
            for name in attrNames:
                # set val [pw::Application getCAESolverAttribute $name]
                val = pw.Application.getCAESolverAttribute(name)
                # set defn [pw::Application getCAESolverAttributeDefinition $name]
                defn = pw.Application.getCAESolverAttributeDefinition(name)
                if ((len(defn) < 7) or (defn[6] is None)):
                    allAttributes.append("| %20.20s | %-*s | %s %s {%s} %s {}" % (val, fmt, name, defn[2], defn[3], \
                        defn[4], defn[5]))
                else:
                    allAttributes.append("| %20.20s | %-*s | %s %s {%s} %s {%s}" % (val, fmt, name, defn[2], defn[3], \
                        defn[4], defn[5], defn[6]))
        else:
            allAttributes.append("%dD Attributes: NONE" % dim)
    except:
        return


# Main
# Check for existing entities
if not pw.Application.isEmpty():
    glf.puts("This script can only be run on an empty project.")
    exit(1)

allAttributes = []
hr = "--------------------------------------------------"
# set currentSolver [pw::Application getCAESolver]
currentSolver = pw.Application.getCAESolver()
# set solverNames [pw::Application getCAESolverNames]
solverNames = pw.Application.getCAESolverNames()
for solverName in solverNames:
    allAttributes.append(solverName)
    allAttributes.append("============================================")
    dumpSolver(solverName, 3)
    dumpSolver(solverName, 2)
    allAttributes.append("")
print('\n'.join(allAttributes))
# pw::Application setCAESolver $currentSolver
pw.Application.setCAESolver(currentSolver)


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
