#!/bin/bash

# Script to build monorepo by cloning all pointwise repositories
# and removing their git headers for holistic integration

set -e  # Exit on any error

echo "Starting monorepo build process..."

# Create directory for cloned repositories
mkdir -p pointwise-repos

# List of all repositories to clone
repos=(
    "AircraftMesher"
    "ConformalModelMesher"
    "GlyphClientPython"
    "pointwise.github.com"
    "OnCloudBlockInit"
    "GridToSource"
    "GridCoordEnum"
    "HelicalConnector"
    "Thicken2Dto3D"
    "UniformDomRefinement"
    "DomainDiagnose"
    "ButterflyMaker"
    "ViewDomsByCons"
    "SymmetryPlane"
    "SplitConsAtIntersection"
    "ShapeWizard"
    "Semicircle"
    "RemoveControlPts"
    "QuickScript"
    "PickDbByColor"
    "ExtrudePipe"
    "ExtrudeEdges"
    "DomAreaBlkVol"
    "Dom2DBEntity"
    "DispRotByAngle"
    "DimensionConFromSpacings"
    "dbCurveExtension"
    "CreateOH"
    "CreateFromAnalyticFunction"
    "ConvertGridToDatabase"
    "ConSplitIntoN"
    "ConSplitAtIsect"
    "ConnectorSpacing"
    "ConEllipse"
    "ConCircleCenterAndRadius"
    "ColorDomByDB"
    "ChangeConWallSpacings"
    "CenterRotationAxes"
    "BrickBlock"
    "BlockDiagnose"
    "GrowthProfile"
    "CaeUnsSU2"
    "CaeUnsTAU"
    "CaeUnsUMCPSEG"
    "Domain2Ellipse"
    "CreateSourceCurveFromConOrDB"
    "CreateFFD"
    "ConGeometricAutoDimension"
    "CalcAngleFromPoints"
    "AirfoilMesh"
    "AirfoilGenerator"
    "GetCenterPoint"
    "ColorEntitiesByType"
    "How-To-Integrate-Plugin-Code"
    "GlyphUtilityLibrary"
    "SmoothStraight"
    "TgridImport"
    "SyncAndCapDisplays"
    "RemoveFluentBCNumber"
    "PointSource"
    "Fillet"
    "DumpCAESolverAttributes"
    "MultiCopyTranslate"
    "GlyphClientPerl"
    "GeomToMesh"
    "ArrayCopyRotate"
    "VSP2CFD"
    "TurningAngleCalculator"
    "TriQuad"
    "SqueezeCon"
    "ScaledOffset"
    "RotateView"
    "ReEntryVehiclePython"
    "QuadWarp"
    "Plot3dMerge"
    "MyDistribution"
    "MultiConnectorSplitWithPercent"
    "KochSnowflake"
    "GridRefine"
    "GridOfLife"
    "GrdpSU2"
    "GrdpOpenFOAM"
    "FitSurfacePatch"
    "CaeUnsOpenFOAM"
    "CaeUnsFluent"
    "CaeUnsAzore"
    "CaeUnsADS"
    "FibonacciSpiral"
    "MeshLink"
)

echo "Found ${#repos[@]} repositories to clone"

# Clone each repository
for repo in "${repos[@]}"; do
    echo "Cloning $repo..."
    if [ "$repo" = "ConformalModelMesher" ]; then
        echo "Skipping $repo as it already exists in the current repository"
        continue
    fi
    
    git clone "https://github.com/pointwise/$repo" "pointwise-repos/$repo"
    
    # Remove .git directory to eliminate git headers
    if [ -d "pointwise-repos/$repo/.git" ]; then
        echo "Removing git headers from $repo"
        rm -rf "pointwise-repos/$repo/.git"
    fi
done

echo "Monorepo build process completed!"
echo "All repositories have been cloned to pointwise-repos/ directory with git headers removed"