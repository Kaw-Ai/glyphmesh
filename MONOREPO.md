# Pointwise Glyph Monorepo

This monorepo contains a collection of Pointwise Glyph scripts and utilities integrated from multiple repositories for holistic development and maintenance.

## Structure

- **Root Directory**: Contains the original ConformalModelMesher project
- **pointwise-repos/**: Contains all cloned Pointwise repositories without git headers

## Included Repositories

This monorepo includes 88 repositories from the Pointwise organization:

### Core Meshing Tools
- AircraftMesher
- ConformalModelMesher (already present in root)
- AirfoilMesh
- AirfoilGenerator
- GridToSource
- GridCoordEnum
- GeomToMesh

### CAE Solvers and Exporters
- CaeUnsSU2
- CaeUnsTAU
- CaeUnsUMCPSEG
- CaeUnsOpenFOAM
- CaeUnsFluent
- CaeUnsAzore
- CaeUnsADS
- GrdpOpenFOAM
- GrdpSU2

### Domain and Block Utilities
- DomainDiagnose
- BlockDiagnose
- BrickBlock
- Domain2Ellipse
- DomAreaBlkVol
- Dom2DBEntity
- UniformDomRefinement
- ViewDomsByCons

### Connector Utilities
- HelicalConnector
- ConnectorSpacing
- ConEllipse
- ConCircleCenterAndRadius
- ConGeometricAutoDimension
- ConSplitAtIsect
- ConSplitIntoN
- ChangeConWallSpacings
- SplitConsAtIntersection

### Geometry and Curve Tools
- CreateFromAnalyticFunction
- CreateSourceCurveFromConOrDB
- CreateFFD
- CreateOH
- dbCurveExtension
- Semicircle
- Fillet
- FitSurfacePatch

### Grid Manipulation
- GridRefine
- GridOfLife
- ConvertGridToDatabase
- Thicken2Dto3D
- ExtrudeEdges
- ExtrudePipe

### Visualization and Display
- SyncAndCapDisplays
- ColorDomByDB
- ColorEntitiesByType
- PickDbByColor
- RotateView
- DispRotByAngle

### Math and Geometry Utilities
- CalcAngleFromPoints
- GetCenterPoint
- TurningAngleCalculator
- DimensionConFromSpacings
- CenterRotationAxes
- FibonacciSpiral
- KochSnowflake

### Array and Copy Operations
- ArrayCopyRotate
- MultiCopyTranslate
- MultiConnectorSplitWithPercent

### Specialized Tools
- ButterflyMaker
- ShapeWizard
- QuickScript
- QuadWarp
- TriQuad
- SqueezeCon
- ScaledOffset
- SmoothStraight
- SymmetryPlane
- MyDistribution
- GrowthProfile
- PointSource
- Plot3dMerge
- TgridImport
- VSP2CFD
- ReEntryVehiclePython

### Client Libraries
- GlyphClientPython
- GlyphClientPerl
- GlyphUtilityLibrary

### Utility and Cleanup Tools
- RemoveControlPts
- RemoveFluentBCNumber
- DumpCAESolverAttributes

### Documentation and Examples
- How-To-Integrate-Plugin-Code
- pointwise.github.com

### Cloud and Integration
- OnCloudBlockInit
- MeshLink

## Build Process

The monorepo was built using the `build_monorepo.sh` script which:

1. Clones all repositories from the pointwise organization
2. Removes git headers (.git directories) from each repository
3. Organizes them in the `pointwise-repos/` directory

## Usage

Each repository maintains its original structure and documentation. Refer to individual README files within each repository directory for specific usage instructions.

## License

All repositories maintain their original licensing. Most are licensed under the Cadence Public License Version 1.0.