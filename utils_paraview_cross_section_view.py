# trace generated using paraview version 5.12.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def plot(fdir, clip_z, outname):

    # create a new 'Legacy VTK Reader'
    fname = fdir + "/06_Mesh/Mesh_3D.vtk"
    mesh_3Dvtk = LegacyVTKReader(registrationName='Mesh_3D.vtk', FileNames=fname)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    mesh_3DvtkDisplay = Show(mesh_3Dvtk, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    mesh_3DvtkDisplay.Selection = None
    mesh_3DvtkDisplay.Representation = 'Surface'
    mesh_3DvtkDisplay.ColorArrayName = [None, '']
    mesh_3DvtkDisplay.LookupTable = None
    mesh_3DvtkDisplay.MapScalars = 1
    mesh_3DvtkDisplay.MultiComponentsMapping = 0
    mesh_3DvtkDisplay.InterpolateScalarsBeforeMapping = 1
    mesh_3DvtkDisplay.UseNanColorForMissingArrays = 0
    mesh_3DvtkDisplay.Opacity = 1.0
    mesh_3DvtkDisplay.PointSize = 2.0
    mesh_3DvtkDisplay.LineWidth = 1.0
    mesh_3DvtkDisplay.RenderLinesAsTubes = 0
    mesh_3DvtkDisplay.RenderPointsAsSpheres = 0
    mesh_3DvtkDisplay.Interpolation = 'Gouraud'
    mesh_3DvtkDisplay.Specular = 0.0
    mesh_3DvtkDisplay.SpecularColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.SpecularPower = 100.0
    mesh_3DvtkDisplay.Luminosity = 0.0
    mesh_3DvtkDisplay.Ambient = 0.0
    mesh_3DvtkDisplay.Diffuse = 1.0
    mesh_3DvtkDisplay.Roughness = 0.3
    mesh_3DvtkDisplay.Metallic = 0.0
    mesh_3DvtkDisplay.EdgeTint = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.Anisotropy = 0.0
    mesh_3DvtkDisplay.AnisotropyRotation = 0.0
    mesh_3DvtkDisplay.BaseIOR = 1.5
    mesh_3DvtkDisplay.CoatStrength = 0.0
    mesh_3DvtkDisplay.CoatIOR = 2.0
    mesh_3DvtkDisplay.CoatRoughness = 0.0
    mesh_3DvtkDisplay.CoatColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.SelectTCoordArray = 'None'
    mesh_3DvtkDisplay.SelectNormalArray = 'None'
    mesh_3DvtkDisplay.SelectTangentArray = 'None'
    mesh_3DvtkDisplay.Texture = None
    mesh_3DvtkDisplay.RepeatTextures = 1
    mesh_3DvtkDisplay.InterpolateTextures = 0
    mesh_3DvtkDisplay.SeamlessU = 0
    mesh_3DvtkDisplay.SeamlessV = 0
    mesh_3DvtkDisplay.UseMipmapTextures = 0
    mesh_3DvtkDisplay.ShowTexturesOnBackface = 1
    mesh_3DvtkDisplay.BaseColorTexture = None
    mesh_3DvtkDisplay.NormalTexture = None
    mesh_3DvtkDisplay.NormalScale = 1.0
    mesh_3DvtkDisplay.CoatNormalTexture = None
    mesh_3DvtkDisplay.CoatNormalScale = 1.0
    mesh_3DvtkDisplay.MaterialTexture = None
    mesh_3DvtkDisplay.OcclusionStrength = 1.0
    mesh_3DvtkDisplay.AnisotropyTexture = None
    mesh_3DvtkDisplay.EmissiveTexture = None
    mesh_3DvtkDisplay.EmissiveFactor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.FlipTextures = 0
    mesh_3DvtkDisplay.EdgeOpacity = 1.0
    mesh_3DvtkDisplay.BackfaceRepresentation = 'Follow Frontface'
    mesh_3DvtkDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.BackfaceOpacity = 1.0
    mesh_3DvtkDisplay.Position = [0.0, 0.0, 0.0]
    mesh_3DvtkDisplay.Scale = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.Orientation = [0.0, 0.0, 0.0]
    mesh_3DvtkDisplay.Origin = [0.0, 0.0, 0.0]
    mesh_3DvtkDisplay.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
    mesh_3DvtkDisplay.Pickable = 1
    mesh_3DvtkDisplay.Triangulate = 0
    mesh_3DvtkDisplay.UseShaderReplacements = 0
    mesh_3DvtkDisplay.ShaderReplacements = ''
    mesh_3DvtkDisplay.NonlinearSubdivisionLevel = 1
    mesh_3DvtkDisplay.MatchBoundariesIgnoringCellOrder = 0
    mesh_3DvtkDisplay.UseDataPartitions = 0
    mesh_3DvtkDisplay.OSPRayUseScaleArray = 'All Approximate'
    mesh_3DvtkDisplay.OSPRayScaleArray = 'gmsh:dim_tags'
    mesh_3DvtkDisplay.OSPRayScaleFunction = 'Piecewise Function'
    mesh_3DvtkDisplay.OSPRayMaterial = 'None'
    mesh_3DvtkDisplay.Assembly = ''
    mesh_3DvtkDisplay.BlockSelectors = ['/']
    mesh_3DvtkDisplay.BlockColors = []
    mesh_3DvtkDisplay.BlockOpacities = []
    mesh_3DvtkDisplay.Orient = 0
    mesh_3DvtkDisplay.OrientationMode = 'Direction'
    mesh_3DvtkDisplay.SelectOrientationVectors = 'None'
    mesh_3DvtkDisplay.Scaling = 0
    mesh_3DvtkDisplay.ScaleMode = 'No Data Scaling Off'
    mesh_3DvtkDisplay.ScaleFactor = 4.744257707130562
    mesh_3DvtkDisplay.SelectScaleArray = 'None'
    mesh_3DvtkDisplay.GlyphType = 'Arrow'
    mesh_3DvtkDisplay.UseGlyphTable = 0
    mesh_3DvtkDisplay.GlyphTableIndexArray = 'None'
    mesh_3DvtkDisplay.UseCompositeGlyphTable = 0
    mesh_3DvtkDisplay.UseGlyphCullingAndLOD = 0
    mesh_3DvtkDisplay.LODValues = []
    mesh_3DvtkDisplay.ColorByLODIndex = 0
    mesh_3DvtkDisplay.GaussianRadius = 0.2372128853565281
    mesh_3DvtkDisplay.ShaderPreset = 'Sphere'
    mesh_3DvtkDisplay.CustomTriangleScale = 3
    mesh_3DvtkDisplay.CustomShader = """ // This custom shader code define a gaussian blur
    // Please take a look into vtkSMPointGaussianRepresentation.cxx
    // for other custom shader examples
    //VTK::Color::Impl
    float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
    float gaussian = exp(-0.5*dist2);
    opacity = opacity*gaussian;
    """
    mesh_3DvtkDisplay.Emissive = 0
    mesh_3DvtkDisplay.ScaleByArray = 0
    mesh_3DvtkDisplay.SetScaleArray = ['POINTS', 'gmsh:dim_tags']
    mesh_3DvtkDisplay.ScaleArrayComponent = 'X'
    mesh_3DvtkDisplay.UseScaleFunction = 1
    mesh_3DvtkDisplay.ScaleTransferFunction = 'Piecewise Function'
    mesh_3DvtkDisplay.OpacityByArray = 0
    mesh_3DvtkDisplay.OpacityArray = ['POINTS', 'gmsh:dim_tags']
    mesh_3DvtkDisplay.OpacityArrayComponent = 'X'
    mesh_3DvtkDisplay.OpacityTransferFunction = 'Piecewise Function'
    mesh_3DvtkDisplay.DataAxesGrid = 'Grid Axes Representation'
    mesh_3DvtkDisplay.SelectionCellLabelBold = 0
    mesh_3DvtkDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    mesh_3DvtkDisplay.SelectionCellLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.SelectionCellLabelFontFile = ''
    mesh_3DvtkDisplay.SelectionCellLabelFontSize = 18
    mesh_3DvtkDisplay.SelectionCellLabelItalic = 0
    mesh_3DvtkDisplay.SelectionCellLabelJustification = 'Left'
    mesh_3DvtkDisplay.SelectionCellLabelOpacity = 1.0
    mesh_3DvtkDisplay.SelectionCellLabelShadow = 0
    mesh_3DvtkDisplay.SelectionPointLabelBold = 0
    mesh_3DvtkDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    mesh_3DvtkDisplay.SelectionPointLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.SelectionPointLabelFontFile = ''
    mesh_3DvtkDisplay.SelectionPointLabelFontSize = 18
    mesh_3DvtkDisplay.SelectionPointLabelItalic = 0
    mesh_3DvtkDisplay.SelectionPointLabelJustification = 'Left'
    mesh_3DvtkDisplay.SelectionPointLabelOpacity = 1.0
    mesh_3DvtkDisplay.SelectionPointLabelShadow = 0
    mesh_3DvtkDisplay.PolarAxes = 'Polar Axes Representation'
    mesh_3DvtkDisplay.ScalarOpacityFunction = None
    mesh_3DvtkDisplay.ScalarOpacityUnitDistance = 3.7183592969141275
    mesh_3DvtkDisplay.UseSeparateOpacityArray = 0
    mesh_3DvtkDisplay.OpacityArrayName = ['POINTS', 'gmsh:dim_tags']
    mesh_3DvtkDisplay.OpacityComponent = 'X'
    mesh_3DvtkDisplay.SelectMapper = 'Projected tetra'
    mesh_3DvtkDisplay.SamplingDimensions = [128, 128, 128]
    mesh_3DvtkDisplay.UseFloatingPointFrameBuffer = 1
    mesh_3DvtkDisplay.SelectInputVectors = ['POINTS', 'gmsh:dim_tags']
    mesh_3DvtkDisplay.NumberOfSteps = 40
    mesh_3DvtkDisplay.StepSize = 0.25
    mesh_3DvtkDisplay.NormalizeVectors = 1
    mesh_3DvtkDisplay.EnhancedLIC = 1
    mesh_3DvtkDisplay.ColorMode = 'Blend'
    mesh_3DvtkDisplay.LICIntensity = 0.8
    mesh_3DvtkDisplay.MapModeBias = 0.0
    mesh_3DvtkDisplay.EnhanceContrast = 'Off'
    mesh_3DvtkDisplay.LowLICContrastEnhancementFactor = 0.0
    mesh_3DvtkDisplay.HighLICContrastEnhancementFactor = 0.0
    mesh_3DvtkDisplay.LowColorContrastEnhancementFactor = 0.0
    mesh_3DvtkDisplay.HighColorContrastEnhancementFactor = 0.0
    mesh_3DvtkDisplay.AntiAlias = 0
    mesh_3DvtkDisplay.MaskOnSurface = 1
    mesh_3DvtkDisplay.MaskThreshold = 0.0
    mesh_3DvtkDisplay.MaskIntensity = 0.0
    mesh_3DvtkDisplay.MaskColor = [0.5, 0.5, 0.5]
    mesh_3DvtkDisplay.GenerateNoiseTexture = 0
    mesh_3DvtkDisplay.NoiseType = 'Gaussian'
    mesh_3DvtkDisplay.NoiseTextureSize = 128
    mesh_3DvtkDisplay.NoiseGrainSize = 2
    mesh_3DvtkDisplay.MinNoiseValue = 0.0
    mesh_3DvtkDisplay.MaxNoiseValue = 0.8
    mesh_3DvtkDisplay.NumberOfNoiseLevels = 1024
    mesh_3DvtkDisplay.ImpulseNoiseProbability = 1.0
    mesh_3DvtkDisplay.ImpulseNoiseBackgroundValue = 0.0
    mesh_3DvtkDisplay.NoiseGeneratorSeed = 1
    mesh_3DvtkDisplay.CompositeStrategy = 'AUTO'
    mesh_3DvtkDisplay.UseLICForLOD = 0
    mesh_3DvtkDisplay.WriteLog = ''

    # init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
    mesh_3DvtkDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 7.0, 1.0, 0.5, 0.0]
    mesh_3DvtkDisplay.OSPRayScaleFunction.UseLogScale = 0

    # init the 'Arrow' selected for 'GlyphType'
    mesh_3DvtkDisplay.GlyphType.TipResolution = 6
    mesh_3DvtkDisplay.GlyphType.TipRadius = 0.1
    mesh_3DvtkDisplay.GlyphType.TipLength = 0.35
    mesh_3DvtkDisplay.GlyphType.ShaftResolution = 6
    mesh_3DvtkDisplay.GlyphType.ShaftRadius = 0.03
    mesh_3DvtkDisplay.GlyphType.Invert = 0

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    mesh_3DvtkDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    mesh_3DvtkDisplay.ScaleTransferFunction.UseLogScale = 0

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    mesh_3DvtkDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    mesh_3DvtkDisplay.OpacityTransferFunction.UseLogScale = 0

    # init the 'Grid Axes Representation' selected for 'DataAxesGrid'
    mesh_3DvtkDisplay.DataAxesGrid.XTitle = 'X Axis'
    mesh_3DvtkDisplay.DataAxesGrid.YTitle = 'Y Axis'
    mesh_3DvtkDisplay.DataAxesGrid.ZTitle = 'Z Axis'
    mesh_3DvtkDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.XTitleFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.XTitleBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.XTitleItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.XTitleFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.XTitleShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.XTitleOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.YTitleFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.YTitleBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.YTitleItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.YTitleFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.YTitleShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.YTitleOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZTitleOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.FacesToRender = 63
    mesh_3DvtkDisplay.DataAxesGrid.CullBackface = 0
    mesh_3DvtkDisplay.DataAxesGrid.CullFrontface = 1
    mesh_3DvtkDisplay.DataAxesGrid.ShowGrid = 0
    mesh_3DvtkDisplay.DataAxesGrid.ShowEdges = 1
    mesh_3DvtkDisplay.DataAxesGrid.ShowTicks = 1
    mesh_3DvtkDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
    mesh_3DvtkDisplay.DataAxesGrid.AxesToLabel = 63
    mesh_3DvtkDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.XLabelFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.XLabelBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.XLabelItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.XLabelFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.XLabelShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.XLabelOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.YLabelFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.YLabelBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.YLabelItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.YLabelFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.YLabelShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.YLabelOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelFontFile = ''
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelBold = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelItalic = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelFontSize = 12
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelShadow = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZLabelOpacity = 1.0
    mesh_3DvtkDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
    mesh_3DvtkDisplay.DataAxesGrid.XAxisPrecision = 2
    mesh_3DvtkDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
    mesh_3DvtkDisplay.DataAxesGrid.XAxisLabels = []
    mesh_3DvtkDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
    mesh_3DvtkDisplay.DataAxesGrid.YAxisPrecision = 2
    mesh_3DvtkDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
    mesh_3DvtkDisplay.DataAxesGrid.YAxisLabels = []
    mesh_3DvtkDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
    mesh_3DvtkDisplay.DataAxesGrid.ZAxisPrecision = 2
    mesh_3DvtkDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
    mesh_3DvtkDisplay.DataAxesGrid.ZAxisLabels = []
    mesh_3DvtkDisplay.DataAxesGrid.UseCustomBounds = 0
    mesh_3DvtkDisplay.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # init the 'Polar Axes Representation' selected for 'PolarAxes'
    mesh_3DvtkDisplay.PolarAxes.Visibility = 0
    mesh_3DvtkDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
    mesh_3DvtkDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
    mesh_3DvtkDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
    mesh_3DvtkDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.EnableCustomRange = 0
    mesh_3DvtkDisplay.PolarAxes.CustomRange = [0.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.AutoPole = 1
    mesh_3DvtkDisplay.PolarAxes.PolarAxisVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.RadialAxesVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.DrawRadialGridlines = 1
    mesh_3DvtkDisplay.PolarAxes.PolarArcsVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.DrawPolarArcsGridlines = 1
    mesh_3DvtkDisplay.PolarAxes.NumberOfRadialAxes = 0
    mesh_3DvtkDisplay.PolarAxes.DeltaAngleRadialAxes = 45.0
    mesh_3DvtkDisplay.PolarAxes.NumberOfPolarAxes = 5
    mesh_3DvtkDisplay.PolarAxes.DeltaRangePolarAxes = 0.0
    mesh_3DvtkDisplay.PolarAxes.CustomMinRadius = 1
    mesh_3DvtkDisplay.PolarAxes.MinimumRadius = 0.0
    mesh_3DvtkDisplay.PolarAxes.CustomAngles = 1
    mesh_3DvtkDisplay.PolarAxes.MinimumAngle = 0.0
    mesh_3DvtkDisplay.PolarAxes.MaximumAngle = 90.0
    mesh_3DvtkDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
    mesh_3DvtkDisplay.PolarAxes.PolarArcResolutionPerDegree = 0.2
    mesh_3DvtkDisplay.PolarAxes.Ratio = 1.0
    mesh_3DvtkDisplay.PolarAxes.EnableOverallColor = 1
    mesh_3DvtkDisplay.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
    mesh_3DvtkDisplay.PolarAxes.PolarTitleOffset = [20.0, 20.0]
    mesh_3DvtkDisplay.PolarAxes.PolarLabelVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
    mesh_3DvtkDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
    mesh_3DvtkDisplay.PolarAxes.PolarLabelOffset = 10.0
    mesh_3DvtkDisplay.PolarAxes.PolarExponentOffset = 5.0
    mesh_3DvtkDisplay.PolarAxes.RadialTitleVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.RadialTitleFormat = '%-#3.1f'
    mesh_3DvtkDisplay.PolarAxes.RadialTitleLocation = 'Bottom'
    mesh_3DvtkDisplay.PolarAxes.RadialTitleOffset = [20.0, 0.0]
    mesh_3DvtkDisplay.PolarAxes.RadialUnitsVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.ScreenSize = 10.0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleFontFile = ''
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleBold = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleItalic = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleShadow = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTitleFontSize = 12
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelFontFile = ''
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelBold = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelItalic = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelShadow = 0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisLabelFontSize = 12
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextBold = 0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextItalic = 0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextShadow = 0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
    mesh_3DvtkDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
    mesh_3DvtkDisplay.PolarAxes.EnableDistanceLOD = 1
    mesh_3DvtkDisplay.PolarAxes.DistanceLODThreshold = 0.7
    mesh_3DvtkDisplay.PolarAxes.EnableViewAngleLOD = 1
    mesh_3DvtkDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
    mesh_3DvtkDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
    mesh_3DvtkDisplay.PolarAxes.PolarTicksVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
    mesh_3DvtkDisplay.PolarAxes.TickLocation = 'Both'
    mesh_3DvtkDisplay.PolarAxes.AxisTickVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.AxisMinorTickVisibility = 0
    mesh_3DvtkDisplay.PolarAxes.AxisTickMatchesPolarAxes = 1
    mesh_3DvtkDisplay.PolarAxes.DeltaRangeMajor = 1.0
    mesh_3DvtkDisplay.PolarAxes.DeltaRangeMinor = 0.5
    mesh_3DvtkDisplay.PolarAxes.ArcTickVisibility = 1
    mesh_3DvtkDisplay.PolarAxes.ArcMinorTickVisibility = 0
    mesh_3DvtkDisplay.PolarAxes.ArcTickMatchesRadialAxes = 1
    mesh_3DvtkDisplay.PolarAxes.DeltaAngleMajor = 10.0
    mesh_3DvtkDisplay.PolarAxes.DeltaAngleMinor = 5.0
    mesh_3DvtkDisplay.PolarAxes.TickRatioRadiusSize = 0.02
    mesh_3DvtkDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
    mesh_3DvtkDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
    mesh_3DvtkDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
    mesh_3DvtkDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
    mesh_3DvtkDisplay.PolarAxes.ArcMajorTickSize = 0.0
    mesh_3DvtkDisplay.PolarAxes.ArcTickRatioSize = 0.3
    mesh_3DvtkDisplay.PolarAxes.ArcMajorTickThickness = 1.0
    mesh_3DvtkDisplay.PolarAxes.ArcTickRatioThickness = 0.5
    mesh_3DvtkDisplay.PolarAxes.Use2DMode = 0
    mesh_3DvtkDisplay.PolarAxes.UseLogAxis = 0

    # reset view to fit data
    renderView1.ResetCamera(False, 0.9)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [0.03812262307586067, 2.28852714869904, 134.97937060780404]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [0.03812262307586067, 2.28852714869904, 134.97937060780404]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraParallelScale = 34.77515867297295

    renderView1.ResetActiveCameraToNegativeX()

    # reset view to fit data
    renderView1.ResetCamera(False, 0.9)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=mesh_3Dvtk)
    clip1.ClipType = 'Plane'
    clip1.HyperTreeGridClipper = 'Plane'
    clip1.Scalars = ['CELLS', 'Epi-Endo-Base-Wall-gmsh:bounding_entities']
    clip1.Value = 2.0
    clip1.Invert = 1
    clip1.Crinkleclip = 0
    clip1.Exact = 0

    # init the 'Plane' selected for 'ClipType'
    clip1.ClipType.Origin = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    clip1.ClipType.Normal = [1.0, 0.0, 0.0]
    clip1.ClipType.Offset = 0.0

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    clip1.HyperTreeGridClipper.Normal = [1.0, 0.0, 0.0]
    clip1.HyperTreeGridClipper.Offset = 0.0
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on clip1.ClipType
    clip1.ClipType.Origin = [0.03812262307586067, 2.28852714869904, clip_z]
    clip1.ClipType.Normal = [0.0, 0.0, 1.0]

    # show data in view
    clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    clip1Display.Selection = None
    clip1Display.Representation = 'Surface'
    clip1Display.ColorArrayName = [None, '']
    clip1Display.LookupTable = None
    clip1Display.MapScalars = 1
    clip1Display.MultiComponentsMapping = 0
    clip1Display.InterpolateScalarsBeforeMapping = 1
    clip1Display.UseNanColorForMissingArrays = 0
    clip1Display.Opacity = 1.0
    clip1Display.PointSize = 2.0
    clip1Display.LineWidth = 1.0
    clip1Display.RenderLinesAsTubes = 0
    clip1Display.RenderPointsAsSpheres = 0
    clip1Display.Interpolation = 'Gouraud'
    clip1Display.Specular = 0.0
    clip1Display.SpecularColor = [1.0, 1.0, 1.0]
    clip1Display.SpecularPower = 100.0
    clip1Display.Luminosity = 0.0
    clip1Display.Ambient = 0.0
    clip1Display.Diffuse = 1.0
    clip1Display.Roughness = 0.3
    clip1Display.Metallic = 0.0
    clip1Display.EdgeTint = [1.0, 1.0, 1.0]
    clip1Display.Anisotropy = 0.0
    clip1Display.AnisotropyRotation = 0.0
    clip1Display.BaseIOR = 1.5
    clip1Display.CoatStrength = 0.0
    clip1Display.CoatIOR = 2.0
    clip1Display.CoatRoughness = 0.0
    clip1Display.CoatColor = [1.0, 1.0, 1.0]
    clip1Display.SelectTCoordArray = 'None'
    clip1Display.SelectNormalArray = 'None'
    clip1Display.SelectTangentArray = 'None'
    clip1Display.Texture = None
    clip1Display.RepeatTextures = 1
    clip1Display.InterpolateTextures = 0
    clip1Display.SeamlessU = 0
    clip1Display.SeamlessV = 0
    clip1Display.UseMipmapTextures = 0
    clip1Display.ShowTexturesOnBackface = 1
    clip1Display.BaseColorTexture = None
    clip1Display.NormalTexture = None
    clip1Display.NormalScale = 1.0
    clip1Display.CoatNormalTexture = None
    clip1Display.CoatNormalScale = 1.0
    clip1Display.MaterialTexture = None
    clip1Display.OcclusionStrength = 1.0
    clip1Display.AnisotropyTexture = None
    clip1Display.EmissiveTexture = None
    clip1Display.EmissiveFactor = [1.0, 1.0, 1.0]
    clip1Display.FlipTextures = 0
    clip1Display.EdgeOpacity = 1.0
    clip1Display.BackfaceRepresentation = 'Follow Frontface'
    clip1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    clip1Display.BackfaceOpacity = 1.0
    clip1Display.Position = [0.0, 0.0, 0.0]
    clip1Display.Scale = [1.0, 1.0, 1.0]
    clip1Display.Orientation = [0.0, 0.0, 0.0]
    clip1Display.Origin = [0.0, 0.0, 0.0]
    clip1Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
    clip1Display.Pickable = 1
    clip1Display.Triangulate = 0
    clip1Display.UseShaderReplacements = 0
    clip1Display.ShaderReplacements = ''
    clip1Display.NonlinearSubdivisionLevel = 1
    clip1Display.MatchBoundariesIgnoringCellOrder = 0
    clip1Display.UseDataPartitions = 0
    clip1Display.OSPRayUseScaleArray = 'All Approximate'
    clip1Display.OSPRayScaleArray = 'gmsh:dim_tags'
    clip1Display.OSPRayScaleFunction = 'Piecewise Function'
    clip1Display.OSPRayMaterial = 'None'
    clip1Display.Assembly = ''
    clip1Display.BlockSelectors = ['/']
    clip1Display.BlockColors = []
    clip1Display.BlockOpacities = []
    clip1Display.Orient = 0
    clip1Display.OrientationMode = 'Direction'
    clip1Display.SelectOrientationVectors = 'None'
    clip1Display.Scaling = 0
    clip1Display.ScaleMode = 'No Data Scaling Off'
    clip1Display.ScaleFactor = 4.744257707130562
    clip1Display.SelectScaleArray = 'None'
    clip1Display.GlyphType = 'Arrow'
    clip1Display.UseGlyphTable = 0
    clip1Display.GlyphTableIndexArray = 'None'
    clip1Display.UseCompositeGlyphTable = 0
    clip1Display.UseGlyphCullingAndLOD = 0
    clip1Display.LODValues = []
    clip1Display.ColorByLODIndex = 0
    clip1Display.GaussianRadius = 0.2372128853565281
    clip1Display.ShaderPreset = 'Sphere'
    clip1Display.CustomTriangleScale = 3
    clip1Display.CustomShader = """ // This custom shader code define a gaussian blur
    // Please take a look into vtkSMPointGaussianRepresentation.cxx
    // for other custom shader examples
    //VTK::Color::Impl
    float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
    float gaussian = exp(-0.5*dist2);
    opacity = opacity*gaussian;
    """
    clip1Display.Emissive = 0
    clip1Display.ScaleByArray = 0
    clip1Display.SetScaleArray = ['POINTS', 'gmsh:dim_tags']
    clip1Display.ScaleArrayComponent = 'X'
    clip1Display.UseScaleFunction = 1
    clip1Display.ScaleTransferFunction = 'Piecewise Function'
    clip1Display.OpacityByArray = 0
    clip1Display.OpacityArray = ['POINTS', 'gmsh:dim_tags']
    clip1Display.OpacityArrayComponent = 'X'
    clip1Display.OpacityTransferFunction = 'Piecewise Function'
    clip1Display.DataAxesGrid = 'Grid Axes Representation'
    clip1Display.SelectionCellLabelBold = 0
    clip1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    clip1Display.SelectionCellLabelFontFamily = 'Arial'
    clip1Display.SelectionCellLabelFontFile = ''
    clip1Display.SelectionCellLabelFontSize = 18
    clip1Display.SelectionCellLabelItalic = 0
    clip1Display.SelectionCellLabelJustification = 'Left'
    clip1Display.SelectionCellLabelOpacity = 1.0
    clip1Display.SelectionCellLabelShadow = 0
    clip1Display.SelectionPointLabelBold = 0
    clip1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    clip1Display.SelectionPointLabelFontFamily = 'Arial'
    clip1Display.SelectionPointLabelFontFile = ''
    clip1Display.SelectionPointLabelFontSize = 18
    clip1Display.SelectionPointLabelItalic = 0
    clip1Display.SelectionPointLabelJustification = 'Left'
    clip1Display.SelectionPointLabelOpacity = 1.0
    clip1Display.SelectionPointLabelShadow = 0
    clip1Display.PolarAxes = 'Polar Axes Representation'
    clip1Display.ScalarOpacityFunction = None
    clip1Display.ScalarOpacityUnitDistance = 3.7183592969141275
    clip1Display.UseSeparateOpacityArray = 0
    clip1Display.OpacityArrayName = ['POINTS', 'gmsh:dim_tags']
    clip1Display.OpacityComponent = 'X'
    clip1Display.SelectMapper = 'Projected tetra'
    clip1Display.SamplingDimensions = [128, 128, 128]
    clip1Display.UseFloatingPointFrameBuffer = 1
    clip1Display.SelectInputVectors = ['POINTS', 'gmsh:dim_tags']
    clip1Display.NumberOfSteps = 40
    clip1Display.StepSize = 0.25
    clip1Display.NormalizeVectors = 1
    clip1Display.EnhancedLIC = 1
    clip1Display.ColorMode = 'Blend'
    clip1Display.LICIntensity = 0.8
    clip1Display.MapModeBias = 0.0
    clip1Display.EnhanceContrast = 'Off'
    clip1Display.LowLICContrastEnhancementFactor = 0.0
    clip1Display.HighLICContrastEnhancementFactor = 0.0
    clip1Display.LowColorContrastEnhancementFactor = 0.0
    clip1Display.HighColorContrastEnhancementFactor = 0.0
    clip1Display.AntiAlias = 0
    clip1Display.MaskOnSurface = 1
    clip1Display.MaskThreshold = 0.0
    clip1Display.MaskIntensity = 0.0
    clip1Display.MaskColor = [0.5, 0.5, 0.5]
    clip1Display.GenerateNoiseTexture = 0
    clip1Display.NoiseType = 'Gaussian'
    clip1Display.NoiseTextureSize = 128
    clip1Display.NoiseGrainSize = 2
    clip1Display.MinNoiseValue = 0.0
    clip1Display.MaxNoiseValue = 0.8
    clip1Display.NumberOfNoiseLevels = 1024
    clip1Display.ImpulseNoiseProbability = 1.0
    clip1Display.ImpulseNoiseBackgroundValue = 0.0
    clip1Display.NoiseGeneratorSeed = 1
    clip1Display.CompositeStrategy = 'AUTO'
    clip1Display.UseLICForLOD = 0
    clip1Display.WriteLog = ''

    # init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
    clip1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 7.0, 1.0, 0.5, 0.0]
    clip1Display.OSPRayScaleFunction.UseLogScale = 0

    # init the 'Arrow' selected for 'GlyphType'
    clip1Display.GlyphType.TipResolution = 6
    clip1Display.GlyphType.TipRadius = 0.1
    clip1Display.GlyphType.TipLength = 0.35
    clip1Display.GlyphType.ShaftResolution = 6
    clip1Display.GlyphType.ShaftRadius = 0.03
    clip1Display.GlyphType.Invert = 0

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    clip1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    clip1Display.ScaleTransferFunction.UseLogScale = 0

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    clip1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    clip1Display.OpacityTransferFunction.UseLogScale = 0

    # init the 'Grid Axes Representation' selected for 'DataAxesGrid'
    clip1Display.DataAxesGrid.XTitle = 'X Axis'
    clip1Display.DataAxesGrid.YTitle = 'Y Axis'
    clip1Display.DataAxesGrid.ZTitle = 'Z Axis'
    clip1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
    clip1Display.DataAxesGrid.XTitleFontFile = ''
    clip1Display.DataAxesGrid.XTitleBold = 0
    clip1Display.DataAxesGrid.XTitleItalic = 0
    clip1Display.DataAxesGrid.XTitleFontSize = 12
    clip1Display.DataAxesGrid.XTitleShadow = 0
    clip1Display.DataAxesGrid.XTitleOpacity = 1.0
    clip1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
    clip1Display.DataAxesGrid.YTitleFontFile = ''
    clip1Display.DataAxesGrid.YTitleBold = 0
    clip1Display.DataAxesGrid.YTitleItalic = 0
    clip1Display.DataAxesGrid.YTitleFontSize = 12
    clip1Display.DataAxesGrid.YTitleShadow = 0
    clip1Display.DataAxesGrid.YTitleOpacity = 1.0
    clip1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
    clip1Display.DataAxesGrid.ZTitleFontFile = ''
    clip1Display.DataAxesGrid.ZTitleBold = 0
    clip1Display.DataAxesGrid.ZTitleItalic = 0
    clip1Display.DataAxesGrid.ZTitleFontSize = 12
    clip1Display.DataAxesGrid.ZTitleShadow = 0
    clip1Display.DataAxesGrid.ZTitleOpacity = 1.0
    clip1Display.DataAxesGrid.FacesToRender = 63
    clip1Display.DataAxesGrid.CullBackface = 0
    clip1Display.DataAxesGrid.CullFrontface = 1
    clip1Display.DataAxesGrid.ShowGrid = 0
    clip1Display.DataAxesGrid.ShowEdges = 1
    clip1Display.DataAxesGrid.ShowTicks = 1
    clip1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
    clip1Display.DataAxesGrid.AxesToLabel = 63
    clip1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
    clip1Display.DataAxesGrid.XLabelFontFile = ''
    clip1Display.DataAxesGrid.XLabelBold = 0
    clip1Display.DataAxesGrid.XLabelItalic = 0
    clip1Display.DataAxesGrid.XLabelFontSize = 12
    clip1Display.DataAxesGrid.XLabelShadow = 0
    clip1Display.DataAxesGrid.XLabelOpacity = 1.0
    clip1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
    clip1Display.DataAxesGrid.YLabelFontFile = ''
    clip1Display.DataAxesGrid.YLabelBold = 0
    clip1Display.DataAxesGrid.YLabelItalic = 0
    clip1Display.DataAxesGrid.YLabelFontSize = 12
    clip1Display.DataAxesGrid.YLabelShadow = 0
    clip1Display.DataAxesGrid.YLabelOpacity = 1.0
    clip1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
    clip1Display.DataAxesGrid.ZLabelFontFile = ''
    clip1Display.DataAxesGrid.ZLabelBold = 0
    clip1Display.DataAxesGrid.ZLabelItalic = 0
    clip1Display.DataAxesGrid.ZLabelFontSize = 12
    clip1Display.DataAxesGrid.ZLabelShadow = 0
    clip1Display.DataAxesGrid.ZLabelOpacity = 1.0
    clip1Display.DataAxesGrid.XAxisNotation = 'Mixed'
    clip1Display.DataAxesGrid.XAxisPrecision = 2
    clip1Display.DataAxesGrid.XAxisUseCustomLabels = 0
    clip1Display.DataAxesGrid.XAxisLabels = []
    clip1Display.DataAxesGrid.YAxisNotation = 'Mixed'
    clip1Display.DataAxesGrid.YAxisPrecision = 2
    clip1Display.DataAxesGrid.YAxisUseCustomLabels = 0
    clip1Display.DataAxesGrid.YAxisLabels = []
    clip1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
    clip1Display.DataAxesGrid.ZAxisPrecision = 2
    clip1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
    clip1Display.DataAxesGrid.ZAxisLabels = []
    clip1Display.DataAxesGrid.UseCustomBounds = 0
    clip1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # init the 'Polar Axes Representation' selected for 'PolarAxes'
    clip1Display.PolarAxes.Visibility = 0
    clip1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
    clip1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
    clip1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
    clip1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    clip1Display.PolarAxes.EnableCustomRange = 0
    clip1Display.PolarAxes.CustomRange = [0.0, 1.0]
    clip1Display.PolarAxes.AutoPole = 1
    clip1Display.PolarAxes.PolarAxisVisibility = 1
    clip1Display.PolarAxes.RadialAxesVisibility = 1
    clip1Display.PolarAxes.DrawRadialGridlines = 1
    clip1Display.PolarAxes.PolarArcsVisibility = 1
    clip1Display.PolarAxes.DrawPolarArcsGridlines = 1
    clip1Display.PolarAxes.NumberOfRadialAxes = 0
    clip1Display.PolarAxes.DeltaAngleRadialAxes = 45.0
    clip1Display.PolarAxes.NumberOfPolarAxes = 5
    clip1Display.PolarAxes.DeltaRangePolarAxes = 0.0
    clip1Display.PolarAxes.CustomMinRadius = 1
    clip1Display.PolarAxes.MinimumRadius = 0.0
    clip1Display.PolarAxes.CustomAngles = 1
    clip1Display.PolarAxes.MinimumAngle = 0.0
    clip1Display.PolarAxes.MaximumAngle = 90.0
    clip1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
    clip1Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
    clip1Display.PolarAxes.Ratio = 1.0
    clip1Display.PolarAxes.EnableOverallColor = 1
    clip1Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
    clip1Display.PolarAxes.PolarAxisTitleVisibility = 1
    clip1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
    clip1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
    clip1Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
    clip1Display.PolarAxes.PolarLabelVisibility = 1
    clip1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
    clip1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
    clip1Display.PolarAxes.PolarLabelOffset = 10.0
    clip1Display.PolarAxes.PolarExponentOffset = 5.0
    clip1Display.PolarAxes.RadialTitleVisibility = 1
    clip1Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
    clip1Display.PolarAxes.RadialTitleLocation = 'Bottom'
    clip1Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
    clip1Display.PolarAxes.RadialUnitsVisibility = 1
    clip1Display.PolarAxes.ScreenSize = 10.0
    clip1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
    clip1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
    clip1Display.PolarAxes.PolarAxisTitleFontFile = ''
    clip1Display.PolarAxes.PolarAxisTitleBold = 0
    clip1Display.PolarAxes.PolarAxisTitleItalic = 0
    clip1Display.PolarAxes.PolarAxisTitleShadow = 0
    clip1Display.PolarAxes.PolarAxisTitleFontSize = 12
    clip1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
    clip1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
    clip1Display.PolarAxes.PolarAxisLabelFontFile = ''
    clip1Display.PolarAxes.PolarAxisLabelBold = 0
    clip1Display.PolarAxes.PolarAxisLabelItalic = 0
    clip1Display.PolarAxes.PolarAxisLabelShadow = 0
    clip1Display.PolarAxes.PolarAxisLabelFontSize = 12
    clip1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
    clip1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
    clip1Display.PolarAxes.LastRadialAxisTextFontFile = ''
    clip1Display.PolarAxes.LastRadialAxisTextBold = 0
    clip1Display.PolarAxes.LastRadialAxisTextItalic = 0
    clip1Display.PolarAxes.LastRadialAxisTextShadow = 0
    clip1Display.PolarAxes.LastRadialAxisTextFontSize = 12
    clip1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
    clip1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
    clip1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    clip1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
    clip1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
    clip1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
    clip1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
    clip1Display.PolarAxes.EnableDistanceLOD = 1
    clip1Display.PolarAxes.DistanceLODThreshold = 0.7
    clip1Display.PolarAxes.EnableViewAngleLOD = 1
    clip1Display.PolarAxes.ViewAngleLODThreshold = 0.7
    clip1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
    clip1Display.PolarAxes.PolarTicksVisibility = 1
    clip1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
    clip1Display.PolarAxes.TickLocation = 'Both'
    clip1Display.PolarAxes.AxisTickVisibility = 1
    clip1Display.PolarAxes.AxisMinorTickVisibility = 0
    clip1Display.PolarAxes.AxisTickMatchesPolarAxes = 1
    clip1Display.PolarAxes.DeltaRangeMajor = 1.0
    clip1Display.PolarAxes.DeltaRangeMinor = 0.5
    clip1Display.PolarAxes.ArcTickVisibility = 1
    clip1Display.PolarAxes.ArcMinorTickVisibility = 0
    clip1Display.PolarAxes.ArcTickMatchesRadialAxes = 1
    clip1Display.PolarAxes.DeltaAngleMajor = 10.0
    clip1Display.PolarAxes.DeltaAngleMinor = 5.0
    clip1Display.PolarAxes.TickRatioRadiusSize = 0.02
    clip1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
    clip1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
    clip1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
    clip1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
    clip1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
    clip1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
    clip1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
    clip1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
    clip1Display.PolarAxes.ArcMajorTickSize = 0.0
    clip1Display.PolarAxes.ArcTickRatioSize = 0.3
    clip1Display.PolarAxes.ArcMajorTickThickness = 1.0
    clip1Display.PolarAxes.ArcTickRatioThickness = 0.5
    clip1Display.PolarAxes.Use2DMode = 0
    clip1Display.PolarAxes.UseLogAxis = 0

    # hide data in view
    Hide(mesh_3Dvtk, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=clip1.ClipType)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # create a new 'Clip'
    clip2 = Clip(registrationName='Clip2', Input=clip1)
    clip2.ClipType = 'Plane'
    clip2.HyperTreeGridClipper = 'Plane'
    clip2.Scalars = ['CELLS', 'Epi-Endo-Base-Wall-gmsh:bounding_entities']
    clip2.Value = 2.0
    clip2.Invert = 1
    clip2.Crinkleclip = 0
    clip2.Exact = 0

    # init the 'Plane' selected for 'ClipType'
    clip2.ClipType.Origin = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    clip2.ClipType.Normal = [1.0, 0.0, 0.0]
    clip2.ClipType.Offset = 0.0

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip2.HyperTreeGridClipper.Origin = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    clip2.HyperTreeGridClipper.Normal = [1.0, 0.0, 0.0]
    clip2.HyperTreeGridClipper.Offset = 0.0
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=clip2.ClipType)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on clip2.ClipType
    clip2.ClipType.Origin = [0.03812262307586067, 2.288527148699039, 0.6184751057046078]

    # show data in view
    clip2Display = Show(clip2, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    clip2Display.Selection = None
    clip2Display.Representation = 'Surface'
    clip2Display.ColorArrayName = [None, '']
    clip2Display.LookupTable = None
    clip2Display.MapScalars = 1
    clip2Display.MultiComponentsMapping = 0
    clip2Display.InterpolateScalarsBeforeMapping = 1
    clip2Display.UseNanColorForMissingArrays = 0
    clip2Display.Opacity = 1.0
    clip2Display.PointSize = 2.0
    clip2Display.LineWidth = 1.0
    clip2Display.RenderLinesAsTubes = 0
    clip2Display.RenderPointsAsSpheres = 0
    clip2Display.Interpolation = 'Gouraud'
    clip2Display.Specular = 0.0
    clip2Display.SpecularColor = [1.0, 1.0, 1.0]
    clip2Display.SpecularPower = 100.0
    clip2Display.Luminosity = 0.0
    clip2Display.Ambient = 0.0
    clip2Display.Diffuse = 1.0
    clip2Display.Roughness = 0.3
    clip2Display.Metallic = 0.0
    clip2Display.EdgeTint = [1.0, 1.0, 1.0]
    clip2Display.Anisotropy = 0.0
    clip2Display.AnisotropyRotation = 0.0
    clip2Display.BaseIOR = 1.5
    clip2Display.CoatStrength = 0.0
    clip2Display.CoatIOR = 2.0
    clip2Display.CoatRoughness = 0.0
    clip2Display.CoatColor = [1.0, 1.0, 1.0]
    clip2Display.SelectTCoordArray = 'None'
    clip2Display.SelectNormalArray = 'None'
    clip2Display.SelectTangentArray = 'None'
    clip2Display.Texture = None
    clip2Display.RepeatTextures = 1
    clip2Display.InterpolateTextures = 0
    clip2Display.SeamlessU = 0
    clip2Display.SeamlessV = 0
    clip2Display.UseMipmapTextures = 0
    clip2Display.ShowTexturesOnBackface = 1
    clip2Display.BaseColorTexture = None
    clip2Display.NormalTexture = None
    clip2Display.NormalScale = 1.0
    clip2Display.CoatNormalTexture = None
    clip2Display.CoatNormalScale = 1.0
    clip2Display.MaterialTexture = None
    clip2Display.OcclusionStrength = 1.0
    clip2Display.AnisotropyTexture = None
    clip2Display.EmissiveTexture = None
    clip2Display.EmissiveFactor = [1.0, 1.0, 1.0]
    clip2Display.FlipTextures = 0
    clip2Display.EdgeOpacity = 1.0
    clip2Display.BackfaceRepresentation = 'Follow Frontface'
    clip2Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    clip2Display.BackfaceOpacity = 1.0
    clip2Display.Position = [0.0, 0.0, 0.0]
    clip2Display.Scale = [1.0, 1.0, 1.0]
    clip2Display.Orientation = [0.0, 0.0, 0.0]
    clip2Display.Origin = [0.0, 0.0, 0.0]
    clip2Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
    clip2Display.Pickable = 1
    clip2Display.Triangulate = 0
    clip2Display.UseShaderReplacements = 0
    clip2Display.ShaderReplacements = ''
    clip2Display.NonlinearSubdivisionLevel = 1
    clip2Display.MatchBoundariesIgnoringCellOrder = 0
    clip2Display.UseDataPartitions = 0
    clip2Display.OSPRayUseScaleArray = 'All Approximate'
    clip2Display.OSPRayScaleArray = 'gmsh:dim_tags'
    clip2Display.OSPRayScaleFunction = 'Piecewise Function'
    clip2Display.OSPRayMaterial = 'None'
    clip2Display.Assembly = ''
    clip2Display.BlockSelectors = ['/']
    clip2Display.BlockColors = []
    clip2Display.BlockOpacities = []
    clip2Display.Orient = 0
    clip2Display.OrientationMode = 'Direction'
    clip2Display.SelectOrientationVectors = 'None'
    clip2Display.Scaling = 0
    clip2Display.ScaleMode = 'No Data Scaling Off'
    clip2Display.ScaleFactor = 4.744257707130562
    clip2Display.SelectScaleArray = 'None'
    clip2Display.GlyphType = 'Arrow'
    clip2Display.UseGlyphTable = 0
    clip2Display.GlyphTableIndexArray = 'None'
    clip2Display.UseCompositeGlyphTable = 0
    clip2Display.UseGlyphCullingAndLOD = 0
    clip2Display.LODValues = []
    clip2Display.ColorByLODIndex = 0
    clip2Display.GaussianRadius = 0.2372128853565281
    clip2Display.ShaderPreset = 'Sphere'
    clip2Display.CustomTriangleScale = 3
    clip2Display.CustomShader = """ // This custom shader code define a gaussian blur
    // Please take a look into vtkSMPointGaussianRepresentation.cxx
    // for other custom shader examples
    //VTK::Color::Impl
    float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
    float gaussian = exp(-0.5*dist2);
    opacity = opacity*gaussian;
    """
    clip2Display.Emissive = 0
    clip2Display.ScaleByArray = 0
    clip2Display.SetScaleArray = ['POINTS', 'gmsh:dim_tags']
    clip2Display.ScaleArrayComponent = 'X'
    clip2Display.UseScaleFunction = 1
    clip2Display.ScaleTransferFunction = 'Piecewise Function'
    clip2Display.OpacityByArray = 0
    clip2Display.OpacityArray = ['POINTS', 'gmsh:dim_tags']
    clip2Display.OpacityArrayComponent = 'X'
    clip2Display.OpacityTransferFunction = 'Piecewise Function'
    clip2Display.DataAxesGrid = 'Grid Axes Representation'
    clip2Display.SelectionCellLabelBold = 0
    clip2Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    clip2Display.SelectionCellLabelFontFamily = 'Arial'
    clip2Display.SelectionCellLabelFontFile = ''
    clip2Display.SelectionCellLabelFontSize = 18
    clip2Display.SelectionCellLabelItalic = 0
    clip2Display.SelectionCellLabelJustification = 'Left'
    clip2Display.SelectionCellLabelOpacity = 1.0
    clip2Display.SelectionCellLabelShadow = 0
    clip2Display.SelectionPointLabelBold = 0
    clip2Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    clip2Display.SelectionPointLabelFontFamily = 'Arial'
    clip2Display.SelectionPointLabelFontFile = ''
    clip2Display.SelectionPointLabelFontSize = 18
    clip2Display.SelectionPointLabelItalic = 0
    clip2Display.SelectionPointLabelJustification = 'Left'
    clip2Display.SelectionPointLabelOpacity = 1.0
    clip2Display.SelectionPointLabelShadow = 0
    clip2Display.PolarAxes = 'Polar Axes Representation'
    clip2Display.ScalarOpacityFunction = None
    clip2Display.ScalarOpacityUnitDistance = 4.1169253690148135
    clip2Display.UseSeparateOpacityArray = 0
    clip2Display.OpacityArrayName = ['POINTS', 'gmsh:dim_tags']
    clip2Display.OpacityComponent = 'X'
    clip2Display.SelectMapper = 'Projected tetra'
    clip2Display.SamplingDimensions = [128, 128, 128]
    clip2Display.UseFloatingPointFrameBuffer = 1
    clip2Display.SelectInputVectors = ['POINTS', 'gmsh:dim_tags']
    clip2Display.NumberOfSteps = 40
    clip2Display.StepSize = 0.25
    clip2Display.NormalizeVectors = 1
    clip2Display.EnhancedLIC = 1
    clip2Display.ColorMode = 'Blend'
    clip2Display.LICIntensity = 0.8
    clip2Display.MapModeBias = 0.0
    clip2Display.EnhanceContrast = 'Off'
    clip2Display.LowLICContrastEnhancementFactor = 0.0
    clip2Display.HighLICContrastEnhancementFactor = 0.0
    clip2Display.LowColorContrastEnhancementFactor = 0.0
    clip2Display.HighColorContrastEnhancementFactor = 0.0
    clip2Display.AntiAlias = 0
    clip2Display.MaskOnSurface = 1
    clip2Display.MaskThreshold = 0.0
    clip2Display.MaskIntensity = 0.0
    clip2Display.MaskColor = [0.5, 0.5, 0.5]
    clip2Display.GenerateNoiseTexture = 0
    clip2Display.NoiseType = 'Gaussian'
    clip2Display.NoiseTextureSize = 128
    clip2Display.NoiseGrainSize = 2
    clip2Display.MinNoiseValue = 0.0
    clip2Display.MaxNoiseValue = 0.8
    clip2Display.NumberOfNoiseLevels = 1024
    clip2Display.ImpulseNoiseProbability = 1.0
    clip2Display.ImpulseNoiseBackgroundValue = 0.0
    clip2Display.NoiseGeneratorSeed = 1
    clip2Display.CompositeStrategy = 'AUTO'
    clip2Display.UseLICForLOD = 0
    clip2Display.WriteLog = ''

    # init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
    clip2Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 7.0, 1.0, 0.5, 0.0]
    clip2Display.OSPRayScaleFunction.UseLogScale = 0

    # init the 'Arrow' selected for 'GlyphType'
    clip2Display.GlyphType.TipResolution = 6
    clip2Display.GlyphType.TipRadius = 0.1
    clip2Display.GlyphType.TipLength = 0.35
    clip2Display.GlyphType.ShaftResolution = 6
    clip2Display.GlyphType.ShaftRadius = 0.03
    clip2Display.GlyphType.Invert = 0

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    clip2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    clip2Display.ScaleTransferFunction.UseLogScale = 0

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    clip2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    clip2Display.OpacityTransferFunction.UseLogScale = 0

    # init the 'Grid Axes Representation' selected for 'DataAxesGrid'
    clip2Display.DataAxesGrid.XTitle = 'X Axis'
    clip2Display.DataAxesGrid.YTitle = 'Y Axis'
    clip2Display.DataAxesGrid.ZTitle = 'Z Axis'
    clip2Display.DataAxesGrid.XTitleFontFamily = 'Arial'
    clip2Display.DataAxesGrid.XTitleFontFile = ''
    clip2Display.DataAxesGrid.XTitleBold = 0
    clip2Display.DataAxesGrid.XTitleItalic = 0
    clip2Display.DataAxesGrid.XTitleFontSize = 12
    clip2Display.DataAxesGrid.XTitleShadow = 0
    clip2Display.DataAxesGrid.XTitleOpacity = 1.0
    clip2Display.DataAxesGrid.YTitleFontFamily = 'Arial'
    clip2Display.DataAxesGrid.YTitleFontFile = ''
    clip2Display.DataAxesGrid.YTitleBold = 0
    clip2Display.DataAxesGrid.YTitleItalic = 0
    clip2Display.DataAxesGrid.YTitleFontSize = 12
    clip2Display.DataAxesGrid.YTitleShadow = 0
    clip2Display.DataAxesGrid.YTitleOpacity = 1.0
    clip2Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
    clip2Display.DataAxesGrid.ZTitleFontFile = ''
    clip2Display.DataAxesGrid.ZTitleBold = 0
    clip2Display.DataAxesGrid.ZTitleItalic = 0
    clip2Display.DataAxesGrid.ZTitleFontSize = 12
    clip2Display.DataAxesGrid.ZTitleShadow = 0
    clip2Display.DataAxesGrid.ZTitleOpacity = 1.0
    clip2Display.DataAxesGrid.FacesToRender = 63
    clip2Display.DataAxesGrid.CullBackface = 0
    clip2Display.DataAxesGrid.CullFrontface = 1
    clip2Display.DataAxesGrid.ShowGrid = 0
    clip2Display.DataAxesGrid.ShowEdges = 1
    clip2Display.DataAxesGrid.ShowTicks = 1
    clip2Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
    clip2Display.DataAxesGrid.AxesToLabel = 63
    clip2Display.DataAxesGrid.XLabelFontFamily = 'Arial'
    clip2Display.DataAxesGrid.XLabelFontFile = ''
    clip2Display.DataAxesGrid.XLabelBold = 0
    clip2Display.DataAxesGrid.XLabelItalic = 0
    clip2Display.DataAxesGrid.XLabelFontSize = 12
    clip2Display.DataAxesGrid.XLabelShadow = 0
    clip2Display.DataAxesGrid.XLabelOpacity = 1.0
    clip2Display.DataAxesGrid.YLabelFontFamily = 'Arial'
    clip2Display.DataAxesGrid.YLabelFontFile = ''
    clip2Display.DataAxesGrid.YLabelBold = 0
    clip2Display.DataAxesGrid.YLabelItalic = 0
    clip2Display.DataAxesGrid.YLabelFontSize = 12
    clip2Display.DataAxesGrid.YLabelShadow = 0
    clip2Display.DataAxesGrid.YLabelOpacity = 1.0
    clip2Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
    clip2Display.DataAxesGrid.ZLabelFontFile = ''
    clip2Display.DataAxesGrid.ZLabelBold = 0
    clip2Display.DataAxesGrid.ZLabelItalic = 0
    clip2Display.DataAxesGrid.ZLabelFontSize = 12
    clip2Display.DataAxesGrid.ZLabelShadow = 0
    clip2Display.DataAxesGrid.ZLabelOpacity = 1.0
    clip2Display.DataAxesGrid.XAxisNotation = 'Mixed'
    clip2Display.DataAxesGrid.XAxisPrecision = 2
    clip2Display.DataAxesGrid.XAxisUseCustomLabels = 0
    clip2Display.DataAxesGrid.XAxisLabels = []
    clip2Display.DataAxesGrid.YAxisNotation = 'Mixed'
    clip2Display.DataAxesGrid.YAxisPrecision = 2
    clip2Display.DataAxesGrid.YAxisUseCustomLabels = 0
    clip2Display.DataAxesGrid.YAxisLabels = []
    clip2Display.DataAxesGrid.ZAxisNotation = 'Mixed'
    clip2Display.DataAxesGrid.ZAxisPrecision = 2
    clip2Display.DataAxesGrid.ZAxisUseCustomLabels = 0
    clip2Display.DataAxesGrid.ZAxisLabels = []
    clip2Display.DataAxesGrid.UseCustomBounds = 0
    clip2Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # init the 'Polar Axes Representation' selected for 'PolarAxes'
    clip2Display.PolarAxes.Visibility = 0
    clip2Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
    clip2Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
    clip2Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
    clip2Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    clip2Display.PolarAxes.EnableCustomRange = 0
    clip2Display.PolarAxes.CustomRange = [0.0, 1.0]
    clip2Display.PolarAxes.AutoPole = 1
    clip2Display.PolarAxes.PolarAxisVisibility = 1
    clip2Display.PolarAxes.RadialAxesVisibility = 1
    clip2Display.PolarAxes.DrawRadialGridlines = 1
    clip2Display.PolarAxes.PolarArcsVisibility = 1
    clip2Display.PolarAxes.DrawPolarArcsGridlines = 1
    clip2Display.PolarAxes.NumberOfRadialAxes = 0
    clip2Display.PolarAxes.DeltaAngleRadialAxes = 45.0
    clip2Display.PolarAxes.NumberOfPolarAxes = 5
    clip2Display.PolarAxes.DeltaRangePolarAxes = 0.0
    clip2Display.PolarAxes.CustomMinRadius = 1
    clip2Display.PolarAxes.MinimumRadius = 0.0
    clip2Display.PolarAxes.CustomAngles = 1
    clip2Display.PolarAxes.MinimumAngle = 0.0
    clip2Display.PolarAxes.MaximumAngle = 90.0
    clip2Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
    clip2Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
    clip2Display.PolarAxes.Ratio = 1.0
    clip2Display.PolarAxes.EnableOverallColor = 1
    clip2Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
    clip2Display.PolarAxes.PolarAxisTitleVisibility = 1
    clip2Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
    clip2Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
    clip2Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
    clip2Display.PolarAxes.PolarLabelVisibility = 1
    clip2Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
    clip2Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
    clip2Display.PolarAxes.PolarLabelOffset = 10.0
    clip2Display.PolarAxes.PolarExponentOffset = 5.0
    clip2Display.PolarAxes.RadialTitleVisibility = 1
    clip2Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
    clip2Display.PolarAxes.RadialTitleLocation = 'Bottom'
    clip2Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
    clip2Display.PolarAxes.RadialUnitsVisibility = 1
    clip2Display.PolarAxes.ScreenSize = 10.0
    clip2Display.PolarAxes.PolarAxisTitleOpacity = 1.0
    clip2Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
    clip2Display.PolarAxes.PolarAxisTitleFontFile = ''
    clip2Display.PolarAxes.PolarAxisTitleBold = 0
    clip2Display.PolarAxes.PolarAxisTitleItalic = 0
    clip2Display.PolarAxes.PolarAxisTitleShadow = 0
    clip2Display.PolarAxes.PolarAxisTitleFontSize = 12
    clip2Display.PolarAxes.PolarAxisLabelOpacity = 1.0
    clip2Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
    clip2Display.PolarAxes.PolarAxisLabelFontFile = ''
    clip2Display.PolarAxes.PolarAxisLabelBold = 0
    clip2Display.PolarAxes.PolarAxisLabelItalic = 0
    clip2Display.PolarAxes.PolarAxisLabelShadow = 0
    clip2Display.PolarAxes.PolarAxisLabelFontSize = 12
    clip2Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
    clip2Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
    clip2Display.PolarAxes.LastRadialAxisTextFontFile = ''
    clip2Display.PolarAxes.LastRadialAxisTextBold = 0
    clip2Display.PolarAxes.LastRadialAxisTextItalic = 0
    clip2Display.PolarAxes.LastRadialAxisTextShadow = 0
    clip2Display.PolarAxes.LastRadialAxisTextFontSize = 12
    clip2Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
    clip2Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
    clip2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    clip2Display.PolarAxes.SecondaryRadialAxesTextBold = 0
    clip2Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
    clip2Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
    clip2Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
    clip2Display.PolarAxes.EnableDistanceLOD = 1
    clip2Display.PolarAxes.DistanceLODThreshold = 0.7
    clip2Display.PolarAxes.EnableViewAngleLOD = 1
    clip2Display.PolarAxes.ViewAngleLODThreshold = 0.7
    clip2Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
    clip2Display.PolarAxes.PolarTicksVisibility = 1
    clip2Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
    clip2Display.PolarAxes.TickLocation = 'Both'
    clip2Display.PolarAxes.AxisTickVisibility = 1
    clip2Display.PolarAxes.AxisMinorTickVisibility = 0
    clip2Display.PolarAxes.AxisTickMatchesPolarAxes = 1
    clip2Display.PolarAxes.DeltaRangeMajor = 1.0
    clip2Display.PolarAxes.DeltaRangeMinor = 0.5
    clip2Display.PolarAxes.ArcTickVisibility = 1
    clip2Display.PolarAxes.ArcMinorTickVisibility = 0
    clip2Display.PolarAxes.ArcTickMatchesRadialAxes = 1
    clip2Display.PolarAxes.DeltaAngleMajor = 10.0
    clip2Display.PolarAxes.DeltaAngleMinor = 5.0
    clip2Display.PolarAxes.TickRatioRadiusSize = 0.02
    clip2Display.PolarAxes.PolarAxisMajorTickSize = 0.0
    clip2Display.PolarAxes.PolarAxisTickRatioSize = 0.3
    clip2Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
    clip2Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
    clip2Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
    clip2Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
    clip2Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
    clip2Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
    clip2Display.PolarAxes.ArcMajorTickSize = 0.0
    clip2Display.PolarAxes.ArcTickRatioSize = 0.3
    clip2Display.PolarAxes.ArcMajorTickThickness = 1.0
    clip2Display.PolarAxes.ArcTickRatioThickness = 0.5
    clip2Display.PolarAxes.Use2DMode = 0
    clip2Display.PolarAxes.UseLogAxis = 0

    # hide data in view
    Hide(clip1, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # set active source
    SetActiveSource(clip1)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # create a new 'Slice'
    slice1 = Slice(registrationName='Slice1', Input=clip1)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.UseDual = 0
    slice1.Crinkleslice = 0
    slice1.Triangulatetheslice = 1
    slice1.SliceOffsetValues = [0.0]
    slice1.PointMergeMethod = 'Uniform Binning'

    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = [0.03912262307586067, 2.28852714869904, 0.6184751057046096]
    slice1.SliceType.Normal = [1.0, 0.0, 0.0]
    slice1.SliceType.Offset = 0.0

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice1.HyperTreeGridSlicer.Origin = [0.03912262307586067, 2.28852714869904, 0.6184751057046096]
    slice1.HyperTreeGridSlicer.Normal = [1.0, 0.0, 0.0]
    slice1.HyperTreeGridSlicer.Offset = 0.0

    # init the 'Uniform Binning' selected for 'PointMergeMethod'
    slice1.PointMergeMethod.Divisions = [50, 50, 50]
    slice1.PointMergeMethod.Numberofpointsperbucket = 8
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # toggle interactive widget visibility (only when running from the GUI)
    HideInteractiveWidgets(proxy=slice1.SliceType)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on slice1.SliceType
    slice1.SliceType.Origin = [0.04012262307586067, 2.288527148699039, 0.6184751057046078]

    # show data in view
    slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice1Display.Selection = None
    slice1Display.Representation = 'Surface'
    slice1Display.ColorArrayName = [None, '']
    slice1Display.LookupTable = None
    slice1Display.MapScalars = 1
    slice1Display.MultiComponentsMapping = 0
    slice1Display.InterpolateScalarsBeforeMapping = 1
    slice1Display.UseNanColorForMissingArrays = 0
    slice1Display.Opacity = 1.0
    slice1Display.PointSize = 2.0
    slice1Display.LineWidth = 1.0
    slice1Display.RenderLinesAsTubes = 0
    slice1Display.RenderPointsAsSpheres = 0
    slice1Display.Interpolation = 'Gouraud'
    slice1Display.Specular = 0.0
    slice1Display.SpecularColor = [1.0, 1.0, 1.0]
    slice1Display.SpecularPower = 100.0
    slice1Display.Luminosity = 0.0
    slice1Display.Ambient = 0.0
    slice1Display.Diffuse = 1.0
    slice1Display.Roughness = 0.3
    slice1Display.Metallic = 0.0
    slice1Display.EdgeTint = [1.0, 1.0, 1.0]
    slice1Display.Anisotropy = 0.0
    slice1Display.AnisotropyRotation = 0.0
    slice1Display.BaseIOR = 1.5
    slice1Display.CoatStrength = 0.0
    slice1Display.CoatIOR = 2.0
    slice1Display.CoatRoughness = 0.0
    slice1Display.CoatColor = [1.0, 1.0, 1.0]
    slice1Display.SelectTCoordArray = 'None'
    slice1Display.SelectNormalArray = 'None'
    slice1Display.SelectTangentArray = 'None'
    slice1Display.Texture = None
    slice1Display.RepeatTextures = 1
    slice1Display.InterpolateTextures = 0
    slice1Display.SeamlessU = 0
    slice1Display.SeamlessV = 0
    slice1Display.UseMipmapTextures = 0
    slice1Display.ShowTexturesOnBackface = 1
    slice1Display.BaseColorTexture = None
    slice1Display.NormalTexture = None
    slice1Display.NormalScale = 1.0
    slice1Display.CoatNormalTexture = None
    slice1Display.CoatNormalScale = 1.0
    slice1Display.MaterialTexture = None
    slice1Display.OcclusionStrength = 1.0
    slice1Display.AnisotropyTexture = None
    slice1Display.EmissiveTexture = None
    slice1Display.EmissiveFactor = [1.0, 1.0, 1.0]
    slice1Display.FlipTextures = 0
    slice1Display.EdgeOpacity = 1.0
    slice1Display.BackfaceRepresentation = 'Follow Frontface'
    slice1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
    slice1Display.BackfaceOpacity = 1.0
    slice1Display.Position = [0.0, 0.0, 0.0]
    slice1Display.Scale = [1.0, 1.0, 1.0]
    slice1Display.Orientation = [0.0, 0.0, 0.0]
    slice1Display.Origin = [0.0, 0.0, 0.0]
    slice1Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
    slice1Display.Pickable = 1
    slice1Display.Triangulate = 0
    slice1Display.UseShaderReplacements = 0
    slice1Display.ShaderReplacements = ''
    slice1Display.NonlinearSubdivisionLevel = 1
    slice1Display.MatchBoundariesIgnoringCellOrder = 0
    slice1Display.UseDataPartitions = 0
    slice1Display.OSPRayUseScaleArray = 'All Approximate'
    slice1Display.OSPRayScaleArray = 'gmsh:dim_tags'
    slice1Display.OSPRayScaleFunction = 'Piecewise Function'
    slice1Display.OSPRayMaterial = 'None'
    slice1Display.Assembly = ''
    slice1Display.BlockSelectors = ['/']
    slice1Display.BlockColors = []
    slice1Display.BlockOpacities = []
    slice1Display.Orient = 0
    slice1Display.OrientationMode = 'Direction'
    slice1Display.SelectOrientationVectors = 'None'
    slice1Display.Scaling = 0
    slice1Display.ScaleMode = 'No Data Scaling Off'
    slice1Display.ScaleFactor = 4.734316674383075
    slice1Display.SelectScaleArray = 'None'
    slice1Display.GlyphType = 'Arrow'
    slice1Display.UseGlyphTable = 0
    slice1Display.GlyphTableIndexArray = 'None'
    slice1Display.UseCompositeGlyphTable = 0
    slice1Display.UseGlyphCullingAndLOD = 0
    slice1Display.LODValues = []
    slice1Display.ColorByLODIndex = 0
    slice1Display.GaussianRadius = 0.23671583371915375
    slice1Display.ShaderPreset = 'Sphere'
    slice1Display.CustomTriangleScale = 3
    slice1Display.CustomShader = """ // This custom shader code define a gaussian blur
    // Please take a look into vtkSMPointGaussianRepresentation.cxx
    // for other custom shader examples
    //VTK::Color::Impl
    float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
    float gaussian = exp(-0.5*dist2);
    opacity = opacity*gaussian;
    """
    slice1Display.Emissive = 0
    slice1Display.ScaleByArray = 0
    slice1Display.SetScaleArray = ['POINTS', 'gmsh:dim_tags']
    slice1Display.ScaleArrayComponent = 'X'
    slice1Display.UseScaleFunction = 1
    slice1Display.ScaleTransferFunction = 'Piecewise Function'
    slice1Display.OpacityByArray = 0
    slice1Display.OpacityArray = ['POINTS', 'gmsh:dim_tags']
    slice1Display.OpacityArrayComponent = 'X'
    slice1Display.OpacityTransferFunction = 'Piecewise Function'
    slice1Display.DataAxesGrid = 'Grid Axes Representation'
    slice1Display.SelectionCellLabelBold = 0
    slice1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
    slice1Display.SelectionCellLabelFontFamily = 'Arial'
    slice1Display.SelectionCellLabelFontFile = ''
    slice1Display.SelectionCellLabelFontSize = 18
    slice1Display.SelectionCellLabelItalic = 0
    slice1Display.SelectionCellLabelJustification = 'Left'
    slice1Display.SelectionCellLabelOpacity = 1.0
    slice1Display.SelectionCellLabelShadow = 0
    slice1Display.SelectionPointLabelBold = 0
    slice1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
    slice1Display.SelectionPointLabelFontFamily = 'Arial'
    slice1Display.SelectionPointLabelFontFile = ''
    slice1Display.SelectionPointLabelFontSize = 18
    slice1Display.SelectionPointLabelItalic = 0
    slice1Display.SelectionPointLabelJustification = 'Left'
    slice1Display.SelectionPointLabelOpacity = 1.0
    slice1Display.SelectionPointLabelShadow = 0
    slice1Display.PolarAxes = 'Polar Axes Representation'
    slice1Display.SelectInputVectors = ['POINTS', 'gmsh:dim_tags']
    slice1Display.NumberOfSteps = 40
    slice1Display.StepSize = 0.25
    slice1Display.NormalizeVectors = 1
    slice1Display.EnhancedLIC = 1
    slice1Display.ColorMode = 'Blend'
    slice1Display.LICIntensity = 0.8
    slice1Display.MapModeBias = 0.0
    slice1Display.EnhanceContrast = 'Off'
    slice1Display.LowLICContrastEnhancementFactor = 0.0
    slice1Display.HighLICContrastEnhancementFactor = 0.0
    slice1Display.LowColorContrastEnhancementFactor = 0.0
    slice1Display.HighColorContrastEnhancementFactor = 0.0
    slice1Display.AntiAlias = 0
    slice1Display.MaskOnSurface = 1
    slice1Display.MaskThreshold = 0.0
    slice1Display.MaskIntensity = 0.0
    slice1Display.MaskColor = [0.5, 0.5, 0.5]
    slice1Display.GenerateNoiseTexture = 0
    slice1Display.NoiseType = 'Gaussian'
    slice1Display.NoiseTextureSize = 128
    slice1Display.NoiseGrainSize = 2
    slice1Display.MinNoiseValue = 0.0
    slice1Display.MaxNoiseValue = 0.8
    slice1Display.NumberOfNoiseLevels = 1024
    slice1Display.ImpulseNoiseProbability = 1.0
    slice1Display.ImpulseNoiseBackgroundValue = 0.0
    slice1Display.NoiseGeneratorSeed = 1
    slice1Display.CompositeStrategy = 'AUTO'
    slice1Display.UseLICForLOD = 0
    slice1Display.WriteLog = ''

    # init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
    slice1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 7.0, 1.0, 0.5, 0.0]
    slice1Display.OSPRayScaleFunction.UseLogScale = 0

    # init the 'Arrow' selected for 'GlyphType'
    slice1Display.GlyphType.TipResolution = 6
    slice1Display.GlyphType.TipRadius = 0.1
    slice1Display.GlyphType.TipLength = 0.35
    slice1Display.GlyphType.ShaftResolution = 6
    slice1Display.GlyphType.ShaftRadius = 0.03
    slice1Display.GlyphType.Invert = 0

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    slice1Display.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    slice1Display.ScaleTransferFunction.UseLogScale = 0

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    slice1Display.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
    slice1Display.OpacityTransferFunction.UseLogScale = 0

    # init the 'Grid Axes Representation' selected for 'DataAxesGrid'
    slice1Display.DataAxesGrid.XTitle = 'X Axis'
    slice1Display.DataAxesGrid.YTitle = 'Y Axis'
    slice1Display.DataAxesGrid.ZTitle = 'Z Axis'
    slice1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
    slice1Display.DataAxesGrid.XTitleFontFile = ''
    slice1Display.DataAxesGrid.XTitleBold = 0
    slice1Display.DataAxesGrid.XTitleItalic = 0
    slice1Display.DataAxesGrid.XTitleFontSize = 12
    slice1Display.DataAxesGrid.XTitleShadow = 0
    slice1Display.DataAxesGrid.XTitleOpacity = 1.0
    slice1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
    slice1Display.DataAxesGrid.YTitleFontFile = ''
    slice1Display.DataAxesGrid.YTitleBold = 0
    slice1Display.DataAxesGrid.YTitleItalic = 0
    slice1Display.DataAxesGrid.YTitleFontSize = 12
    slice1Display.DataAxesGrid.YTitleShadow = 0
    slice1Display.DataAxesGrid.YTitleOpacity = 1.0
    slice1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
    slice1Display.DataAxesGrid.ZTitleFontFile = ''
    slice1Display.DataAxesGrid.ZTitleBold = 0
    slice1Display.DataAxesGrid.ZTitleItalic = 0
    slice1Display.DataAxesGrid.ZTitleFontSize = 12
    slice1Display.DataAxesGrid.ZTitleShadow = 0
    slice1Display.DataAxesGrid.ZTitleOpacity = 1.0
    slice1Display.DataAxesGrid.FacesToRender = 63
    slice1Display.DataAxesGrid.CullBackface = 0
    slice1Display.DataAxesGrid.CullFrontface = 1
    slice1Display.DataAxesGrid.ShowGrid = 0
    slice1Display.DataAxesGrid.ShowEdges = 1
    slice1Display.DataAxesGrid.ShowTicks = 1
    slice1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
    slice1Display.DataAxesGrid.AxesToLabel = 63
    slice1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
    slice1Display.DataAxesGrid.XLabelFontFile = ''
    slice1Display.DataAxesGrid.XLabelBold = 0
    slice1Display.DataAxesGrid.XLabelItalic = 0
    slice1Display.DataAxesGrid.XLabelFontSize = 12
    slice1Display.DataAxesGrid.XLabelShadow = 0
    slice1Display.DataAxesGrid.XLabelOpacity = 1.0
    slice1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
    slice1Display.DataAxesGrid.YLabelFontFile = ''
    slice1Display.DataAxesGrid.YLabelBold = 0
    slice1Display.DataAxesGrid.YLabelItalic = 0
    slice1Display.DataAxesGrid.YLabelFontSize = 12
    slice1Display.DataAxesGrid.YLabelShadow = 0
    slice1Display.DataAxesGrid.YLabelOpacity = 1.0
    slice1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
    slice1Display.DataAxesGrid.ZLabelFontFile = ''
    slice1Display.DataAxesGrid.ZLabelBold = 0
    slice1Display.DataAxesGrid.ZLabelItalic = 0
    slice1Display.DataAxesGrid.ZLabelFontSize = 12
    slice1Display.DataAxesGrid.ZLabelShadow = 0
    slice1Display.DataAxesGrid.ZLabelOpacity = 1.0
    slice1Display.DataAxesGrid.XAxisNotation = 'Mixed'
    slice1Display.DataAxesGrid.XAxisPrecision = 2
    slice1Display.DataAxesGrid.XAxisUseCustomLabels = 0
    slice1Display.DataAxesGrid.XAxisLabels = []
    slice1Display.DataAxesGrid.YAxisNotation = 'Mixed'
    slice1Display.DataAxesGrid.YAxisPrecision = 2
    slice1Display.DataAxesGrid.YAxisUseCustomLabels = 0
    slice1Display.DataAxesGrid.YAxisLabels = []
    slice1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
    slice1Display.DataAxesGrid.ZAxisPrecision = 2
    slice1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
    slice1Display.DataAxesGrid.ZAxisLabels = []
    slice1Display.DataAxesGrid.UseCustomBounds = 0
    slice1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # init the 'Polar Axes Representation' selected for 'PolarAxes'
    slice1Display.PolarAxes.Visibility = 0
    slice1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
    slice1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
    slice1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
    slice1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    slice1Display.PolarAxes.EnableCustomRange = 0
    slice1Display.PolarAxes.CustomRange = [0.0, 1.0]
    slice1Display.PolarAxes.AutoPole = 1
    slice1Display.PolarAxes.PolarAxisVisibility = 1
    slice1Display.PolarAxes.RadialAxesVisibility = 1
    slice1Display.PolarAxes.DrawRadialGridlines = 1
    slice1Display.PolarAxes.PolarArcsVisibility = 1
    slice1Display.PolarAxes.DrawPolarArcsGridlines = 1
    slice1Display.PolarAxes.NumberOfRadialAxes = 0
    slice1Display.PolarAxes.DeltaAngleRadialAxes = 45.0
    slice1Display.PolarAxes.NumberOfPolarAxes = 5
    slice1Display.PolarAxes.DeltaRangePolarAxes = 0.0
    slice1Display.PolarAxes.CustomMinRadius = 1
    slice1Display.PolarAxes.MinimumRadius = 0.0
    slice1Display.PolarAxes.CustomAngles = 1
    slice1Display.PolarAxes.MinimumAngle = 0.0
    slice1Display.PolarAxes.MaximumAngle = 90.0
    slice1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
    slice1Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
    slice1Display.PolarAxes.Ratio = 1.0
    slice1Display.PolarAxes.EnableOverallColor = 1
    slice1Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
    slice1Display.PolarAxes.PolarAxisTitleVisibility = 1
    slice1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
    slice1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
    slice1Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
    slice1Display.PolarAxes.PolarLabelVisibility = 1
    slice1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
    slice1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
    slice1Display.PolarAxes.PolarLabelOffset = 10.0
    slice1Display.PolarAxes.PolarExponentOffset = 5.0
    slice1Display.PolarAxes.RadialTitleVisibility = 1
    slice1Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
    slice1Display.PolarAxes.RadialTitleLocation = 'Bottom'
    slice1Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
    slice1Display.PolarAxes.RadialUnitsVisibility = 1
    slice1Display.PolarAxes.ScreenSize = 10.0
    slice1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
    slice1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
    slice1Display.PolarAxes.PolarAxisTitleFontFile = ''
    slice1Display.PolarAxes.PolarAxisTitleBold = 0
    slice1Display.PolarAxes.PolarAxisTitleItalic = 0
    slice1Display.PolarAxes.PolarAxisTitleShadow = 0
    slice1Display.PolarAxes.PolarAxisTitleFontSize = 12
    slice1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
    slice1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
    slice1Display.PolarAxes.PolarAxisLabelFontFile = ''
    slice1Display.PolarAxes.PolarAxisLabelBold = 0
    slice1Display.PolarAxes.PolarAxisLabelItalic = 0
    slice1Display.PolarAxes.PolarAxisLabelShadow = 0
    slice1Display.PolarAxes.PolarAxisLabelFontSize = 12
    slice1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
    slice1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
    slice1Display.PolarAxes.LastRadialAxisTextFontFile = ''
    slice1Display.PolarAxes.LastRadialAxisTextBold = 0
    slice1Display.PolarAxes.LastRadialAxisTextItalic = 0
    slice1Display.PolarAxes.LastRadialAxisTextShadow = 0
    slice1Display.PolarAxes.LastRadialAxisTextFontSize = 12
    slice1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
    slice1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
    slice1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    slice1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
    slice1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
    slice1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
    slice1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
    slice1Display.PolarAxes.EnableDistanceLOD = 1
    slice1Display.PolarAxes.DistanceLODThreshold = 0.7
    slice1Display.PolarAxes.EnableViewAngleLOD = 1
    slice1Display.PolarAxes.ViewAngleLODThreshold = 0.7
    slice1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
    slice1Display.PolarAxes.PolarTicksVisibility = 1
    slice1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
    slice1Display.PolarAxes.TickLocation = 'Both'
    slice1Display.PolarAxes.AxisTickVisibility = 1
    slice1Display.PolarAxes.AxisMinorTickVisibility = 0
    slice1Display.PolarAxes.AxisTickMatchesPolarAxes = 1
    slice1Display.PolarAxes.DeltaRangeMajor = 1.0
    slice1Display.PolarAxes.DeltaRangeMinor = 0.5
    slice1Display.PolarAxes.ArcTickVisibility = 1
    slice1Display.PolarAxes.ArcMinorTickVisibility = 0
    slice1Display.PolarAxes.ArcTickMatchesRadialAxes = 1
    slice1Display.PolarAxes.DeltaAngleMajor = 10.0
    slice1Display.PolarAxes.DeltaAngleMinor = 5.0
    slice1Display.PolarAxes.TickRatioRadiusSize = 0.02
    slice1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
    slice1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
    slice1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
    slice1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
    slice1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
    slice1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
    slice1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
    slice1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
    slice1Display.PolarAxes.ArcMajorTickSize = 0.0
    slice1Display.PolarAxes.ArcTickRatioSize = 0.3
    slice1Display.PolarAxes.ArcMajorTickThickness = 1.0
    slice1Display.PolarAxes.ArcTickRatioThickness = 0.5
    slice1Display.PolarAxes.Use2DMode = 0
    slice1Display.PolarAxes.UseLogAxis = 0

    # hide data in view
    Hide(clip1, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # change solid color
    slice1Display.AmbientColor = [0.9333333333333333, 0.5372549019607843, 0.4588235294117647]
    slice1Display.DiffuseColor = [0.9333333333333333, 0.5372549019607843, 0.4588235294117647]
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # set active source
    SetActiveSource(clip2)
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on renderView1
    renderView1.UseColorPaletteForBackground = 0
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on renderView1
    renderView1.Background = [1.0, 1.0, 1.0]
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on renderView1
    renderView1.OrientationAxesVisibility = 0
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [134.39901812517527, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraFocalPoint = [0.03812262307586067, 2.28852714869904, 0.6184751057046096]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 34.77515867297295

    # Properties modified on renderView1
    renderView1.CenterOfRotation = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraPosition = [90.10641775680212, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [90.10641775680212, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(2867, 954)

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414

    # save screenshot

    SaveScreenshot(filename=outname, viewOrLayout=renderView1, location=16, ImageResolution=[2867, 954],
        FontScaling='Scale fonts proportionally',
        OverrideColorPalette='',
        StereoMode='No change',
        TransparentBackground=1,
        SaveInBackground=0,
        EmbedParaViewState=0, 
        # PNG options
        CompressionLevel='0',
        MetaData=['Application', 'ParaView'])
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414
    # Adjust camera

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(2867, 954)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.CameraPosition = [111.51602192368996, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraFocalPoint = [-11.844078275997077, 0.473568536885999, 0.3397113171771746]
    renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraParallelScale = 26.386730030937414

    # # destroy clip1
    Delete(clip1)
    del clip1

    # # destroy clip1
    Delete(clip2)
    del clip2

    # destroy clip1
    Delete(slice1)
    del slice1

    # destroy clip1
    Delete(mesh_3DvtkDisplay)
    del mesh_3DvtkDisplay


    ##--------------------------------------------
    ## You may need to add some code at the end of this python script depending on your usage, eg:
    #
    ## Render all views to see them appears
    # RenderAllViews()
    #
    ## Interact with the view, usefull when running from pvpython
    # Interact()
    #
    ## Save a screenshot of the active view
    # SaveScreenshot("path/to/screenshot.png")
    #
    ## Save a screenshot of a layout (multiple splitted view)
    # SaveScreenshot("path/to/screenshot.png", GetLayout())
    #
    ## Save all "Extractors" from the pipeline browser
    # SaveExtracts()
    #
    ## Save a animation of the current active view
    # SaveAnimation()
    #
    ## Please refer to the documentation of paraview.simple
    ## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
    ##--------------------------------------------

import os
import re



dir = "/Users/javad/Docker/MitralDisjunction/00_data/modes_3/Results"
outdir = "/Users/javad/Docker/MitralDisjunction/00_data/modes_3/Images"
# mesh_folder = "06_Mesh"

# Ensure output directory exists
os.makedirs(outdir, exist_ok=True)


def extract_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')

sorted_dirs = sorted(
    [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))],
    key=extract_number
)
print(sorted_dirs)

# z_values = [25, 18.5, 25, 17, 25, 25, 25, 25, 25 ,17]

for i, folder  in enumerate(sorted_dirs):
    print(folder, i)
    fdir = os.path.join(dir, folder)
    outfname = f'{folder}.png'
    outpath = os.path.join(outdir, outfname)
    # plot(fdir, float(z_values[i]), outpath)
    plot(fdir, 30, outpath)

    print(outpath)

