import os
import unittest
import logging
import vtk, qt, ctk, slicer
import numpy as np
import SimpleITK as sitk
import sitkUtils
import SurfaceToolbox
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# StitchVolumes
#


class StitchVolumes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Stitch Volumes"
        self.parent.categories = ["Utilities"]
        self.parent.dependencies = ["SurfaceToolbox", "MarkupsToModel", "CropVolume"]
        self.parent.contributors = ["Mike Bindschadler (Seattle Children's Hospital)"]
        self.parent.helpText = """
    This module allows a user to stitch or blend together two or more image volumes. These image
    volumes should be positioned in a consistent world coordinate system before stitching (no 
    registration is performed by this module).  A bounding ROI can be supplied, or one can 
    be automatically generated to enclose all supplied inputs. Output volume resolution can be 
    specified or can mimic an input image. Overlapping regions can be smoothly blended or
    more discretely stitched together. A voxel threshold can also optionally be supplied; 
    voxels with values at or below the threshold are discarded before stitching. Finally,
    voxels which are inside the ROI but outside of all input image volumes, or in which all
    input voxels are below an applied threshold, are filled with a specified default voxel
    value. Unlike previous versions of this module, input images can be arranged arbitrarily
    in space, and multiple images overlapping the same region should be handled properly. 
    """
        self.parent.helpText += '<p>For more information see the <a href="https://github.com/PerkLab/SlicerSandbox#stitch-volumes">online documentation</a>.</p>'
        self.parent.acknowledgementText = """
    This work was funded by Seattle Children's Hospital.
    """


#
# StitchVolumesWidget
#


class StitchVolumesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self.updatingGUIFromParameterNode = False
        self._parameterNode = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/StitchVolumes.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        ui = self.ui  # abbreviated name

        # Set scene in MRML widgets. Make sure that in Qt designer
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create a new parameterNode
        # This parameterNode stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.logic = StitchVolumesLogic()
        ui.parameterNodeSelector.addAttribute(
            "vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName
        )
        self.setParameterNode(self.logic.getParameterNode())

        # Connections
        ui.parameterNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.setParameterNode
        )
        ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        ui.roiSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.volumeSelector1.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.volumeSelector2.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.volumeSelector3.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.volumeSelector4.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.volumeSelector5.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.outputSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.voxelResolutionMatchNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        ui.enableVoxelThresholdCheckBox.connect(
            "toggled(bool)", self.updateParameterNodeFromGUI
        )
        ui.voxelThresholdSlider.connect(
            "valueChanged(double)", self.updateParameterNodeFromGUI
        )
        spinBoxesToConnect = [
            ui.defaultVoxelValueSpinBox,
            ui.isotropicResolutionSpinBox,
            ui.IResolutionSpinBox,
            ui.JResolutionSpinBox,
            ui.KResolutionSpinBox,
        ]
        for spinBox in spinBoxesToConnect:
            spinBox.connect("valueChanged(double)", self.updateParameterNodeFromGUI)

        self.extentModeButtonToModeNameMap = {
            ui.automaticROIRadioButton: "Automatic",
            ui.useSelectedROIRadioButton: "Manual",
        }
        # Connect positively toggling each radio button to changing the stored mode
        for button, extentMode in self.extentModeButtonToModeNameMap.items():
            button.connect(
                "toggled(bool)",
                lambda toggle, modeName=extentMode: self.onExtentModeSelectionChanged(
                    modeName, toggle
                ),
            )
        self.resolutionModeButtonToModeNameMap = {
            ui.voxelResolutionMatchRadioButton: "MatchVolume",
            ui.isotropicResolutionRadioButton: "Isotropic",
            ui.anisotropicResolutionRadioButton: "Anisotropic",
        }
        for button, resolutionMode in self.resolutionModeButtonToModeNameMap.items():
            button.connect(
                "toggled(bool)",
                lambda toggle, modeName=resolutionMode: self.onResolutionModeSelectionChanged(
                    modeName, toggle
                ),
            )

        self.weightingModeButtonToModeNameMap = {
            ui.blendRadioButton: "blend",
            ui.stitchRadioButton: "stitch",
        }
        for button, weightingMode in self.weightingModeButtonToModeNameMap.items():
            button.connect(
                "toggled(bool)",
                lambda toggle, modeName=weightingMode: self.onWeightingModeSelectionChanged(
                    modeName, toggle
                ),
            )
        # Initial GUI update
        self.updateGUIFromParameterNode()

    def onExtentModeSelectionChanged(self, extentMode, toggleValue):
        """User clicked on one of the extent mode radio buttons.
        Update mode if  this is a postive toggle click.
        """
        if toggleValue:
            self._parameterNode.SetParameter("ExtentMode", extentMode)

    def onResolutionModeSelectionChanged(self, resolutionMode, toggleValue):
        """User clicked on one of the resolution mode radio buttons.
        Update mode if  this is a postive toggle click.
        """
        if toggleValue:
            self._parameterNode.SetParameter("ResolutionMode", resolutionMode)

    def onWeightingModeSelectionChanged(self, weightingMode, toggleValue):
        """User clicked on one of the weighting method radio buttons.
        Update mode if  this is a postive toggle click.
        """
        if toggleValue:
            self._parameterNode.SetParameter("WeightingMode", weightingMode)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def setParameterNode(self, inputParameterNode):
        """
        Adds observers to the selected parameter node. Observation is needed because when the
        parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Set parameter node in the parameter node selector widget
        wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
        self.ui.parameterNodeSelector.setCurrentNode(inputParameterNode)
        self.ui.parameterNodeSelector.blockSignals(wasBlocked)

        if inputParameterNode == self._parameterNode:
            # No change
            return

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        if inputParameterNode is not None:
            self.addObserver(
                inputParameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None:
            return

        self.updatingGUIFromParameterNode = True

        # abbreviate to shorten later lines
        pn = self._parameterNode
        ui = self.ui

        # Disable all sections if no parameter node is selected
        # self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None
        # self.ui.advancedCollapsibleButton.enabled = self._parameterNode is not None

        # Ensure radio buttons function correctly (only one selected at a time)
        extentMode = pn.GetParameter("ExtentMode")
        ui.automaticROIRadioButton.setChecked(extentMode == "Automatic")
        ui.useSelectedROIRadioButton.setChecked(extentMode == "Manual")
        resolutionMode = pn.GetParameter("ResolutionMode")
        ui.voxelResolutionMatchRadioButton.setChecked(resolutionMode == "MatchVolume")
        ui.isotropicResolutionRadioButton.setChecked(resolutionMode == "Isotropic")
        ui.anisotropicResolutionRadioButton.setChecked(resolutionMode == "Anisotropic")
        weightingMode = pn.GetParameter("WeightingMode")
        ui.blendRadioButton.setChecked(weightingMode == "blend")
        ui.stitchRadioButton.setChecked(weightingMode == "stitch")

        # We should only set widget values if these following parameters are not empty
        # ('' to float conversion fails otherwise)
        if pn.GetParameter("IsotropicResolution"):
            ui.isotropicResolutionSpinBox.value = float(
                pn.GetParameter("IsotropicResolution")
            )
        if pn.GetParameter("IResolution"):
            ui.IResolutionSpinBox.value = float(pn.GetParameter("IResolution"))
        if pn.GetParameter("JResolution"):
            ui.JResolutionSpinBox.value = float(pn.GetParameter("JResolution"))
        if pn.GetParameter("KResolution"):
            ui.KResolutionSpinBox.value = float(pn.GetParameter("KResolution"))
        if pn.GetParameter("VoxelThresholdValue"):
            ui.voxelThresholdSlider.value = float(
                pn.GetParameter("VoxelThresholdValue")
            )
        if pn.GetParameter("DefaultVoxelValue"):
            ui.defaultVoxelValueSpinBox.value = float(
                pn.GetParameter("DefaultVoxelValue")
            )
        # Update each widget from parameter node reference
        ui.volumeSelector1.setCurrentNode(pn.GetNodeReference("InputVol1"))
        ui.volumeSelector2.setCurrentNode(pn.GetNodeReference("InputVol2"))
        ui.volumeSelector3.setCurrentNode(pn.GetNodeReference("InputVol3"))
        ui.volumeSelector4.setCurrentNode(pn.GetNodeReference("InputVol4"))
        ui.volumeSelector5.setCurrentNode(pn.GetNodeReference("InputVol5"))
        ui.roiSelector.setCurrentNode(pn.GetNodeReference("StitchedVolumeROI"))
        ui.voxelResolutionMatchNodeSelector.setCurrentNode(
            pn.GetNodeReference("ResolutionMatchVol")
        )
        ui.outputSelector.setCurrentNode(pn.GetNodeReference("OutputVolume"))
        ui.enableVoxelThresholdCheckBox.setChecked(
            pn.GetParameter("EnableVoxelThreshold") == "True"
        )
        # Update buttons states and tooltips
        # Enable the Stitch Volumes button if there is a valid set of inputs
        validInputsFlag, validInputsMsg = self.checkValidInputsPresent()
        ui.applyButton.enabled = validInputsFlag
        ui.applyButton.toolTip = validInputsMsg
        # Done updating
        self.updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self.updatingGUIFromParameterNode:
            # Currently updating GUI from parameter node, avoid infinite loop!
            return

        if self._parameterNode is None:
            return

        pn = self._parameterNode
        ui = self.ui
        pn.SetNodeReferenceID("StitchedVolumeROI", ui.roiSelector.currentNodeID)
        pn.SetNodeReferenceID("InputVol1", ui.volumeSelector1.currentNodeID)
        pn.SetNodeReferenceID("InputVol2", ui.volumeSelector2.currentNodeID)
        pn.SetNodeReferenceID("InputVol3", ui.volumeSelector3.currentNodeID)
        pn.SetNodeReferenceID("InputVol4", ui.volumeSelector4.currentNodeID)
        pn.SetNodeReferenceID("InputVol5", ui.volumeSelector5.currentNodeID)
        pn.SetNodeReferenceID(
            "ResolutionMatchVol", ui.voxelResolutionMatchNodeSelector.currentNodeID
        )
        pn.SetNodeReferenceID("OutputVolume", ui.outputSelector.currentNodeID)
        pn.SetParameter(
            "EnableVoxelThreshold",
            "True" if ui.enableVoxelThresholdCheckBox.checked else "False",
        )
        pn.SetParameter("VoxelThresholdValue", f"{ui.voxelThresholdSlider.value}")
        pn.SetParameter("DefaultVoxelValue", f"{ui.defaultVoxelValueSpinBox.value}")

        pn.SetParameter("IsotropicResolution", f"{ui.isotropicResolutionSpinBox.value}")
        pn.SetParameter("IResolution", f"{ui.IResolutionSpinBox.value}")
        pn.SetParameter("JResolution", f"{ui.JResolutionSpinBox.value}")
        pn.SetParameter("KResolution", f"{ui.KResolutionSpinBox.value}")

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.invertedOutputSelector.currentNodeID)

    def checkValidInputsPresent(self) -> tuple[bool, str]:
        """Check whether a valid set of inputs is present.
        A valid set means
          * at least two input volumes
          * a selected ROI or automatic extent mode selected
          * a valid voxel resolution selected (valid match volume, valid isotropic, or valid anisotropic)
        If a valid set is not present, return False and a message explaining what is missing.
        If a valid set is present, return True and a message. In either case,
        the message should be suitable to use as a tooltip for the button.
        """
        # Number of images validation
        nImages = len(self.gather_original_nodes())
        if nImages < 2:
            return (
                False,
                "At least two input images are required to enable stitching!",
            )
        pn = self._parameterNode
        # Extent validation
        extentMode = pn.GetParameter("ExtentMode")
        if extentMode == "Manual":
            if not pn.GetNodeReference("StitchedVolumeROI"):
                return (False, "Choose ROI to enable stitching!")
        elif extentMode == "Automatic":
            pass  # always valid
        else:
            return (
                False,
                "Invalid extent mode has been set, it must be 'Manual' or 'Automatic'!",
            )
        # Resolution Validation
        resolutionMode = pn.GetParameter("ResolutionMode")
        if resolutionMode == "Isotropic":
            isoRes = float(pn.GetParameter("IsotropicResolution"))
            if isoRes <= 0.0:
                return (
                    False,
                    f"Invalid isotropic resolution of {isoRes:0.2f} has been set!",
                )
        elif resolutionMode == "Anisotropic":
            anisoRes = (
                float(pn.GetParameter(pName))
                for pName in ["IResolution", "JResolution", "KResolution"]
            )
            for res in anisoRes:
                if not res or float(res) <= 0:
                    return (False, "Invalid anisotropic resolution supplied!")
        elif resolutionMode == "MatchVolume":
            if not pn.GetNodeReference("ResolutionMatchVol"):
                return (False, "No image provided to match resolution to!")
        else:
            return (False, f"Invalid resolution mode provided: {resolutionMode}!")
        # Blend mode validation
        weightingMode = pn.GetParameter("WeightingMode")
        if not ((weightingMode == "blend") or weightingMode == "stitch"):
            return (
                False,
                f"Invalid blending mode provided: '{weightingMode}', but must be either 'blend' or 'stitch'",
            )
        # If we made it this far, then there is a valid set of inputs
        return (True, "Run stitching computation!")

    def onApplyButton(self):
        """
        Run processing when user clicks "Stitch Volumes" button.
        """
        try:
            # Gather image inputs
            origNodes = self.gather_original_nodes()
            # Gather ROI
            pn = self._parameterNode
            extentMode = pn.GetParameter("ExtentMode")
            if extentMode == "Automatic":
                roiNode = self.logic.createAutomaticROI(origNodes)
                pn.SetNodeReferenceID("StitchedVolumeROI", roiNode.GetID())
            roiNode = pn.GetNodeReference("StitchedVolumeROI")
            # Gather resolution
            resolutionMode = pn.GetParameter("ResolutionMode")
            if resolutionMode == "MatchVolume":
                matchVolNode = pn.GetNodeReference("ResolutionMatchVol")
                voxelSpacingMm = matchVolNode.GetSpacing()
            elif resolutionMode == "Isotropic":
                isoSpacing = float(pn.GetParameter("IsotropicResolution"))
                voxelSpacingMm = (isoSpacing, isoSpacing, isoSpacing)
            elif resolutionMode == "Anisotropic":
                iRes = float(pn.GetParameter("IResolution"))
                jRes = float(pn.GetParameter("JResolution"))
                kRes = float(pn.GetParameter("KResolution"))
                voxelSpacingMm = (iRes, jRes, kRes)
            else:
                raise ValueError(f"Invalid resolution mode provided: {resolutionMode}")
            # Gather blending mode
            weightingMode = pn.GetParameter("WeightingMode")
            # Gather threshold info
            useThresholdValue = pn.GetParameter("EnableVoxelThreshold") == "True"
            thresholdValue = float(pn.GetParameter("VoxelThresholdValue"))
            # Default voxel value
            defaultVoxelValue = float(pn.GetParameter("DefaultVoxelValue"))
            # Output node
            output_node = pn.GetNodeReference("OutputVolume")
            # Run the stitching
            output_node = self.logic.blend_volumes(
                origNodes,
                roiNode,
                voxelSpacingMm,
                weightingMethod=weightingMode,
                thresholdValue=thresholdValue,
                useThresholdValue=useThresholdValue,
                defaultVoxelValue=defaultVoxelValue,
                outputVolNode=output_node,
                keepIntermediateNodes=False,
            )
            pn.SetNodeReferenceID("OutputVolume", output_node.GetID())
            # self.logic.stitch_volumes(
            #    origNodes, roiNode, output_node, keepIntermediateVolumes=False
            # )

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback

            traceback.print_exc()

    def gather_original_nodes(self):
        orig_nodes = []
        if self.ui.volumeSelector1.currentNode():
            orig_nodes.append(self.ui.volumeSelector1.currentNode())
        if self.ui.volumeSelector2.currentNode():
            orig_nodes.append(self.ui.volumeSelector2.currentNode())
        if self.ui.volumeSelector3.currentNode():
            orig_nodes.append(self.ui.volumeSelector3.currentNode())
        if self.ui.volumeSelector4.currentNode():
            orig_nodes.append(self.ui.volumeSelector4.currentNode())
        if self.ui.volumeSelector5.currentNode():
            orig_nodes.append(self.ui.volumeSelector5.currentNode())
        return orig_nodes


#
# StitchVolumesLogic
#


class StitchVolumesLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setDefaultParameters(self, parameterNode, forceDefaultValue=False):
        """
        Initialize parameter node with default settings. Note that if
        the parameter already has a value which is not an empty string,
        this function will not overwrite it unless forceDefaultValue is
        set to True.
        """
        defaultParameterMap = {
            "ExtentMode": "Automatic",
            "ResolutionMode": "MatchVolume",
            "WeightingMode": "blend",
            "EnableVoxelThreshold": "True",
            "VoxelThresholdValue": "-1024.0",
            "IsotropicResolution": "3.0",
            "IResolution": "1.5",
            "JResolution": "1.5",
            "KResolution": "3.0",
            "DefaultVoxelValue": "-2000",
        }
        pn = parameterNode
        for parName, parValue in defaultParameterMap.items():
            currentValue = pn.GetParameter(parName)  # returns '' if not set
            if not currentValue or forceDefaultValue:
                # Set the default value
                pn.SetParameter(parName, parValue)

    def clone_node(self, node_to_clone):
        """Node to clone must be in mrml scene subject hierarchy"""
        # Clone the node (from script repository)
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
            slicer.mrmlScene
        )
        itemIDToClone = shNode.GetItemByDataNode(node_to_clone)
        clonedItemID = (
            slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
                shNode, itemIDToClone
            )
        )
        clonedNode = shNode.GetItemDataNode(clonedItemID)
        return clonedNode

    def clone_and_harden(self, input_node):
        """Create clone of input node and harden any parent transforms on the cloned copy"""
        cloned_node = self.clone_node(input_node)
        cloned_node.SetAndObserveTransformNodeID(input_node.GetTransformNodeID())
        cloned_node.HardenTransform()
        return cloned_node

    def createAutomaticROI(self, originalNodes, outputROI=None):
        """Starting from an ROI which fits the first input node,
        the ROI is expanded without reorientation until it
        minimally contains all the bounding boxes of all input
        nodes.
        """
        if not outputROI:
            outputROI = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode", "StitchingBoundaryROI"
            )
        # Fit ROI to first original image node
        outputROI = self.fitROIToVolume(originalNodes[0], roiNode=outputROI)
        # Find all bounding boxes
        outputROI.SetROIType(outputROI.ROITypeBoundingBox)
        corners = []
        for volNode in originalNodes:
            corners.extend(self.getVolumeNodeCorners(volNode))
        # Add all of these corners as control points
        for corner in corners:
            outputROI.AddControlPointWorld(corner)
        # Consider converting back to ROITypeBox,
        # would need to recalculate center and size
        # or radius?
        # Remove annoying display fill color
        dn = outputROI.GetDisplayNode()
        dn.SetFillVisibility(0)

        return outputROI

    def fitROIToVolume(self, volNode, roiNode=None):
        """Fits markups ROI node to an image volume."""
        if roiNode is None:
            roiNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode", "FittedROI"
            )
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLCropVolumeParametersNode"
        )
        cropVolumeParameters.SetInputVolumeNodeID(volNode.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(
            cropVolumeParameters
        )  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)
        return roiNode

    def blend_volumes(
        self,
        origVolNodes,
        roiNode,
        voxelSpacingMm=(1, 1, 1),
        prioritizeVoxelSize=True,
        thresholdValue=0,
        defaultVoxelValue=-100,
        useThresholdValue=True,
        outputVolNode=None,
        weightingMethod="blend",
        keepIntermediateNodes=False,
    ):
        """Use the newer approach of blending all volumes together"""
        ## Create resample template
        templateVolume = self.createVolumeFromROIandVoxelSize(
            roiNode,
            voxelSizeMm=voxelSpacingMm,
            prioritizeVoxelSize=prioritizeVoxelSize,
            fillVoxelValue=defaultVoxelValue,
        )
        templateArr = slicer.util.arrayFromVolume(templateVolume)
        resampledShape = templateArr.shape
        nImages = len(origVolNodes)
        ## Create Segmentation with templateVolume as the source geometry
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
            templateVolume
        )
        ## Loop over original nodes
        if keepIntermediateNodes:
            resampledNodes = []
        volMasksAll = np.zeros((nImages + 1, *resampledShape))
        volArrsAll = np.zeros_like(volMasksAll)  # to hold voxel values
        for idx, origNode in enumerate(origVolNodes):
            ## Create box model from orig image node
            boxModel = self.createVolNodeBoxModel(origNode)
            ## Create segment in segmentation node from box model
            segmentID = self.createSegmentFromModel(segmentationNode, boxModel)
            ## Resample original node into template volume geometry
            resampledNode = resample(
                origNode,
                templateVolume,
                interpolationMode="Linear",
                defaultValue=defaultVoxelValue,
            )
            volArrsAll[idx, :, :, :] = slicer.util.arrayFromVolume(resampledNode)
            ## Create binary numpy array from segment showing volume location
            volMaskArr = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentationNode, segmentID, templateVolume
            )
            ## Keep the volume mask arrays and the resampled images for future use
            volMasksAll[idx, :, :, :] = volMaskArr

            if not keepIntermediateNodes:
                # Clean up temporary nodes
                slicer.mrmlScene.RemoveNode(boxModel)
                slicer.mrmlScene.RemoveNode(resampledNode)
            else:
                # Put in a list
                resampledNodes.append(resampledNode)

        if not keepIntermediateNodes:
            # No longer need segmentation node
            slicer.mrmlScene.RemoveNode(segmentationNode)

        ## Loop again after binary maps gathered
        weightsAll = np.zeros_like(volMasksAll)  # initialize
        imIdxs = list(range(nImages))
        for imIdx in imIdxs:
            ## Create mask set (volAlone, notVolMask, volIntersectionsMask)
            otherImIdxs = [otherImIdx for otherImIdx in imIdxs if otherImIdx != imIdx]
            volMaskArr = volMasksAll[imIdx, :, :, :]
            volIntersectionsMask = np.zeros_like(volMaskArr)
            for otherImIdx in otherImIdxs:
                # Find intersection
                otherVolMaskArr = volMasksAll[otherImIdx, :, :, :]
                thisIntersection = np.logical_and(volMaskArr, otherVolMaskArr)
                # Add to intersections mask
                volIntersectionsMask = np.logical_or(
                    volIntersectionsMask, thisIntersection
                )
            volAloneMask = np.logical_and(
                volMaskArr, np.logical_not(volIntersectionsMask)
            )
            notVolMask = np.logical_not(volMaskArr)
            ## Get weights for blending
            weightArr = self.getWeightArray(
                volAloneMask, notVolMask, volIntersectionsMask, method=weightingMethod
            )
            ## Store results
            weightsAll[imIdx, :, :, :] = weightArr

        if useThresholdValue:
            ## Handle ignoring regions where voxel value too low (set weight to 0)
            for imIdx in range(nImages):
                resampledArr = volArrsAll[imIdx, :, :, :]
                weightsAll[imIdx, resampledArr <= thresholdValue] = 0
        ## Handle applying default value where sum of weights is zero
        weightSumZeroMask = np.equal(np.sum(weightsAll, axis=0), 0)
        weightsAll[nImages, weightSumZeroMask] = 1
        ## Add default value image as final layer in array
        volArrsAll[nImages, :, :, :] = defaultVoxelValue
        ## Finally, apply weighted average!
        blendedImageArray = np.average(volArrsAll, weights=weightsAll, axis=0)
        ## Assign to output node
        if not outputVolNode:
            outputVolNode = self.clone_node(templateVolume)
            outputVolNode.SetName(slicer.mrmlScene.GenerateUniqueName("StitchedImage"))
        slicer.util.updateVolumeFromArray(outputVolNode, blendedImageArray)
        if not keepIntermediateNodes:
            slicer.mrmlScene.RemoveNode(templateVolume)
        # Adjust display of new node so that the range of voxel values is visible
        dn = outputVolNode.GetDisplayNode()
        dn.SetWindowLevelMinMax(np.min(blendedImageArray), np.max(blendedImageArray))
        return outputVolNode

    def getWeightArray(
        self,
        volAloneMask: np.ndarray,
        notVolMask: np.ndarray,
        volIntersectionsMask: np.ndarray,
        method: str = "blend",
    ):
        """ """
        if method == "blend":
            weightArr = self.getBlendWeightArray(
                volAloneMask, notVolMask, volIntersectionsMask
            )
        elif method == "stitch":
            weightArr = self.getStitchWeightArray(
                volAloneMask, notVolMask, volIntersectionsMask
            )
        else:
            raise ValueError(
                f"Unknown weight array calculation method specified: '{method}'"
            )
        return weightArr

    def getBlendWeightArray(self, volAloneMask, notVolMask, volIntersectionsMask):
        """Calculate blending weights for intersection regions
        # TLDR: Combine into weights map (d0/(d0+d1), 1s, 0s)
        Overview: weights inside the non-overlapped image volume
        region should be 1. Weights outside the image volume (but
        inside the resampled volume) should be zero. Weights
        within intersections of this image with other images should
        be near one adjacent to non-overlapped regions and fade to
        near zero adjacent to outer edges of the overlapped region.
        The formula d0/(d0+d1) achieves this.  d0 is the distance
        from outside the volume mask, d1 is the distance from inside
        the volume mask.  As d0 goes to zero, d0/(d0+d1) goes to zero.
        At the inner edge, d1 goes to zero, and d0 is non-zero, so
        d0/(d0+d1) goes to one. In between, this should change
        reasonably smoothly for any sensible approximate distance
        metric.
        """
        """
        # NOTE: if we are running into memory or processing time
        # problems, we could likely make the distance map calculation
        # faster and smaller by sensibly cropping the calculations into
        # smaller problems.  We only are going to use the distance maps
        # in the overlap regions, but currently we calculate them
        # everywhere.

        Also NOTE, individual weights are guaranteed in this method to
        be 0 to 1 inclusive, but the method does NOT guarantee that
        the sum of weights across multiple different images is one.
        The weighted average function (np.average(, weights=weights))
        which will be used later does not require the weights to sum
        to one. This is not a problem, this approach should still
        do a nice job of smoothly blending regardless; there is
        no need to decide precisely what fraction of each image
        is represented at each voxel.
        """
        ## Create distance maps
        d1 = self.createDistanceMapArray(volAloneMask)
        d0 = self.createDistanceMapArray(notVolMask)

        weightArr = np.zeros(d1.shape)
        weightArr[volAloneMask] = 1
        weightArr[notVolMask] = 0
        d0Masked = d0[volIntersectionsMask]
        d1Masked = d1[volIntersectionsMask]
        weightArr[volIntersectionsMask] = d0Masked / (d0Masked + d1Masked)
        return weightArr

    def getStitchWeightArray(self, volAloneMask, notVolMask, volIntersectionsMask):
        """A weighting scheme which is more like the prior stitching approach,
        where there should potentially be more visible seams if the volumes
        differ significantly. Regions with d0/(d0+d1) >= 0.5, i.e. more than
        half-way close to this volume, are given a weight of 1, and the rest
        of the intersection regions (the further half) are given a weight of
        zero.
        Here, we have no need for any fractional weights, so we can use an
        integer data type rather than floats, saving some memory usage.

        Another method of acheiving a similar result would be to use a grow-
        from-seeds approach.  This would have the advantage of being
        guaranteed to
        """
        d1 = self.createDistanceMapArray(volAloneMask)
        d0 = self.createDistanceMapArray(notVolMask)
        weightArr = np.full_like(d1, 0, dtype=np.uint8)
        weightArr[volAloneMask] = 1
        weightArr[volIntersectionsMask] = (
            ((d0 / (d0 + d1)) >= 0.5)[volIntersectionsMask]
        ).astype(np.uint8)
        return weightArr

    def stitch_volumes(
        self, origNodes, roiNode, outputNode, keepIntermediateVolumes=False
    ):
        """NO LONGER USED, behavior replaced by blend_volumes()"""
        # Stitch together the supplied original volumes, resampling them
        # into the space defined by the supplied roi, putting the stitched
        # output into a volume with the given stitched volume name

        ch_roi_node = self.clone_and_harden(roiNode)
        ch_orig_node = self.clone_and_harden(origNodes[0])

        # Crop/Resample first orig node
        ref_vol_node = resample_volume(ch_roi_node, ch_orig_node, "ReferenceVolume")
        # Clean up cloned ROI node
        slicer.mrmlScene.RemoveNode(ch_roi_node)
        # Resample other nodes
        resamp_vol_nodes = []
        for orig_node in origNodes:
            resampled_name = "Resamp_" + orig_node.GetName()
            resamp_node = createOrReplaceNode(resampled_name)
            ch_orig_node = self.clone_and_harden(orig_node)
            resamp_vol_nodes.append(resample(ch_orig_node, ref_vol_node, resamp_node))
            # Clean up cloned/hardened orig node
            slicer.mrmlScene.RemoveNode(ch_orig_node)
        if not outputNode:
            # Create output volume node to hold stitched image
            output_node_name = slicer.mrmlScene.GenerateUniqueName("Stitched_Volume")
            outputNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", output_node_name
            )
        imArrays = [
            slicer.util.arrayFromVolume(resamp_vol_node)
            for resamp_vol_node in resamp_vol_nodes
        ]
        # Copy all image and orientation data from the reference volume to the output volume
        outputNode.SetOrigin(ref_vol_node.GetOrigin())
        outputNode.SetSpacing(ref_vol_node.GetSpacing())
        imageDirections = vtk.vtkMatrix4x4()
        ref_vol_node.GetIJKToRASDirectionMatrix(imageDirections)
        outputNode.SetIJKToRASDirectionMatrix(imageDirections)
        imageData = vtk.vtkImageData()
        imageData.DeepCopy(ref_vol_node.GetImageData())
        outputNode.SetAndObserveImageData(imageData)

        # Find the dimension to stitch together (I,J,or K)
        dim_to_stitch = find_dim_to_stitch(origNodes, resamp_vol_nodes[0])
        # dim_to_stitch is 0, 1, or 2, depending on whether the dimension to stitch is
        # K,J, or I, respectively (recalling that np arrays are KJI)
        other_dims = tuple({0, 1, 2} - {dim_to_stitch})  # set subtraction
        # We can now sample each resampled volume in along the stitch dimension to
        # figure out where the data starts and
        # stops for each of them.  Then, we can order them by data start value.
        dataSlices = [np.sum(imArray, axis=other_dims) != 0 for imArray in imArrays]
        dataStartIdxs = [np.nonzero(dataSlice)[0][0] for dataSlice in dataSlices]
        dataEndIdxs = [np.nonzero(dataSlice)[0][-1] for dataSlice in dataSlices]
        # Re-order in increasing dataStartIdx order
        ordered = sorted(
            zip(dataStartIdxs, imArrays, dataEndIdxs), key=lambda pair: pair[0]
        )
        orderedDataStartIdxs, orderedImArrays, orderedDataEndIdxs = zip(*ordered)
        imCombined = np.zeros(imArrays[0].shape)
        # We can use the starting and ending indices to determine whether there is overlap
        priorOverlapFlag = False
        for imIdx in range(len(orderedImArrays)):
            imArray = orderedImArrays[imIdx]
            start1 = orderedDataStartIdxs[imIdx]
            end1 = orderedDataEndIdxs[imIdx] + 1  # add 1 because of python indexing
            if imIdx == (len(orderedImArrays) - 1):
                # There is no next volume, just run out to the end of volume
                start2 = end1 + 1
            else:
                # Get the start idx of the next volume
                start2 = orderedDataStartIdxs[imIdx + 1]
            # print('\n---\nstart1:%i\nend1:%i\nstart2:%i\n'%(start1,end1,start2))
            if priorOverlapFlag:
                start1 = nextStartIdx
            # Is there overlap?
            if start2 < end1:
                # There is overlap, the end idx should be shortened
                end1 = np.ceil((end1 + 1 + start2) / 2.0).astype(
                    int
                )  # don't add one, already accounted for
                priorOverlapFlag = True
                nextStartIdx = end1
            else:
                priorOverlapFlag = False
                nextStartIdx = None
            sliceIndexTuple = getSliceIndexTuple(start1, end1, dim_to_stitch)
            imCombined[sliceIndexTuple] = imArray[sliceIndexTuple]
            # print(sliceIndexTuple)
        # New approaches for overlap regions
        # The stitched voxel value is the average of all image volumes
        # which overlap this voxel and which have a voxel value greater
        # than a threshold value at this location.

        priorOverlapFlag = False
        for imIdx, imArray in enumerate(orderedImArrays):
            start1 = orderedDataStartIdxs[imIdx]
            end1

        # Put the result into the stitched volume
        slicer.util.updateVolumeFromArray(outputNode, imCombined)
        # Clean up
        if not keepIntermediateVolumes:
            for resamp_vol_node in resamp_vol_nodes:
                slicer.mrmlScene.RemoveNode(resamp_vol_node)
            slicer.mrmlScene.RemoveNode(ref_vol_node)
        # Return stitched volume node
        return outputNode

    def createDistanceMapArray(self, binaryArray: np.ndarray) -> np.ndarray:
        """Create an approximate distance map based on a binary array.
        Uses SimpleITK's ApproximateSignedDistanceMap filter and
        returns the distance map.
        Note that this filter returns a uniform positive distance if there are
        no ones in the array, and a uniform negative distance if there are
        no zeros in the array. (The uniform value is N*sqrt(3) for an NxNxN
        array)
        """
        # convert to ITK image
        sIm = sitk.GetImageFromArray(binaryArray.astype("uint8"))
        sOutIm = sitk.ApproximateSignedDistanceMap(sIm, 1, 0)
        # output is float32 ITK image
        # convert  back to numpy array
        distanceMapArray = sitk.GetArrayFromImage(sOutIm)
        return distanceMapArray

    def createVolNodeBoxModel(self, volNode, boxModelOutput=None):
        """Create a model node which is a closed rectilinear box
        representing the enclosed volume for a volume node
        """
        if boxModelOutput is None:
            boxModelOutput = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", "VolumeBoundaryModel"
            )
        corners = self.getVolumeNodeCorners(volNode)
        tmpFiducialNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "TempVolBoundaryCorners"
        )
        for c in corners:
            tmpFiducialNode.AddFiducial(*c)
        # Make the model
        smoothing = False  # do not smooth box!
        slicer.modules.markupstomodel.logic().UpdateClosedSurfaceModel(
            tmpFiducialNode, boxModelOutput, smoothing
        )
        self.ensureSurfaceNormalsOutward(boxModelOutput)
        # Clean up temporary node
        slicer.mrmlScene.RemoveNode(tmpFiducialNode)
        return boxModelOutput

    def getVolumeNodeCorners(self, volNode):
        import numpy as np

        ijkToRas = vtk.vtkMatrix4x4()
        volNode.GetIJKToRASMatrix(ijkToRas)
        volumeRasToWorld = vtk.vtkMatrix4x4()
        slicer.vtkMRMLTransformNode.GetMatrixTransformBetweenNodes(
            volNode.GetParentTransformNode(), None, volumeRasToWorld
        )
        ijkToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(volumeRasToWorld, ijkToRas, ijkToWorld)
        dims = volNode.GetImageData().GetDimensions()
        corners = []
        # note the -.5 an +.5 to get the outside voxel corners instead of the voxel centers
        for i in (0 - 0.5, dims[0] - 1 + 0.5):
            for j in (0 - 0.5, dims[1] - 1 + 0.5):
                for k in (0 - 0.5, dims[2] - 1 + 0.5):
                    cornerWorldH = np.zeros(4)
                    ijkToWorld.MultiplyPoint([i, j, k, 1], cornerWorldH)
                    corners.append(cornerWorldH[0:3])
        # Corners order is
        # 0: 000,
        # 1: 001,
        # 2: 010,
        # 3: 011,
        # 4: 100,
        # 5: 101,
        # 6: 110,
        # 7: 111
        return corners

    def ensureSurfaceNormalsOutward(self, modelNode):
        """Ensure surface normals are all directed outwards. Sometimes after plane cut, some
        normals are wrong, and this should fix them.
        """
        logic = SurfaceToolbox.SurfaceToolboxLogic()
        logic.computeNormals(
            modelNode, modelNode, autoOrient=True, split=True, splitAngle=30.0
        )

    def createSegmentFromModel(
        self, segmentationNode, modelNode, renamedSegmentName=None
    ):
        """Create segment in existing segmentation node (which must have segmentation geometry
        already defined) corresponding to inside of input modelNode.  Segment can be optionally
        be renamed; the default name is the same as the modelNode.GetName()
        """
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
            modelNode, segmentationNode
        )
        # new segment is added to the end of the list of segments, so it is now last
        segIdx = segmentationNode.GetSegmentation().GetNumberOfSegments() - 1
        newSegment = segmentationNode.GetSegmentation().GetNthSegment(segIdx)
        # Rename if requested
        if renamedSegmentName is not None:
            newSegment.SetName(renamedSegmentName)
        newSegmentID = segmentationNode.GetSegmentation().GetNthSegmentID(segIdx)
        return newSegmentID

    def createVolumeFromROIandVoxelSize(
        self,
        ROINode,
        voxelSizeMm=(1.0, 1.0, 1.0),
        prioritizeVoxelSize=True,
        voxelType=vtk.VTK_INT,
        fillVoxelValue=0,
        voxelTol=0.1,  # fraction of a voxel it is OK to shrink the ROI by (rather than growing by 1-voxelTol voxels)
    ):
        """Create an empty scalar volume node with the given resolution, location, and
        orientation. The resolution must be given directly (single or scalar value interpreted
        as an isotropic edge length), and the location, size, and orientation are derived from
        the ROINode (a vtkMRMLAnnotationROINode). If prioritizeVoxelSize is True (the default),
        and the size of the ROI is not already an integer number of voxels across in each dimension,
        the ROI boundaries are adjusted to be an integer number of voxels across. If
        prioritizeVoxelSize is False, then the ROI is left unchanged, and the voxel dimensions
        are minimally adjusted such that the existing ROI is an integer number of voxels across.
        If ROI dimensions are adjusted, the ROI center is not moved; adjustments are symmetric around
        the existing ROI center. When adjusting, it needs to be determined whether the ROI dimension
        is increased or decreased.  If the remainder fraction of a voxel is >= than voxelTol, then
        the ROI dimension is increased to the next integer multiple of the voxel size. If the
        remainder fraction of a voxel is less than voxelTol, then the ROI dimension is decreased
        to the next integer multiple of the voxel size. The default voxelTol value of 0.1 provides
        behavior biased towards expanding the ROI except for very small inconsistencies.  voxelTol
        is ignored if prioritizeVoxelSize is False.
        """
        # Ensure resolutionMm can be converted to a list of 3 voxel edge lengths
        # If voxel size is a scalar or a one-element list, interpret that as a request for
        # isotropic voxels with that edge length
        if hasattr(voxelSizeMm, "__len__"):
            if len(voxelSizeMm) == 1:
                voxelSizeMm = [voxelSizeMm[0]] * 3
            elif not len(voxelSizeMm) == 3:
                raise Exception(
                    "voxelSizeMm must either have 1 or 3 elements; it does not."
                )
        else:
            try:
                v = float(voxelSizeMm)
                voxelSizeMm = [v] * 3
            except:
                raise Exception(
                    "voxelSizeMm does not appear to be a number or a list of one or three numbers."
                )

        # Resolve any tension between the ROI size and resolution if ROI is
        # not an integer number of voxels in all dimensions
        ROIRadiusXYZMm = [0] * 3  # initialize
        ROINode.GetRadiusXYZ(ROIRadiusXYZMm)  # fill in ROI sizes
        ROIDiamXYZMm = 2 * np.array(
            ROIRadiusXYZMm
        )  # need to double radii to get box dims
        numVoxelsAcrossFloat = np.divide(ROIDiamXYZMm, voxelSizeMm)
        if prioritizeVoxelSize:
            # Adjust ROI size by increasing it to the next integer multiple of the voxel edge length
            numVoxAcrossInt = []
            for voxAcross in numVoxelsAcrossFloat:
                # If over by less voxelTol of a voxel, don't ceiling it
                diff = voxAcross - np.round(voxAcross)
                if diff > 0 and diff < voxelTol:
                    voxAcrossInt = np.round(
                        voxAcross
                    )  # round it down, which will shrink the ROI by up to voxelTol voxels
                else:
                    voxAcrossInt = np.ceil(
                        voxAcross
                    )  # otherwise, grow ROI to the next integer voxel size
                numVoxAcrossInt.append(voxAcrossInt)
            # Figure out new ROI dimensions
            adjustedROIDiamXYZMm = np.multiply(numVoxAcrossInt, voxelSizeMm)
            adjustedROIRadiusXYZMm = (
                0.5 * adjustedROIDiamXYZMm
            )  # radii are half box dims
            # Apply adjustment
            ROINode.SetRadiusXYZ(adjustedROIRadiusXYZMm)
        else:  # prioritize ROI dimension, adjust voxel resolution
            numVoxAcrossInt = np.round(numVoxelsAcrossFloat)
            # Adjust voxel resolution
            adjustedVoxelSizeMm = np.divide(ROIDiamXYZMm, numVoxAcrossInt)
            voxelSizeMm = adjustedVoxelSizeMm

        #
        volumeName = "OutputTemplateVolume"
        imageDirections, roiOrigin = self.getROIDirectionsAndOrigin(ROINode)
        # Normalize directions to unit length
        for imDir in imageDirections:
            imDir[:] = imDir / np.linalg.norm(imDir)
        # The image origin is the center of the corner voxel, not the outside corner.
        # The ROI origin is the outside corner. To convert between the two, we need to
        # add half a voxel in each of the voxel dimensions
        iOffset = voxelSizeMm[0] / 2 * imageDirections[0]
        jOffset = voxelSizeMm[1] / 2 * imageDirections[1]
        kOffset = voxelSizeMm[2] / 2 * imageDirections[2]
        imageOrigin = np.array(roiOrigin) + iOffset + jOffset + kOffset

        # Create volume node
        templateVolNode = self.createVolumeNodeFromScratch(
            volumeName,
            imageSizeVox=numVoxAcrossInt,
            imageOrigin=imageOrigin,
            imageSpacingMm=voxelSizeMm,
            imageDirections=imageDirections,
            voxelType=voxelType,  # e.g. vtk.VTK_INT
            fillVoxelValue=fillVoxelValue,
        )
        return templateVolNode

    def getROIDirectionsAndOrigin(self, roiNode):
        """ """
        # Processing is different depending on whether the roiNode is AnnotationsMarkup or MarkupsROINode
        if isinstance(roiNode, slicer.vtkMRMLMarkupsROINode):
            axis0 = [0, 0, 0]
            roiNode.GetXAxisWorld(
                axis0
            )  # This respects soft transforms applied to the ROI!
            axis1 = [0, 0, 0]
            roiNode.GetYAxisWorld(axis1)
            axis2 = [0, 0, 0]
            roiNode.GetZAxisWorld(axis2)
            # These axes are the columns of the IJKToRAS directions matrix, but when
            # we supply a list of directions to the imageDirections, it takes a list of rows,
            # so we need to transpose
            directions = np.transpose(
                np.stack((axis0, axis1, axis2))
            )  # for imageDirections
            center = [0, 0, 0]
            roiNode.GetCenterWorld(center)
            radiusXYZ = [0, 0, 0]
            roiNode.GetRadiusXYZ(radiusXYZ)
            # The origin in the corner where the axes all point along the ROI
            origin = (
                np.array(center)
                - np.array(axis0) * radiusXYZ[0]
                - np.array(axis1) * radiusXYZ[1]
                - np.array(axis2) * radiusXYZ[2]
            )
        else:
            # Input is not markupsROINode, must be older annotations ROI instead
            T_id = roiNode.GetTransformNodeID()
            if T_id:
                T = slicer.mrmlScene.GetNodeByID(T_id)
            else:
                T = None
            if T:
                # Transform node is present
                # transformMatrix = slicer.util.arrayFromTransformMatrix(T) # numpy 4x4 array
                # if nested transform, then above will fail! # TODO TODO
                worldToROITransformMatrix = vtk.vtkMatrix4x4()
                T.GetMatrixTransformBetweenNodes(None, T, worldToROITransformMatrix)
                # then convert to numpy
            else:
                worldToROITransformMatrix = (
                    vtk.vtkMatrix4x4()
                )  # defaults to identity matrix
                # transformMatrix = np.eye(4)
            # Convert to directions (for image directions)
            axis0 = np.array(
                [worldToROITransformMatrix.GetElement(i, 0) for i in range(3)]
            )
            axis1 = np.array(
                [worldToROITransformMatrix.GetElement(i, 1) for i in range(3)]
            )
            axis2 = np.array(
                [worldToROITransformMatrix.GetElement(i, 2) for i in range(3)]
            )
            directions = (axis0, axis1, axis2)  # for imageDirections
            # Find origin of roiNode (RAS world coord)
            # Origin is Center - radius1 * direction1 - radius2 * direction2 - radius3 * direction3
            ROIToWorldTransformMatrix = vtk.vtkMatrix4x4()
            ROIToWorldTransformMatrix.DeepCopy(worldToROITransformMatrix)  # copy
            ROIToWorldTransformMatrix.Invert()  # invert worldToROI to get ROIToWorld
            # To adjust the origin location I need to use the axes of the ROIToWorldTransformMatrix
            ax0 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 0) for i in range(3)]
            )
            ax1 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 1) for i in range(3)]
            )
            ax2 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 2) for i in range(3)]
            )
            boxDirections = (ax0, ax1, ax2)
            TransformOrigin4 = ROIToWorldTransformMatrix.MultiplyPoint([0, 0, 0, 1])
            TransformOrigin = TransformOrigin4[:3]
            roiCenter = [0] * 3  # intialize
            roiNode.GetXYZ(roiCenter)  # fill
            # I want to transform the roiCenter using roiToWorld
            transfRoiCenter4 = ROIToWorldTransformMatrix.MultiplyPoint([*roiCenter, 1])
            transfRoiCenter = transfRoiCenter4[:3]
            # Now need to subtract
            radXYZ = [0] * 3
            roiNode.GetRadiusXYZ(radXYZ)
            origin = (
                np.array(transfRoiCenter)
                - ax0 * radXYZ[0]
                - ax1 * radXYZ[1]
                - ax2 * radXYZ[2]
            )

        # Return outputs
        return directions, origin

    def createVolumeNodeFromScratch(
        self,
        nodeName="VolumeFromScratch",
        imageSizeVox=(256, 256, 256),  # image size in voxels
        imageSpacingMm=(2.0, 2.0, 2.0),  # voxel size in mm
        imageOrigin=(0.0, 0.0, 0.0),
        imageDirections=(
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ),  # Image axis directions IJK to RAS,  (these should be orthogonal!)
        fillVoxelValue=0,
        voxelType=vtk.VTK_INT,
    ):
        """Create a scalar volume node from scratch, given information on
        name, size, spacing, origin, image directions, fill value, and voxel
        type.
        """
        imageData = vtk.vtkImageData()
        imageSizeVoxInt = [int(v) for v in imageSizeVox]
        imageData.SetDimensions(imageSizeVoxInt)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Normalize and check orthogonality image directions
        import numpy as np
        import logging

        imageDirectionsUnit = [np.divide(d, np.linalg.norm(d)) for d in imageDirections]
        angleTolDegrees = 1  # allow non-orthogonality up to 1 degree
        for pair in ([0, 1], [1, 2], [2, 0]):
            angleBetween = np.degrees(
                np.arccos(
                    np.dot(imageDirectionsUnit[pair[0]], imageDirectionsUnit[pair[1]])
                )
            )
            if abs(90 - angleBetween) > angleTolDegrees:
                logging.warning(
                    "Warning! imageDirections #%i and #%i supplied to createVolumeNodeFromScratch are not orthogonal!"
                    % (pair[0], pair[1])
                )
                # Continue anyway, because volume nodes can sort of handle non-orthogonal image directions (though they're not generally expected)
        # Create volume node
        volumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", nodeName
        )
        volumeNode.SetOrigin(imageOrigin)
        volumeNode.SetSpacing(imageSpacingMm)
        volumeNode.SetIJKToRASDirections(imageDirections)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        return volumeNode

    def setup_segment_editor(self, segmentationNode=None, masterVolumeNode=None):
        """Runs standard setup of segment editor widget and segment editor node"""
        if segmentationNode is None:
            # Create segmentation node
            segmentationNode = slicer.vtkMRMLSegmentationNode()
            slicer.mrmlScene.AddNode(segmentationNode)
            segmentationNode.CreateDefaultDisplayNodes()
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        if masterVolumeNode:
            segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)
        return segmentEditorWidget, segmentEditorNode, segmentationNode


#
# StitchVolumesTest
#


class StitchVolumesTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.delayDisplay("Starting the test suite...")
        self.setUp()
        # out = self.test_SimpleOverlap()
        # out = self.test_NoOverlapWithGap()
        # out = self.test_ThreeVolumesDiagonal()
        # out = self.test_ROIRounding()
        # out = self.test_Threshold()
        # out = self.test_StitchWeighting()
        out = self.test_Isotropic()

        # self.test_StitchVolumes1()
        # self.test_StitchVolumes2()
        self.delayDisplay("Completed all tests!")
        return out

    def test_SimpleOverlap(self, keepIntermediateNodes=False):
        """Basic test of two overlapping volumes. They should
        blend in the overlap region, and retain values elsewhere.
        There should be no voxels filled with the default value
        in the result because the input images should fill the
        whole region.
        """
        self.delayDisplay("Starting SimpleOverlap test")
        logic = StitchVolumesLogic()
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(50, 50, 50),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=100,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(50, 50, 50),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 20),
            fillVoxelValue=0,
        )
        roiAuto = logic.createAutomaticROI([volA, volB])
        volOut = logic.blend_volumes(
            [volA, volB],
            roiNode=roiAuto,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=-100,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        ### Verify results ###
        outArr = slicer.util.arrayFromVolume(volOut)
        # The output image should have shape (60,50,50) (KJI)
        self.assertTrue(
            outArr.shape == (60, 50, 50), "Output array has incorrect shape"
        )
        # The volA only region should have voxel value uniformly 100
        self.assertTrue(
            np.all(outArr[0:10, :, :] == 100),
            "Output array has incorrect value where only volA is present",
        )
        # The volB only region should have voxel value uniformly 0
        self.assertTrue(
            np.all(outArr[50:60, :, :] == 0),
            "Output array has incorrect value where only volB is present",
        )
        # The intersection region should have voxel values strictly between 0 and 1
        self.assertTrue(
            np.all(outArr[10:50, :, :] > 0) and np.all(outArr[10:50, :, :] < 100),
            "Output array has incorrect values in the intersection region between volA and volB",
        )
        self.assertFalse(
            np.any(outArr == -100),
            "Output array contains voxels set to the default voxel value, but it shouldn't",
        )
        self.delayDisplay("SimpleOverlap test passed!")
        return volOut

    def test_NoOverlapWithGap(self, keepIntermediateNodes=False):
        """Test a case where the two images to blend do not overlap and there
        is a gap between them.  The output should be an image where the gap
        is filled with the defaultVoxelValue, and the input images are replicated
        where they belong (though cast to float instead of their original type).
        """
        self.delayDisplay("Starting NoOverlapWithGap test")
        logic = StitchVolumesLogic()
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=100,
        )
        # volB matches A, except shifted 30 mm (15 voxels) in
        # a superior direction, and filled with 0s intstead of
        # 100s.
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 30),
            fillVoxelValue=0,
        )
        roiAuto = logic.createAutomaticROI([volA, volB])
        volOut = logic.blend_volumes(
            [volA, volB],
            roiNode=roiAuto,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=-100,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        ### Verify results ###
        outArr = slicer.util.arrayFromVolume(volOut)
        # The output image should have shape (25,10,10) (KJI)
        self.assertTrue(
            outArr.shape == (25, 10, 10), "Output array has incorrect shape"
        )
        # The volA only region should have voxel value uniformly 100
        self.assertTrue(
            np.all(outArr[0:10, :, :] == 100),
            "Output array has incorrect value where only volA is present",
        )
        # The volB only region should have voxel value uniformly 0
        self.assertTrue(
            np.all(outArr[15:25, :, :] == 0),
            "Output array has incorrect value where only volB is present",
        )
        # The gap region should have voxel default value
        self.assertTrue(
            np.all(outArr[10:15, :, :] == -100),
            "Output array has incorrect values in the gap region between volA and volB",
        )
        self.assertTrue(
            np.all((outArr == -100) | (outArr == 100) | (outArr == 0)),
            "Output array contains voxels set to values which are not in volA, volB, or the default voxel value, but it shouldn't",
        )
        self.delayDisplay("NoOverlapWithGap test passed!")
        return volOut

    def test_Isotropic(self, keepIntermediateNodes=True):
        """Test resampling to a different isotropic voxel
        size.
        """
        self.delayDisplay("Starting Isotropic test")
        logic = StitchVolumesLogic()
        # VolA is 2mm isotropic, 20 mm a side cubic
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(1, 1, 1),  # puts corner at (0,0,0)
            fillVoxelValue=100,
        )
        # VolB is 1 mm isotropic, 20 mm a side cubic,
        # and offset by 10 mm in two dimensions
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(20, 20, 20),
            imageSpacingMm=(1, 1, 1),
            imageOrigin=(0.5, 10.5, 10.5),  # puts corner at 0,10,10
            fillVoxelValue=0,
        )
        roiAuto = logic.createAutomaticROI([volA, volB])
        outputVoxelSpacing = (3.0, 3.0, 3.0)
        volOut = logic.blend_volumes(
            [volA, volB],
            roiNode=roiAuto,
            voxelSpacingMm=outputVoxelSpacing,
            useThresholdValue=False,
            defaultVoxelValue=-100,
            weightingMethod="stitch",
            keepIntermediateNodes=keepIntermediateNodes,
        )
        ### Verify results ###
        outArr = slicer.util.arrayFromVolume(volOut)
        # The output image should have shape (10,10,7) (KJI)
        self.assertTrue(outArr.shape == (10, 10, 7), "Output array has incorrect shape")
        # TODO: add more tests
        self.delayDisplay("Isotropic basic test passed")
        return volOut

    def test_StitchWeighting(self, keepIntermediateNodes=False):
        """Test a case where we use the 'stitch' weighting method rather than
        the default 'blend' weighting method. The two images to stitch overlap
        diagonally. The output should be an image where the values are all
        one image's value or the others (except for outside both, where the
        value should be the defaultVoxelValue)
        """
        self.delayDisplay("Starting StitchWighting test")
        logic = StitchVolumesLogic()
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=100,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 10, 10),
            fillVoxelValue=0,
        )
        roiAuto = logic.createAutomaticROI([volA, volB])
        volOut = logic.blend_volumes(
            [volA, volB],
            roiNode=roiAuto,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=-100,
            weightingMethod="stitch",
            keepIntermediateNodes=keepIntermediateNodes,
        )
        ### Verify results ###
        outArr = slicer.util.arrayFromVolume(volOut)
        # The output image should have shape (15,15,10) (KJI)
        self.assertTrue(
            outArr.shape == (15, 15, 10), "Output array has incorrect shape"
        )
        # The volA only region should have voxel value uniformly 100
        volAMask = np.zeros_like(outArr).astype(bool)
        volAMask[0:10, 0:10, :] = True
        volBMask = np.full_like(volAMask, False)
        volBMask[5:, 5:, :] = True
        intersectionMask = (volAMask) & (volBMask)
        volAOnlyMask = (volAMask) & (~intersectionMask)
        volBOnlyMask = (volBMask) & (~intersectionMask)
        self.assertTrue(
            np.all(outArr[volAOnlyMask] == 100),
            "Output array has incorrect value where only volA is present",
        )
        # The volB only region should have voxel value uniformly 0
        self.assertTrue(
            np.all(outArr[volBOnlyMask] == 0),
            "Output array has incorrect value where only volB is present",
        )
        # The intersection region should have only volAValue or volBValue
        self.assertTrue(
            np.all((outArr[intersectionMask] == 100) | (outArr[intersectionMask] == 0)),
            "Intersection region contains unexpected values!",
        )
        # All voxels should be 100, 0, or -100
        self.assertTrue(
            np.all((outArr == -100) | (outArr == 100) | (outArr == 0)),
            "Output array contains voxels set to values which are not in volA, volB, or the default voxel value, but it shouldn't",
        )
        self.delayDisplay("StitchWighting test passed!")
        return volOut

    def test_ThreeVolumesDiagonal(self, keepIntermediateNodes=False):
        """Three volumes to blend, staggered in two dimensions, only two overlapping
        at a time.  The center volume has a value of 100, the outer volumes have
        a value of zero, and the default voxel value is -100.  The overlap regions
        should blend smoothly from 100 to 0, and the outside regions should be -100.
        """
        self.delayDisplay("Starting ThreeVolumesDiagonal test")
        logic = StitchVolumesLogic()
        volAValue = 100
        volBValue = 0
        volCValue = 0
        defaultValue = -100
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volAValue,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 10, 10),
            fillVoxelValue=volBValue,
        )
        volC = logic.createVolumeNodeFromScratch(
            "C",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, -10, -10),
            fillVoxelValue=volCValue,
        )
        inputVols = [volA, volB, volC]
        roiAuto = logic.createAutomaticROI(inputVols)
        volOut = logic.blend_volumes(
            inputVols,
            roiNode=roiAuto,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=defaultValue,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        ### Verify results ###
        outArr = slicer.util.arrayFromVolume(volOut)
        # The output image should have shape (20,20,10) (KJI)
        self.assertTrue(
            outArr.shape == (20, 20, 10), "Output array has incorrect shape"
        )
        volAMask = np.full_like(outArr, False, dtype=bool)
        volAMask[5:15, 5:15, :] = True
        volBMask = np.full_like(volAMask, False, dtype=bool)
        volBMask[0:10, 0:10, :] = True
        volCMask = np.full_like(volAMask, False, dtype=bool)
        volCMask[10:20, 10:20, :] = True
        volAOnlyMask = (volAMask) & (~volBMask) & (~volCMask)
        volBOnlyMask = (volBMask) & (~volAMask) & (~volCMask)
        volCOnlyMask = (volCMask) & (~volAMask) & (~volBMask)
        intersectionACMask = (volAMask) & (volCMask)
        intersectionABMask = (volAMask) & (volBMask)
        outsideMask = (~volAMask) & (~volBMask) & (~volCMask)
        # The volA only region should have voxel value uniformly 100
        self.assertTrue(
            np.all(outArr[volAOnlyMask] == volAValue),
            "Output array has incorrect value where only volA is present",
        )
        # The volB only region should have voxel value uniformly 0
        self.assertTrue(
            np.all(outArr[volBOnlyMask] == volBValue),
            "Output array has incorrect value where only volB is present",
        )
        self.assertTrue(
            np.all(outArr[volCOnlyMask] == volCValue),
            "Output array has incorrect value where only volC is present",
        )
        # Intersection regions should have intermediate values
        lowerValueAB = np.min((volAValue, volBValue))
        upperValueAB = np.max((volAValue, volBValue))
        self.assertTrue(
            upperValueAB != lowerValueAB,
            "For this test, the volA and volB voxel values should differ, but they don't.",
        )
        self.assertTrue(
            np.all(
                (outArr[intersectionABMask] > lowerValueAB)
                & (outArr[intersectionABMask] < upperValueAB)
            ),
            "Output array has incorrect values in the intersection region between volA and volB",
        )
        lowerValueAC = np.min((volAValue, volCValue))
        upperValueAC = np.max((volAValue, volCValue))
        self.assertTrue(
            upperValueAC != lowerValueAC,
            "For this test, the volA and volC voxel values should differ, but they don't.",
        )
        self.assertTrue(
            np.all(
                (outArr[intersectionACMask] > lowerValueAC)
                & (outArr[intersectionACMask] < upperValueAC)
            ),
            "Output array has incorrect values in the intersection region between volA and volC",
        )
        # Outside regions should have default value
        self.assertTrue(
            np.all(outArr[outsideMask] == defaultValue),
            "Output array has incorrect values outside all input volumes.",
        )
        self.delayDisplay("ThreeVolumesDiagonal test passed!")
        return volOut

    def test_PartialVoxels(self, keepIntermediateNodes=False):
        """Test case where overlapping volumes do not fall exactly on the same extended
        voxel grid, so one needs to be resampled.
        """
        self.delayDisplay("Starting PartialVoxels test")
        logic = StitchVolumesLogic()
        volAValue = 100
        volBValue = 0
        volCValue = 0
        defaultValue = -100
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volAValue,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 10, 10),
            fillVoxelValue=volBValue,
        )
        inputVols = [volA, volB, volC]
        roiAuto = logic.createAutomaticROI(inputVols)
        volOut = logic.blend_volumes(
            inputVols,
            roiNode=roiAuto,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=defaultValue,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        self.delayDisplay("PartialVoxels test NOT YET WRITTEN!!")
        raise Exception("PartialVoxels test NOT YET WRITTEN!!")
        return volOut

    def test_ROIRounding(self, keepIntermediateNodes=False):
        """Test case ensuring correct behavior when the input ROI boundary is not
        an exact number of voxel across and therefore is adjusted.
        In this test, the autoROI is expanded by 1.5 voxels, and we verify that
        the ROI and the output are correctly expanded by 2 voxels.
        """
        self.delayDisplay("Starting ROIRounding test")
        logic = StitchVolumesLogic()
        volAValue = 100
        volBValue = 0
        defaultValue = -100
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volAValue,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 10),
            fillVoxelValue=volBValue,
        )
        inputVols = [volA, volB]
        # Start with automatically created ROI, then adjust
        roi = logic.createAutomaticROI(inputVols)
        origRadiusXYZ = np.zeros(3)  # pre-allocate
        roi.GetRadiusXYZ(origRadiusXYZ)
        # Increase by 3mm in one dimension (1.5x 2mm voxels)
        newRadiusXYZ = np.zeros(3)
        newRadiusXYZ[:] = origRadiusXYZ[:]
        newRadiusXYZ[0] = origRadiusXYZ[0] + 1.5
        roi.SetRadiusXYZ(newRadiusXYZ)
        volOut = logic.blend_volumes(
            inputVols,
            roiNode=roi,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            defaultVoxelValue=defaultValue,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        # Get the roi size after adjustment
        finalRadiusXYZ = np.zeros(3)
        roi.GetRadiusXYZ(finalRadiusXYZ)
        ### Verify Results ###
        # Original fit ROI should be 20x20x30 mm, so radii should be (10,10,15)
        self.assertTrue(
            np.all(origRadiusXYZ == np.array((10, 10, 15))),
            f"Original ROI has incorrect size! {origRadiusXYZ}, but should be (10,10,15)!",
        )
        # Expanded ROI should have radii 11.5, 10, 15
        self.assertTrue(
            np.all(newRadiusXYZ == (11.5, 10, 15)),
            "New ROI (before blending) has incorrect size!",
        )
        # Adjustment prioritizing voxel size should expand the radius with dimension 11.5
        # to be an integer number of voxels.  It is currently 23 mm across, which is not
        # a whole number of 2mm voxels across.  The remainder is 1mm, which is greater than
        # 0.1*2mm (i.e. voxelTol * voxelSizeMm), so it should be expanded rather than
        # shrunken, so the right dimension is 24mm across, which corresponds to a radius
        # of 12 mm, so the correct final radii are (12, 10, 15)
        self.assertTrue(
            np.all(finalRadiusXYZ == (12, 10, 15)), "ROI adjusted size is incorrect!"
        )
        # Output image voxel size should be (2,2,2), not changed from the input voxel size
        outputSpacing = np.array(volOut.GetSpacing())
        self.assertTrue(
            np.all(np.array(outputSpacing == np.array((2.0, 2.0, 2.0)))),
            f"Output stitched volume has incorrect spacing! {outputSpacing}, but should be (2,2,2)!",
        )
        # Output image dimensions in voxels should be (12,10,15)
        outputDimensions = np.array(volOut.GetImageData().GetDimensions())
        self.assertTrue(
            np.all(outputDimensions == np.array((12, 10, 15))),
            f"Output stitched image has incorrect dimensions in voxels! {outputDimensions}, but should be (12,10,15)!",
        )

        ## All assertions passed
        self.delayDisplay("ROIRounding test passed!")
        return volOut

    def test_SoftTransform(self):
        """Make sure that soft transforms on input volumes are respected."""
        logic = StitchVolumesLogic()
        volAValue = 100
        volBValue = 0
        defaultValue = -100
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volAValue,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volBValue,
        )
        raise Exception("Test not written yet")

    def test_Threshold(self, keepIntermediateNodes=False):
        """Make sure that threshold value is respected and that the useThresholdValue
        flag is respected.

        """
        # Construct more commplex test volumes, where an area of one of them is below the
        # threshold value.  volA is 100 everywhere. volB is half 25, half -25, split
        # L and R, and is displaced 6 mm right and 6 mm superior relative to volA.
        logic = StitchVolumesLogic()
        volAValue = 100
        volBValueHigh = 20
        volBValueLow = -20
        defaultValue = -100
        thresholdValue = 0
        volA = logic.createVolumeNodeFromScratch(
            "A",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(0, 0, 0),
            fillVoxelValue=volAValue,
        )
        volB = logic.createVolumeNodeFromScratch(
            "B",
            imageSizeVox=(10, 10, 10),
            imageSpacingMm=(2, 2, 2),
            imageOrigin=(6, 0, 6),
            fillVoxelValue=volBValueLow,
        )
        volBarr = slicer.util.arrayFromVolume(volB)
        volBarr[:, :, 5:] = volBValueHigh  # KJI indexing, Right half is High
        slicer.util.arrayFromVolumeModified(volB)
        inputVols = [volA, volB]
        # Run with threshold applied
        roiNode = logic.createAutomaticROI(inputVols)
        volOutThresh = logic.blend_volumes(
            inputVols,
            roiNode=roiNode,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=True,
            thresholdValue=thresholdValue,
            defaultVoxelValue=defaultValue,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        volOutThresh.SetName("StitchedWithThreshold")
        volOutNoThresh = logic.blend_volumes(
            inputVols,
            roiNode=roiNode,
            voxelSpacingMm=(2, 2, 2),
            useThresholdValue=False,
            thresholdValue=thresholdValue,
            defaultVoxelValue=defaultValue,
            keepIntermediateNodes=keepIntermediateNodes,
        )
        volOutNoThresh.SetName("StitchedNoThreshold")
        ### Results Verification ###
        noThreshArr = slicer.util.arrayFromVolume(volOutNoThresh)
        threshArr = slicer.util.arrayFromVolume(volOutThresh)
        volAMask = np.zeros_like(threshArr, dtype=bool)
        volAMask[:10, :10, :10] = True
        volBMask = np.zeros_like(threshArr, dtype=bool)
        volBMask[3:, :10, 3:] = True
        belowThreshMask = np.zeros_like(threshArr, dtype=bool)
        belowThreshMask[3:, :10, 3:8] = True
        # All voxel values should be different in the threshold region
        self.assertTrue(
            np.all(noThreshArr[belowThreshMask] != threshArr[belowThreshMask]),
            "All voxel values in the below-threshold region should be different, but they aren't.",
        )
        # All voxel values should be the same in the non-threshold region
        self.assertTrue(
            np.all(noThreshArr[~belowThreshMask] == threshArr[~belowThreshMask]),
            "All voxel values outside the below-threshold region should match, but they don't.",
        )
        # Voxel values in the B-only region and the threshold region should
        # be volBValueLow if no threshold and defaultVoxel value if threshold
        intersectionMask = (volAMask) & (volBMask)
        belowThreshIntersectionMask = (intersectionMask) & (belowThreshMask)
        volBAloneAndBelowThresh = (volBMask) & (~intersectionMask) & (belowThreshMask)
        self.assertTrue(
            np.all(noThreshArr[volBAloneAndBelowThresh] == volBValueLow),
            "If no threshold is applied, this region should have value volBValueLow, but it doesn't",
        )
        self.assertTrue(
            np.all(threshArr[volBAloneAndBelowThresh] == defaultValue),
            "If threshold is applied, then this region should have value defaultValue, but it doesn't",
        )
        # Voxel values in the threshold intersection region should be volAValue
        # with threshold applied and <volAValue without threshold applied
        self.assertTrue(
            np.all(noThreshArr[belowThreshIntersectionMask] < volAValue),
            "If no threshold applied, the below-threshold intersection region should blend the volAValue with volBValueLow, but that doesn't seem to be the case.",
        )
        self.assertTrue(
            np.all(threshArr[belowThreshIntersectionMask] == volAValue),
            "If threshold is applied, the below-threshold intersection region should have the unblended volAValue everywhere, but it doesn't",
        )

        self.delayDisplay("Threshold test passed!")
        # raise Exception("Test not written yet")
        return

    def test_CoincidentVolumes(self):
        """Ensure that the processing works even if two input volumes are identical
        (no region in the ROI is outside both volumes and no region is inside only
        one of the volumes)
        """
        raise Exception("Test not written yet")

    def test_InsetVolume(self):
        """Test case where one image volume is entirely contained within the other."""
        raise Exception("Test not written yet")

    def test_OutputMatchesROI(self):
        """Ensure that the bounds of the output image match the bounds of a specified
        ROI within one voxel spacing
        """
        raise Exception("Test not written yet")

    def test_StitchVolumes1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """
        """ This test loads the MRHead sample image volume, clones it and translates it 
        50 mm in the superior direction, and then stitches it together with the untranslated
        original.  An ROI is created which is fitted to the original image volume and then 
        symmetrically expanded 50 mm in the Superior-Inferior direction.  The stitched
        image volume size is set by the ROI, so there is a 25 mm inferior region which is
        all zeros because it is outside both image volumes. The top 25 mm of the
        translated image is cropped off because it is outside the ROI. Finally, there
        is a visible seam halfway into the overlap region (which is correct in this case 
        because they should not seamlessly meet).  All that is verified by the current
        test is that the stitching runs without error, and that the bounds of the 
        stitched volume are very close to the bounds of the ROI. 
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        inputVolume = SampleData.downloadFromURL(
            nodeNames="MRHead",
            fileNames="MR-Head.nrrd",
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/MD5/39b01631b7b38232a220007230624c8e",
            checksums="MD5:39b01631b7b38232a220007230624c8e",
        )[0]
        self.delayDisplay("Finished with download and loading")

        volumeCopy = slicer.vtkSlicerVolumesLogic().CloneVolume(
            slicer.mrmlScene, inputVolume, "cloned_copy"
        )

        # Create transform matrix with 50mm translation
        import numpy as np

        transformMatrixForCopy = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 50], [0, 0, 0, 1]]
        )
        TNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        TNode.SetAndObserveMatrixTransformToParent(
            slicer.util.vtkMatrixFromArray(transformMatrixForCopy)
        )
        # Apply transform to cloned copy and harden
        volumeCopy.SetAndObserveTransformNodeID(TNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(volumeCopy)

        # Create markupsROI and fit to input volume
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        # Set the axes directions of the roi to match those of the image volume
        # (if we don't do this before fitting using CropVolumes the ROI based image
        # directions can be permuted versions of the original image directions, and
        # we want them to match exactly)
        imageDirectionMatrix = vtk.vtkMatrix4x4()
        volumeCopy.GetIJKToRASDirectionMatrix(imageDirectionMatrix)
        roiNode.SetAndObserveObjectToNodeMatrix(imageDirectionMatrix)

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLCropVolumeParametersNode"
        )
        cropVolumeParameters.SetInputVolumeNodeID(inputVolume.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(
            cropVolumeParameters
        )  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        # Expand ROI to include some of the copy volume and some empty space
        sz = list(roiNode.GetSize())
        sz[1] = sz[1] + 50  # axis 1 is the superior-inferior axis for MRHead
        roiNode.SetSize(*sz)

        # Test the module logic

        logic = StitchVolumesLogic()
        stitched_node = logic.stitch_volumes(
            [inputVolume, volumeCopy],
            roiNode,
            None,
            keepIntermediateVolumes=False,
        )

        # Check results

        # Check that stitched image bounds are very close to ROI edges
        stitched_bnds = np.zeros((6))
        stitched_node.GetBounds(stitched_bnds)
        roi_bnds = np.zeros((6))
        roiNode.GetBounds(roi_bnds)
        maxVoxelSize = np.max(stitched_node.GetSpacing())
        maxBndsDeviation = np.max(np.abs(roi_bnds - stitched_bnds))
        self.assertLess(
            maxBndsDeviation,
            maxVoxelSize,
            msg="RAS bounds of stitched volume are greater than 1 voxel off from bounds of ROI!",
        )

        # TODO: implement more tests, for example
        # Could also spot check voxel values
        # - outside both volumes should be 0
        # - the outer corner voxel values should match
        # - the inner corner voxel values (in the overlap region) should not

        self.delayDisplay("Test passed")

    def test_StitchVolumes2(self):
        """
        This test is identical to test_StitchVolumes1(), except that the transform on the
        translated cloned copy of MRHead is not hardened.  The result should be identical
        to the result from test_StitchVolumes1 (but won't be if soft transforms are not
        respected, that's the test).
        This test loads the MRHead sample image volume, clones it, and a applies a
        soft transform which translates it 50 mm in the superior direction, and then
        stitches it together with the untransformed original.  An ROI is created which is
        fitted to the original image volume and then
        symmetrically expanded 50 mm in the Superior-Inferior direction.  The stitched
        image volume size is set by the ROI, so there is a 25 mm inferior region which is
        all zeros because it is outside both image volumes. The top 25 mm of the
        translated image is cropped off because it is outside the ROI. Finally, there
        is a visible seam halfway into the overlap region (which is correct in this case
        because they should not seamlessly meet).  All that is verified by the current
        test is that the stitching runs without error, and that the bounds of the
        stitched volume are very close to the bounds of the ROI.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        inputVolume = SampleData.downloadFromURL(
            nodeNames="MRHead",
            fileNames="MR-Head.nrrd",
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/MD5/39b01631b7b38232a220007230624c8e",
            checksums="MD5:39b01631b7b38232a220007230624c8e",
        )[0]
        self.delayDisplay("Finished with download and loading")

        volumeCopy = slicer.vtkSlicerVolumesLogic().CloneVolume(
            slicer.mrmlScene, inputVolume, "cloned_copy"
        )

        # Create transform matrix with 50mm translation
        import numpy as np

        transformMatrixForCopy = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 50], [0, 0, 0, 1]]
        )
        TNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        TNode.SetAndObserveMatrixTransformToParent(
            slicer.util.vtkMatrixFromArray(transformMatrixForCopy)
        )
        # Apply transform to cloned copy and harden
        volumeCopy.SetAndObserveTransformNodeID(TNode.GetID())
        # slicer.vtkSlicerTransformLogic().hardenTransform(volumeCopy)

        # Create markupsROI and fit to input volume
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        # Set the axes directions of the roi to match those of the image volume
        # (if we don't do this before fitting using CropVolumes the ROI based image
        # directions can be permuted versions of the original image directions, and
        # we want them to match exactly)
        imageDirectionMatrix = vtk.vtkMatrix4x4()
        volumeCopy.GetIJKToRASDirectionMatrix(imageDirectionMatrix)
        roiNode.SetAndObserveObjectToNodeMatrix(imageDirectionMatrix)

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLCropVolumeParametersNode"
        )
        cropVolumeParameters.SetInputVolumeNodeID(inputVolume.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(
            cropVolumeParameters
        )  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        # Expand ROI to include some of the copy volume and some empty space
        sz = list(roiNode.GetSize())
        sz[1] = sz[1] + 50  # axis 1 is the superior-inferior axis for MRHead
        roiNode.SetSize(*sz)

        # Test the module logic

        logic = StitchVolumesLogic()
        stitched_node = logic.stitch_volumes(
            [inputVolume, volumeCopy],
            roiNode,
            None,
            keepIntermediateVolumes=False,
        )

        # Check results

        # Check that stitched image bounds are very close to ROI edges
        stitched_bnds = np.zeros((6))
        stitched_node.GetBounds(stitched_bnds)
        roi_bnds = np.zeros((6))
        roiNode.GetBounds(roi_bnds)
        maxVoxelSize = np.max(stitched_node.GetSpacing())
        maxBndsDeviation = np.max(np.abs(roi_bnds - stitched_bnds))
        self.assertLess(
            maxBndsDeviation,
            maxVoxelSize,
            msg="RAS bounds of stitched volume are greater than 1 voxel off from bounds of ROI!",
        )

        # TODO: implement more tests, for example
        # Could also spot check voxel values
        # - outside both volumes should be 0
        # - the outer corner voxel values should match
        # - the inner corner voxel values (in the overlap region) should not

        self.delayDisplay("Test passed")


####################
#
# Subfunctions
#
####################


def get_RAS_center(vol_node):
    """Find the RAS coordinate center of the image volume from the RAS bounds"""
    b = [0] * 6
    vol_node.GetRASBounds(
        b
    )  # GetRASBounds() takes parent transforms into account, unlike GetBounds()!
    cen = [np.mean([b[0], b[1]]), np.mean([b[2], b[3]]), np.mean([b[4], b[5]])]
    return cen


def find_dim_to_stitch(orig_nodes, resamp_node):
    # This function determines the dimension to stitch the original nodes along by
    # finding the image axis dimension (I,J,or K) which is best aligned with the
    # vector between the centers of the furthest apart original volumes.
    # A resampled volume is needed just in case its IJK direction matrix
    # differs from the original nodes. I believe this method should be
    # fairly robust.
    RAS_centers = [get_RAS_center(vol) for vol in orig_nodes]
    dists = [
        np.linalg.norm(np.subtract(RAS_center, RAS_centers[0]))
        for RAS_center in RAS_centers
    ]
    furthest_from_first = np.argmax(dists)
    stitch_vect = np.subtract(RAS_centers[0], RAS_centers[furthest_from_first])
    stitch_vect = stitch_vect / np.linalg.norm(stitch_vect)
    # RAS_biggest_change_idx= np.argmax(np.abs(stitch_vect))
    # Now I need to know which image volume axis (I,J,or K) is most aligned with the stitching vector
    # We can do this by comparing the dot products of each of the I J and K vectors with the stitch
    # vector.  The one with the maximum abs dot product is the winner
    ijkdirs = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]  # NOTE these will be the ROWS, not columns of ijk to ras matrix
    resamp_node.GetIJKToRASDirections(ijkdirs)  # fill in values
    ijkdirs_np = np.array(ijkdirs)
    # Compute dot products with the columns of ijk to ras matrix
    absDotsIJK = [np.abs(np.dot(d, stitch_vect)) for d in ijkdirs_np.T]
    IJKmatchIdx = np.argmax(absDotsIJK)
    KJImatchIdx = 2 - IJKmatchIdx
    dim_to_stitch = KJImatchIdx
    return dim_to_stitch


def createOrReplaceNode(name, nodeClass="vtkMRMLScalarVolumeNode"):
    try:
        node = slicer.util.getNode(name)
    except:
        node = slicer.mrmlScene.AddNewNodeByClass(nodeClass, name)
    return node


def resample_volume(roi_node, input_vol_node, output_vol_name):
    """Carry out the cropping of input_vol_node to the space described by roi_node"""
    cropVolumeNode = slicer.vtkMRMLCropVolumeParametersNode()
    cropVolumeNode.SetScene(slicer.mrmlScene)
    cropVolumeNode.SetName("MyCropVolumeParametersNode")
    cropVolumeNode.SetIsotropicResampling(False)
    cropVolumeNode.SetInterpolationMode(
        cropVolumeNode.InterpolationNearestNeighbor
    )  # use nearest neighbor to avoid resampling artifacts
    cropVolumeNode.SetFillValue(
        0
    )  # needs to be zero so that sum of filled slices is zero
    cropVolumeNode.SetROINodeID(roi_node.GetID())  # roi
    slicer.mrmlScene.AddNode(cropVolumeNode)
    output_vol_node = createOrReplaceNode(output_vol_name, "vtkMRMLScalarVolumeNode")
    cropVolumeNode.SetInputVolumeNodeID(input_vol_node.GetID())  # input
    cropVolumeNode.SetOutputVolumeNodeID(output_vol_node.GetID())  # output
    slicer.modules.cropvolume.logic().Apply(cropVolumeNode)  # do the crop
    slicer.mrmlScene.RemoveNode(cropVolumeNode)
    return output_vol_node


def resample(
    vol_node_to_resample,
    reference_vol_node,
    output_vol_node=None,
    interpolationMode="NearestNeighbor",
    defaultValue=0,
):
    """Handle resampling a second node based on the geometry of reference node."""
    # Switch method and warn if NearestNeighbor is selected and inappropriate
    if interpolationMode == "NearestNeighbor":
        import numpy as np

        maxVoxDimDiff = np.max(
            np.abs(
                np.subtract(
                    reference_vol_node.GetSpacing(), vol_node_to_resample.GetSpacing()
                )
            )
        )
        if maxVoxDimDiff > 1e-4:
            interpolationMode = "Linear"
            logging.warning(
                "Automatically switching from NearestNeighbor interpolation to Linear interpolation because the volume to resample (%s) has a different resolution (%0.2fmm x %0.2fmm x %0.2fmm) than the first original volume (%s, %0.2fmm x %0.2fmm x %0.2fmm)"
                % (
                    vol_node_to_resample.GetName(),
                    *vol_node_to_resample.GetSpacing(),
                    reference_vol_node.GetName(),
                    *reference_vol_node.GetSpacing(),
                )
            )
    inputVolID = vol_node_to_resample.GetID()
    refVolID = reference_vol_node.GetID()
    if output_vol_node is None:
        output_vol_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    outputVolID = output_vol_node.GetID()
    params = {
        "inputVolume": inputVolID,
        "referenceVolume": refVolID,
        "outputVolume": outputVolID,
        "interpolationMode": interpolationMode,
        "defaultValue": defaultValue,
    }
    slicer.cli.runSync(slicer.modules.brainsresample, None, params)
    return output_vol_node


def getSliceIndexTuple(start, end, dim_to_stitch, nDims=3):
    # Constructs a tuple which can be used as an index into a 3D array
    # To illustrate, if the dim_to_stitch were 1, the output would be
    # (slice(None),slice(start:end),slice(None)), which can be used in
    # indexing into a 3D array equivalently to arr[:,start:end,:]
    sliceIndexList = []
    for dim in range(nDims):
        if dim == dim_to_stitch:
            sliceIndexList.append(slice(start, end))
        else:
            sliceIndexList.append(slice(None))
    return tuple(sliceIndexList)


""" def rename_dixon_dicom_volumes(volNodes=None):
    # substitutes the "imageType N" with the Dixon type ("F","W","OP", or "IP")
    # If volume is not a DICOM volume, then it is left unchanged
    import re

    if volNodes is None:
        # Gather all scalar volumes in the scene
        volNodes = []
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneItemID = shNode.GetSceneItemID()
        c = vtk.vtkCollection()
        shNode.GetDataNodesInBranch(sceneItemID, c, "vtkMRMLScalarVolumeNode")
        for idx in range(c.GetNumberOfItems()):
            volNodes.append(c.GetItemAsObject(idx))
    # Loop over all volumes, renaming only if DICOM and if node name matches r"imageType \d"
    for volNode in volNodes:
        uids = volNode.GetAttribute("DICOM.instanceUIDs")  # empty for non DICOM volumes
        imageTypeField = "0008,0008"  # DICOM field corresponding to ImageType
        if uids is not None:
            uid = uids.split()[
                0
            ]  # all of these UIDs have the same ImageType (at least so far as I tested)
            filename = slicer.dicomDatabase.fileForInstance(uid)
            imageType = slicer.dicomDatabase.fileValue(
                filename, imageTypeField
            )  # looks like "DERIVED\PRIMARY\OP\OP\DERIVED"
            dixonType = imageType.split("\\")[
                2
            ]  # pulls out the 3rd entry in that field
            origVolName = volNode.GetName()
            # Substitute dixon type for 'imageType N'
            newName = re.sub(r"imageType \d", dixonType, origVolName)
            volNode.SetName(newName) """
