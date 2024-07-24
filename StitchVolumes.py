import os
import unittest
import logging
import vtk, qt, ctk, slicer
import numpy as np
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# StitchVolumes
#


class StitchVolumes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Stitch Volumes"
        self.parent.categories = [
            "MikeTools"
        ]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Mike Bindschadler (Seattle Children's Hospital)"]
        self.parent.helpText = """
    This module allows a user to stitch together two or more volumes.  A set of volumes to stitch, as well
    as a rectangular ROI (to define the output geometry) is supplied, and this module produces an output
    volume which represents all the input volumes cropped, resampled, and stitched together. Areas of overlap
    between original volumes are handled by finding the center of the overlap region, and assigning each half
    of the overlap to the closer original volume.  
"""
        self.parent.helpText += (
            self.getDefaultModuleDocumentationLink()
        )  # TODO: verify that the default URL is correct or change it to the actual documentation
        self.parent.acknowledgementText = """
    This work was funded by Seattle Children's Hospital.
"""  # TODO: replace with organization, grant and thanks.


#
# StitchVolumesWidget
#


class StitchVolumesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
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

        # Set scene in MRML widgets. Make sure that in Qt designer
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        """# Example of adding widgets dynamically (without Qt designer).
    # This approach is not recommended, but only shown as an illustrative example.
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "More"
    parametersCollapsibleButton.collapsed = True
    self.layout.addWidget(parametersCollapsibleButton)
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    self.invertedOutputSelector = slicer.qMRMLNodeComboBox()
    self.invertedOutputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.invertedOutputSelector.addEnabled = True
    self.invertedOutputSelector.removeEnabled = True
    self.invertedOutputSelector.noneEnabled = True
    self.invertedOutputSelector.setMRMLScene(slicer.mrmlScene)
    self.invertedOutputSelector.setToolTip("Result with inverted threshold will be written into this volume")
    parametersFormLayout.addRow("Inverted output volume: ", self.invertedOutputSelector)
    """
        # Create a new parameterNode
        # This parameterNode stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.logic = StitchVolumesLogic()
        self.ui.parameterNodeSelector.addAttribute(
            "vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName
        )
        self.setParameterNode(self.logic.getParameterNode())

        # Connections
        self.ui.parameterNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.setParameterNode
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.roiSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.volumeSelector1.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.volumeSelector2.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.volumeSelector3.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.volumeSelector4.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.volumeSelector5.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Initial GUI update
        self.updateGUIFromParameterNode()

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

        # Unobserve previusly selected parameter node and add an observer to the newly selected.
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

        # Disable all sections if no parameter node is selected
        self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None
        # self.ui.advancedCollapsibleButton.enabled = self._parameterNode is not None
        if self._parameterNode is None:
            return

        # Update each widget from parameter node
        # Need to temporarily block signals to prevent infinite recursion (MRML node update triggers
        # GUI update, which triggers MRML node update, which triggers GUI update, ...)

        wasBlocked = self.ui.roiSelector.blockSignals(True)
        self.ui.roiSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("StitchedVolumeROI")
        )
        self.ui.roiSelector.blockSignals(wasBlocked)
        wasBlocked = self.ui.volumeSelector1.blockSignals(True)
        self.ui.volumeSelector1.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVol1")
        )
        self.ui.volumeSelector1.blockSignals(wasBlocked)
        wasBlocked = self.ui.volumeSelector2.blockSignals(True)
        self.ui.volumeSelector2.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVol2")
        )
        self.ui.volumeSelector2.blockSignals(wasBlocked)
        wasBlocked = self.ui.volumeSelector3.blockSignals(True)
        self.ui.volumeSelector3.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVol3")
        )
        self.ui.volumeSelector3.blockSignals(wasBlocked)
        wasBlocked = self.ui.volumeSelector4.blockSignals(True)
        self.ui.volumeSelector4.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVol4")
        )
        self.ui.volumeSelector4.blockSignals(wasBlocked)
        wasBlocked = self.ui.volumeSelector5.blockSignals(True)
        self.ui.volumeSelector5.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVol5")
        )
        self.ui.volumeSelector5.blockSignals(wasBlocked)

        # What about other values? (current text, e.g.)?  The example code did not update them here

        # Update buttons states and tooltips
        # Enable the Stitch Volumes button if there is an ROI, at least two original volumes, and a name for the output vol
        if (
            self._parameterNode.GetNodeReference("StitchedVolumeROI")
            and self._parameterNode.GetNodeReference("InputVol1")
            and self._parameterNode.GetNodeReference("InputVol2")
            and self._parameterNode.GetParameter("OutputVolName")
        ):
            self.ui.applyButton.toolTip = "Compute stitched volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Enter inputs to enable stitching"
            self.ui.applyButton.enabled = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None:
            return

        self._parameterNode.SetNodeReferenceID(
            "StitchedVolumeROI", self.ui.roiSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputVol1", self.ui.volumeSelector1.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputVol2", self.ui.volumeSelector2.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputVol3", self.ui.volumeSelector3.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputVol4", self.ui.volumeSelector4.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "InputVol5", self.ui.volumeSelector5.currentNodeID
        )
        self._parameterNode.SetParameter(
            "OutputVolName", self.ui.stitchVolNameLineEdit.text
        )

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.invertedOutputSelector.currentNodeID)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            # Gather inputs
            orig_nodes = self.gather_original_nodes()
            roi_node = self.ui.roiSelector.currentNode()
            stitched_vol_name = self.ui.stitchVolNameLineEdit.text
            # Run the stitching
            self.logic.stitch_volumes(
                orig_nodes, roi_node, stitched_vol_name, keep_intermediate_volumes=False
            )

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
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("OutputVolName"):
            parameterNode.SetParameter("OutputVolName", "S")

    def stitch_volumes(
        self, orig_nodes, roi_node, stitched_vol_name, keep_intermediate_volumes=False
    ):
        # Stitch together the supplied original volumes, resampling them
        # into the space defined by the supplied roi, putting the stitched
        # output into a volume with the given stitched volume name

        # Crop/Resample first orig node
        ref_vol_node = resample_volume(roi_node, orig_nodes[0], "ReferenceVolume")
        # Resample other nodes
        resamp_vol_nodes = []
        for orig_node in orig_nodes:
            resampled_name = "Resamp_" + orig_node.GetName()
            resamp_node = createOrReplaceNode(resampled_name)
            resamp_vol_nodes.append(resample(orig_node, ref_vol_node, resamp_node))
        imArrays = [
            slicer.util.arrayFromVolume(resamp_vol_node)
            for resamp_vol_node in resamp_vol_nodes
        ]
        # Create output stitched volume node, create by cloning one of the resamp nodes
        # (it doesn't matter which one, it's just being used to get orientation and spacing)
        stitched_vol_node = slicer.vtkSlicerVolumesLogic().CloneVolume(
            slicer.mrmlScene, resamp_vol_nodes[0], stitched_vol_name
        )
        # Find the dimension to stitch together (I,J,or K)
        dim_to_stitch = find_dim_to_stitch(orig_nodes, resamp_vol_nodes[0])
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
        
        # Filling voxel values in stitched volume
        # Prepopulate stitched area with the minimum value in the first of the original images (i.e. air which has -1000 in CT or 0 in MRI). This could error if there are no air voxels in the first image volume - if there is another way to find the normalised value of CT/MRI/other modalities that could be preferable...
        imCombined = np.full(imArrays[0].shape, np.min(orderedImArrays[0].flatten()))
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
            
            # Using the maximum intensity between corresponding voxels in what will be the output volume.
            imCombined = np.maximum(imCombined, imArray)        
            # print(sliceIndexTuple)

        # Put the result into the stitched volume
        slicer.util.updateVolumeFromArray(stitched_vol_node, imCombined)
        # Clean up
        if not keep_intermediate_volumes:
            for resamp_vol_node in resamp_vol_nodes:
                slicer.mrmlScene.RemoveNode(resamp_vol_node)
            slicer.mrmlScene.RemoveNode(ref_vol_node)
        # Return stitched volume node
        return stitched_vol_node

    def run_resample_stitched(
        self,
        stitchedVolume,
        resampledStitchedVolume,
        pixelResolutionMm=1,
        coronalSliceThicknessMm=1,
    ):
        """Resample the input volume to the output resolution, defaulting to
        1 mm isotropic resolution, but allowing thicker coronal slices if desired.
        """
        return

    def run_reorient(self, inputVolume, outputVolume, newOrientation="Coronal"):
        """Run the Orient Scalar Volume CLI module to reorient voxel order
        to new orientation, default Coronal
        """
        parameters = {
            "inputVolume1": inputVolume.GetID(),
            "outputVolume": outputVolume.GetID(),
            "orientation": newOrientation,
        }
        cliNode = slicer.cli.runSync(
            slicer.modules.orientscalarvolume, None, parameters
        )
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
        # success
        slicer.mrmlScene.RemoveNode(cliNode)
        return  # no return values needed


#
# StitchVolumesTest
#


class StitchVolumesTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_StitchVolumes1()

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

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 279)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 50

        # Test the module logic

        logic = StitchVolumesLogic()

        # Test algorithm with non-inverted threshold
        logic.run(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.run(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")


####################
#
# Subfunctions
#
####################


def get_RAS_center(vol_node):
    b = [0] * 6
    vol_node.GetBounds(b)
    cen = [np.mean([b[0], b[1]]), np.mean([b[2], b[3]]), np.mean([b[4], b[5]])]
    return cen


def ras_to_ijk(
    point_ras, vol_node, return_ints_flag=False, use_volume_transform_flag=True
):
    # Return the IJK coord corresponding to the RAS location
    # of the supplied point in the given volume.

    if use_volume_transform_flag:
        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, vol_node.GetParentTransformNode(), transformRasToVolumeRas
        )
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_ras[0:3])
    else:
        point_VolumeRas = point_ras
    # Get voxel coordinates from physical coordinates
    volumeRasToIjk = vtk.vtkMatrix4x4()
    vol_node.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
    # Trim homogenous coord
    point_ijk = point_Ijk[0:3]
    # Round to integers if requested
    if return_ints_flag:
        point_ijk = [int(round(c)) for c in point_ijk]
    return point_ijk


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
    ijkdirs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    resamp_node.GetIJKToRASDirections(ijkdirs)
    absDotsIJK = [np.abs(np.dot(d, stitch_vect)) for d in ijkdirs]
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
    # Carry out the cropping
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
):
    # Handle resampling a second node based on the geometry of reference node.
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
        # Want to set this to the lowest values in any volume (i.e. air). 
        "defaultValue": -1000, # Currently -1000 to CT, but will need to change to a modality agnostic value. 
        # Development opportunity: Could do something similar to imCombined prepopulation in the def_stitch_volumes(), class StitchVolumesLogic.
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


def rename_dixon_dicom_volumes(volNodes=None):
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
            volNode.SetName(newName)
