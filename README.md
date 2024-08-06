# SlicerStitchImageVolumes
A 3D Slicer (slicer.org) module for stitching together multiple image volumes into a single larger image volume.

This module allows a user to stitch or blend together two or more image volumes. These image volumes should be positioned in a consistent world coordinate system before stitching (no  registration is performed by this module).  A bounding ROI can be supplied, or one can be automatically generated to enclose all supplied inputs. Output volume resolution can be 
 specified or can mimic an input image. Overlapping regions can be smoothly blended or more discretely stitched together. A voxel threshold can also optionally be supplied; voxels with values at or below the threshold are discarded before stitching. Finally, voxels which are inside the ROI but outside of all input image volumes, or in which all input voxels are below an applied threshold, are filled with a specified default voxel value. Unlike previous versions of this module, input images can be arranged arbitrarily in space, and multiple images overlapping the same region should be handled properly.

 Test example 1:  The image below shows the result of stitching together a cubic image with voxel value 100 (white) and a cubic image with voxel value 0 (gray), using the settings shown on the right. Voxels in the overlap region are a blend of the two overlapping images, weighted such that the closer image is gradually more highly weighted. The bounding ROI is automatically calculated, and voxels outside the original images are assigned the default voxel value (-100 in this case).

<img src="https://github.com/user-attachments/assets/0436b509-2014-4c18-aef7-9fc001930948" width="500">

<img src="https://github.com/user-attachments/assets/068bdc25-90a9-45b0-8cee-58b149258da0" width="500">

Test example 2: Same images and settings as above, except that the weighting method is switched from "Blend" to "Stitch". The "Stitch" method assigns voxels in the overlap region the voxel value of the closest image bounding the region, rather than a distance weighted blending of all overlapping volumes. 

<img src="https://github.com/user-attachments/assets/b5a2e7ed-d131-4d4d-b71d-037475700b54" width="500">

If you use this module, please consider submitting an example of the resulting combined image to show other users what the results look like with real images in actual use cases. 
