# SlicerStitchImageVolumes
A 3D Slicer (slicer.org) module for stitching together multiple image volumes into a single larger image volume.

This simple 3D Slicer module allows a user to stitch together two or more image volumes.  A set of volumes to stitch, as well as a rectangular ROI (to define the output geometry) is supplied, and this module produces an output volume which represents all the input volumes cropped, resampled, and stitched together. Areas of overlap between original volumes are handled by finding the center of the overlap region, and assigning each half of the overlap to the closer original volume.

The resolution (voxel dimensions) of the output stitched volume is set to match the first input image.  If other image volumes are at the same resolution, the stitched volume uses nearest-neighbor interpolation in order to avoid any image degradation due to interpolation, but please note that this could mean that there is a physical space shift of up to 1/2 voxel in each dimension for the positioning of one original volume compared to where it appears in the stitched volume's physical space. If original volumes are not at the same voxel resolution, then interpolation is definitely required, and linear interpolation is used.  Voxels in the stitched image which are outside all original image volumes are assigned a voxel value of zero. 

![Screenshot 2022-04-28 092940](https://user-images.githubusercontent.com/3981795/165800473-12bfe4d3-8e39-40d7-b854-6ef5a9940f7a.jpg)
