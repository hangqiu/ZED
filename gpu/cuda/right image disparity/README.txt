 
** This sample demonstrates how to get the disparity of the right image, and how to compute the depth from the disparity.
** For visualization purpose we convert the depth into color and merge it with the current right image           
** Some events are linked with keys(using opencv)
 
This program perform basic warping based on the left disparity, it handle initial occlusions but do not filled them.
Due to warping, occlusions appears in the new DISPARITY/DEPTH map

This sample needs CUDA and OpenCV



		Contact: support@stereolabs.com
