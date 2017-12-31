#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "random.clh"
#include "texCoordTransforms.clh"
#include "gridCoordTransforms.clh"

/*
 * Computes simple 3D grid noise.
 * seedBase - Random seed. Will be combined with the grid box coordinates to get
 *   a consistent value for each grid box.
 * numBoxesH - Number of grid box spaces lying along each axis. Must be at least
 *   1.
 * evalPts - Array containing the 3D points at which to evaluate the noise. Each
 *   worker indexes this array by get_global_id(0) to determine its evaluation
 *   point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id(0) to determine where to store its result.
 */
__kernel void gridNoise3D(uint seedBase, uint numBoxesH,
  __global const double *evalPts, __global double *result)
{
  // Compute the evaluation point, normalized into the base cube (unit cube
  // centered at (0.5, 0.5, 0.5)).
  size_t pixelIdx = get_global_id(0);
  double3 evalPt = vload3(pixelIdx, evalPts);
  normalizeTexPt3D(&evalPt);
  
  double boxSize = 1.0 / numBoxesH;
  
  // Find the grid box containing the normalized evaluation point.
  uint3 boxCoords = findBoxForPt3D(boxSize, evalPt);
  
  // Seed by the box coordinates instead of the worker ID so that the same value
  // will be generated for pixels in the same grid box.
  uint seed = initWorkerSeed(seedBase,
    (boxCoords.z * numBoxesH + boxCoords.y) * numBoxesH + boxCoords.x);
  
  result[pixelIdx] = randDouble(&seed);
}
