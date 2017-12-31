#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "distMetrics.clh"
#include "texCoordTransforms.clh"
#include "gridCoordTransforms.clh"

/*
 * Determines the range of indices in the cell point array at which to find the
 * cell points contained in the specified grid box. Returns the first index
 * and last index plus one as a uint2.
 */
uint2 getPtIdxRange(uint numBoxesH, uint numPtsPerBox, int3 boxCoords)
{
  uint3 coords = normalizeBoxCoords3D(numBoxesH, boxCoords);
  uint boxIdx = (coords.z * numBoxesH + coords.y) * numBoxesH + coords.x;
  uint firstPtIdx = boxIdx * numPtsPerBox;
  return (uint2) (firstPtIdx, firstPtIdx + numPtsPerBox);
}

/*
 * Computes 3D cellular noise using a modified version of Worley's grid-based
 * cellular noise algorithm.
 * numBoxesH - Number of grid box spaces lying along each axis. Must be at least
 *   1.
 * numPtsPerBox - Number of cell points in each grid box. Must be at least 1.
 * distMetric - Indicates which distance metric to use.
 * cellPts - Array containing the cell center points, grouped by grid box.
 * evalPts - Array containing the 3D points at which to evaluate the noise. Each
 *   worker indexes this array by get_global_id[0] to determine its evaluation
 *   point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id[0] to determine where to store its result.
 */
__kernel void cellNoise3D(const uint numBoxesH, const uint numPtsPerBox,
  const distMetric metricID, __global const double *cellPts,
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
  
  // Apply a modified Worley's algorithm to find the distance to the closest
  // cell point.
  double minDist = INFINITY;
  for (int boxX = ((int) boxCoords.x) - 1; boxX <= (int) boxCoords.x + 1;
    boxX++)
  {
    for (int boxY = ((int) boxCoords.y) - 1; boxY <= (int) boxCoords.y + 1;
      boxY++)
    {
      for (int boxZ = ((int) boxCoords.z) - 1; boxZ <= (int) boxCoords.z + 1;
        boxZ++)
      {
        uint2 ptIdxRange = getPtIdxRange(numBoxesH, numPtsPerBox,
          (int3) (boxX, boxY, boxZ));
        for (uint ptIdx = ptIdxRange.x; ptIdx < ptIdxRange.y; ptIdx++) {
          double3 cellPt = vload3(ptIdx, cellPts);
          double3 delta = loopingDelta3D(evalPt, cellPt);
          double newDist = computeDist3DDelta(metricID, delta);
          if (newDist < minDist) {
            // Uncommenting this line and commenting out other lines that modify
            // result[pixelIdx] can be used to generate a voronoi diagram instead
            // of cellular noise. Might be useful some time.
            // result[pixelIdx] = ptIdx / (double) (numBoxesH * numBoxesH * numBoxesH * numPtsPerBox);
            minDist = newDist;
          }
        }
      }
    }
  }
  
  result[pixelIdx] = minDist;
}
