#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "distMetrics.cl"
#include "texCoordTransforms.cl"
#include "gridCoordTransforms.cl"

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
 * Computes 3D cellular noise for points on a sphere using a modified version of
 * Worley's grid-based cellular noise algorithm.
 * numBoxesH - Number of grid box spaces lying along each axis. Must be at least
 *   1.
 * numPtsPerBox - Number of cell points in each grid box. Must be at least 1.
 * distMetric - Indicates which distance metric to use.
 * cellPts - Array containing the cell center points, grouped by grid box.
 * evalPts - Array containing the (sphere-mapped) points at which to evaluate
 * the noise. Each worker indexes this array by get_global_id[0] to determine
 *   its evaluation point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id[0] to determine where to store its result.
 */
__kernel void sphereCellNoise3D(const uint numBoxesH, const uint numPtsPerBox,
  const distMetric metricID, __global const double *cellPts,
  __global const double2 *evalPts, __global double *result)
{
  // Compute the evaluation point, normalized into the base square (unit square
  // centered at (0.5, 0.5)).
  size_t pixel_idx = get_global_id(0);
  double2 evalPt = evalPts[pixel_idx];
  normalizeTexPt2D(&evalPt);
  
  // Compute the Cartesian position corresponding to the evaluation point.
  double3 evalPtCart = sphericalToCartesian(evalPt, 0.5) + 0.5;
  
  double boxSize = 1.0 / numBoxesH;
  
  // Find the grid box containing the normalized evaluation point.
  uint3 boxCoords = findBoxForPt3D(boxSize, evalPtCart);
  
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
          double3 delta = loopingDelta3D(evalPtCart, cellPt);
          double newDist = computeDist3DDelta(metricID, delta);
          if (newDist < minDist) {
            // Uncommenting this line and commenting out other lines that modify
            // result[pixel_idx] can be used to generate a voronoi diagram instead
            // of cellular noise. Might be useful some time.
            // result[pixel_idx] = ptIdx / (double) (numBoxesH * numBoxesH * numBoxesH * numPtsPerBox);
            minDist = newDist;
          }
        }
      }
    }
  }
  
  result[pixel_idx] = minDist;
}
