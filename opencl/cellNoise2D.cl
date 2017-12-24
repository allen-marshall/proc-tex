#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "distMetrics.clh"
#include "texCoordTransforms.clh"
#include "gridCoordTransforms.clh"

/*
 * Determines the range of indices in the cell point array at which to find the
 * cell points contained in the specified grid box. Returns the first index
 * and last index plus one as a uint2.
 */
uint2 getPtIdxRange(uint numBoxesH, uint numPtsPerBox, int2 boxCoords)
{
  uint2 coords = normalizeBoxCoords2D(numBoxesH, boxCoords);
  uint boxIdx = coords.y * numBoxesH + coords.x;
  uint firstPtIdx = boxIdx * numPtsPerBox;
  return (uint2) (firstPtIdx, firstPtIdx + numPtsPerBox);
}

/*
 * Computes 2D cellular noise using a modified version of Worley's grid-based
 * cellular noise algorithm.
 * numBoxesH - Number of grid box spaces lying along each axis. Must be at least
 *   1.
 * numPtsPerBox - Number of cell points in each grid box. Must be at least 1.
 * distMetric - Indicates which distance metric to use.
 * cellPts - Array containing the cell center points, grouped by grid box.
 * evalPts - Array containing the points at which to evaluate the noise. Each
 *   worker indexes this array by get_global_id[0] to determine its evaluation
 *   point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id[0] to determine where to store its result.
 */
__kernel void cellNoise2D(const uint numBoxesH, const uint numPtsPerBox,
  const distMetric metricID, __global const double2 *cellPts,
  __global const double2 *evalPts, __global double *result)
{
  // Compute the evaluation point, normalized into the base square (unit square
  // centered at (0.5, 0.5)).
  size_t pixelIdx = get_global_id(0);
  double2 evalPt = evalPts[pixelIdx];
  normalizeTexPt2D(&evalPt);
  
  double boxSize = 1.0 / numBoxesH;
  
  // Find the grid box containing the normalized evaluation point.
  uint2 boxCoords = findBoxForPt2D(boxSize, evalPt);
  
  // Apply a modified Worley's algorithm to find the distance to the closest
  // cell point.
  double minDist = INFINITY;
  for (int boxX = ((int) boxCoords.x) - 1; boxX <= (int) boxCoords.x + 1;
    boxX++)
  {
    for (int boxY = ((int) boxCoords.y) - 1; boxY <= (int) boxCoords.y + 1;
      boxY++)
    {
      uint2 ptIdxRange = getPtIdxRange(numBoxesH, numPtsPerBox,
        (int2) (boxX, boxY));
      for (uint ptIdx = ptIdxRange.x; ptIdx < ptIdxRange.y; ptIdx++) {
        double2 cellPt = cellPts[ptIdx];
        double2 delta = loopingDelta2D(evalPt, cellPt);
        double newDist = computeDist2DDelta(metricID, delta);
        if (newDist < minDist) {
          // Uncommenting this line and commenting out other lines that modify
          // result[pixelIdx] can be used to generate a voronoi diagram instead
          // of cellular noise. Might be useful some time.
          // result[pixelIdx] = ptIdx / (double) (numBoxesH * numBoxesH * numPtsPerBox);
          minDist = newDist;
        }
      }
    }
  }
  
  result[pixelIdx] = minDist;
}
