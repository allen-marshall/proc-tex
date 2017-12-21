#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// TODO: Support more distance metrics.
typedef enum type_distMetric {
  L2_NORM,
  L2_NORM_SQUARED
} distMetric;

/*
 * Translates a spatial coordinate into the base square (unit square centered at
 * (0.5, 0.5), taking spatial looping into account so that a seamless texture
 * can be generated.
 */
double normalizeEvalCoord(double coord) {
  // Convert to nonnegative number with same modulus so that fmod's rounding
  // toward zero gives us what we want.
  if (coord < 0) {
    coord += trunc(coord) + 1;
  }
  return fmod(coord, 1);
}

/*
 * Translates a point in 2D space into the base square, taking spatial looping
 * into account so that a seamless texture can be generated.
 */
void normalizeEvalPt(double2 *evalPt) {
  evalPt->x = normalizeEvalCoord(evalPt->x);
  evalPt->y = normalizeEvalCoord(evalPt->y);
}

/*
 * Translates a grid square coordinate to the equivalent coordinate with grid
 * square inside the base square. Similar to normalizeEvalCoord, but for grid
 * coordinates.
 */
uint normalizeSquareCoord(uint gridDim, int coord) {
  if (coord < 0) {
    coord += gridDim;
  }
  else if (coord >= gridDim) {
    coord -= gridDim;
  }
  return (uint) coord;
}

/*
 * Translates grid square coordinates to the equivalent coordinates with grid
 * square inside the base square. Similar to normalizeEvalPt, but for grid
 * coordinates.
 */
uint2 normalizeSquareCoords(uint numGridRows, uint numGridCols, int2 coord) {
  return (uint2) (normalizeSquareCoord(numGridCols, coord.x),
    normalizeSquareCoord(numGridRows, coord.y));
}

/*
 * Determines the (normalized) coordinates of the grid square containing the
 * specified (normalized) point.
 */
uint2 findSquareForPt(double2 squareSize, double2 evalPt) {
  return convert_uint2(trunc(evalPt / squareSize));
}

/*
 * Determines the range of indices in the cell point array at which to find the
 * cell points contained in the specified grid square. Returns the first index
 * and last index plus one as a uint2.
 */
uint2 getPtIdxRange(uint numGridRows, uint numGridCols, uint numPtsInSquare,
  int2 squareCoords)
{
  uint2 coords = normalizeSquareCoords(numGridRows, numGridCols, squareCoords);
  uint squareIdx = coords.y * numGridCols + coords.x;
  uint firstPtIdx = squareIdx * numPtsInSquare;
  return (uint2) (firstPtIdx, firstPtIdx + numPtsInSquare);
}

/*
 * Computes the absolute delta x and delta y between two (normalized) points,
 * taking spatial looping into account so a seamless texture can be generated.
 */
double2 computeDelta(double2 pt0, double2 pt1) {
  double2 result = fabs(pt1 - pt0);
  result = min(result, 1 - result);
  return result;
}

/*
 * Distance metric for L2_NORM.
 */
double distL2Norm(double2 delta) {
  return length(delta);
}

/*
 * Distance metric for L2_NORM_SQUARED.
 */
double distL2NormSquared(double2 delta) {
  return delta.x * delta.x + delta.y * delta.y;
}

/*
 * Computes the distance metric between two points based on the absolute delta x
 * and delta y between the points.
 */
double computeDist(distMetric metricID, double2 delta) {
  switch (metricID) {
  case L2_NORM:
    return distL2Norm(delta);
  case L2_NORM_SQUARED:
    return distL2NormSquared(delta);
  default:
    return distL2Norm(delta);
  }
}

/*
 * Computes cellular noise using a modified version of Worley's grid-based
 * cellular noise algorithm.
 * numGridCols - Number of columns in the grid. Must be at least 1.
 * numGridRows - Number of rows in the grid. Must be at least 1.
 * numPtsInSquare - Number of cell points in each grid square. Must be at least
 *   1.
 * distMetric - Indicates which distance metric to use.
 * cellPts - Array containing the cell points, grouped by grid square.
 * evalPts - Array containing the points at which to evaluate the noise. Each
 *   worker indexes this array by get_global_id[0] to determine its evaluation
 *   point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id[0] to determine where to store its result.
 */
__kernel void cellNoise2D(const uint numGridCols, const uint numGridRows,
  const uint numPtsInSquare, const distMetric metricID,
  __global const double2 *cellPts, __global const double2 *evalPts,
  __global double *result)
{
  // Compute the evaluation point, normalized into the base square (unit square
  // centered at (0.5, 0.5)).
  size_t pixel_idx = get_global_id(0);
  double2 evalPt = evalPts[pixel_idx];
  normalizeEvalPt(&evalPt);
  
  double2 squareSize = (double2) (1.0 / numGridRows, 1.0 / numGridCols);
  
  // Find the grid square containing the normalized evaluation point.
  uint2 evalSquareCoords = findSquareForPt(squareSize, evalPt);
  
  // Apply a modified Worley's algorithm to find the distance to the closest
  // cell point.
  double minDist = INFINITY;
  for (int squareX = ((int) evalSquareCoords.x) - 1;
    squareX <= (int) evalSquareCoords.x + 1; squareX++)
  {
    for (int squareY = ((int) evalSquareCoords.y) - 1;
      squareY <= (int) evalSquareCoords.y + 1; squareY++)
    {
      uint2 ptIdxRange = getPtIdxRange(numGridRows, numGridCols, numPtsInSquare,
        (int2) (squareX, squareY));
      for (uint ptIdx = ptIdxRange.x; ptIdx < ptIdxRange.y; ptIdx++) {
        double2 cellPt = cellPts[ptIdx];
        double2 delta = computeDelta(evalPt, cellPt);
        double newDist = computeDist(metricID, delta);
        if (newDist < minDist) {
          // Uncommenting this line and commenting out other lines that modify
          // result[pixel_idx] can be used to generate a voronoi diagram instead
          // of cellular noise. Might be useful some time.
          // result[pixel_idx] = ptIdx / (double) (numGridCols * numGridRows * numPtsInSquare);
          minDist = newDist;
        }
      }
    }
  }
  
  result[pixel_idx] = minDist;
}
