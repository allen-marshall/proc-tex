#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// TODO: Support more distance metrics.
typedef enum type_distMetric {
  L2_NORM,
  L2_NORM_SQUARED
} distMetric;

double normalizeEvalCoord(double coord) {
  // Convert to nonnegative number with same modulus so that fmod's rounding
  // toward zero gives us what we want.
  if (coord < 0) {
    coord += trunc(coord) + 1;
  }
  return fmod(coord, 1);
}

void normalizeEvalPt(double2 *evalPt) {
  evalPt->x = normalizeEvalCoord(evalPt->x);
  evalPt->y = normalizeEvalCoord(evalPt->y);
}

uint normalizeSquareCoord(uint gridDim, int coord) {
  if (coord < 0) {
    coord += gridDim;
  }
  else if (coord >= gridDim) {
    coord -= gridDim;
  }
  return (uint) coord;
}

uint2 normalizeSquareCoords(uint numGridRows, uint numGridCols, int2 coord) {
  return (uint2) (normalizeSquareCoord(numGridCols, coord.x),
    normalizeSquareCoord(numGridRows, coord.y));
}

uint2 findSquareForPt(double2 squareSize, double2 evalPt) {
  return convert_uint2(trunc(evalPt / squareSize));
}

uint2 getPtIdxRange(uint numGridRows, uint numGridCols, uint numPtsInSquare,
  int2 squareCoords)
{
  uint2 coords = normalizeSquareCoords(numGridRows, numGridCols, squareCoords);
  uint squareIdx = coords.y * numGridCols + coords.x;
  uint firstPtIdx = squareIdx * numPtsInSquare;
  return (uint2) (firstPtIdx, firstPtIdx + numPtsInSquare);
}

double2 computeDelta(double2 pt0, double2 pt1) {
  double2 result = fabs(pt1 - pt0);
  result = min(result, 1 - result);
  return result;
}

double distL2Norm(double2 delta) {
  return length(delta);
}

double distL2NormSquared(double2 delta) {
  return delta.x * delta.x + delta.y * delta.y;
}

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

__kernel void cellNoise2D(const uint numGridCols, const uint numGridRows,
  const uint numPtsInSquare, const distMetric metricID,
  __global const double2 *cellPts, __global const double2 *evalPts,
  __global double *result)
{
  size_t pixel_idx = get_global_id(0);
  double2 evalPt = evalPts[pixel_idx];
  normalizeEvalPt(&evalPt);
  
  double2 squareSize = (double2) (1.0 / numGridRows, 1.0 / numGridCols);
  
  uint2 evalSquareCoords = findSquareForPt(squareSize, evalPt);
  
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
          //result[pixel_idx] = ptIdx / (double) (numGridCols * numGridRows * numPtsInSquare);
          minDist = newDist;
        }
      }
    }
  }
  
  result[pixel_idx] = minDist;
}
