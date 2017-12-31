#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "distMetrics.clh"
#include "texCoordTransforms.clh"
#include "gridCoordTransforms.clh"

double gradDotProd(uint numBoxesH, double boxSize, uint3 boxCoords,
  double3 evalPt, __global const double *gradients)
{
  uint boxIdx = ((boxCoords.z * numBoxesH) + boxCoords.y) * numBoxesH
    + boxCoords.x;
  
  double3 gradPos = convert_double3(boxCoords) * boxSize;
  double3 gradient = vload3(boxIdx, gradients);
  
  // Find smallest displacement, taking spatial looping into account.
  double3 displacement = evalPt - gradPos;
  if (displacement.x > 0.5) {
    displacement.x = displacement.x - 1;
  }
  if (displacement.y > 0.5) {
    displacement.y = displacement.y - 1;
  }
  if (displacement.z > 0.5) {
    displacement.z = displacement.z - 1;
  }
  
  return dot(displacement, gradient);
}

/*
 * Computes 3D Perlin-like noise.
 * numBoxesH - Number of grid box spaces lying along each axis. Must be at least
 *   1.
 * gradients - Array containing the gradient vectors, grouped by grid box.
 * evalPts - Array containing the 3D points at which to evaluate the noise. Each
 *   worker indexes this array by get_global_id[0] to determine its evaluation
 *   point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id[0] to determine where to store its result.
 */
__kernel void perlinNoise3D(uint numBoxesH, __global const double *gradients,
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
  
  // Compute interpolation factors.
  double3 relEvalPt = evalPt / boxSize - floor(evalPt / boxSize);
  double lerpFacX = 1 - smoothstep(0, 1, relEvalPt.x);
  double lerpFacY = 1 - smoothstep(0, 1, relEvalPt.y);
  double lerpFacZ = 1 - smoothstep(0, 1, relEvalPt.z);
  
  // Compute interpolated value for Perlin noise.
  double resultVal = 0;
  uint boxX, boxY, boxZ;
  double currLerpFacX, currLerpFacY, currLerpFacZ;
  for (boxX = boxCoords.x, currLerpFacX = lerpFacX;
    boxX <= boxCoords.x + 1; boxX++, currLerpFacX = 1 - currLerpFacX)
  {
    for (boxY = boxCoords.y, currLerpFacY = lerpFacY;
      boxY <= boxCoords.y + 1; boxY++, currLerpFacY = 1 - currLerpFacY)
    {
      for (boxZ = boxCoords.z, currLerpFacZ = lerpFacZ;
        boxZ <= boxCoords.z + 1; boxZ++, currLerpFacZ = 1 - currLerpFacZ)
      {
        resultVal += currLerpFacX * currLerpFacY * currLerpFacZ
          * gradDotProd(numBoxesH, boxSize,
              normalizeBoxCoords3D(numBoxesH, (int3) (boxX, boxY, boxZ)),
              evalPt, gradients);
      }
    }
  }
  
  result[pixelIdx] = resultVal;
}
