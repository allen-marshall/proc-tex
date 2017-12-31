#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "texCoordTransforms.clh"

/*
 * Converts texture coordinates for a 2D sphere-mapped texture to the
 * corresponding coordinates for a 3D texture.
 * radius - Radius of the sphere.
 * center - Center of the sphere.
 * evalPts - Array containing the sphere-mapped points to convert. Each worker
 *   indexes this array by get_global_id(0) to determine its evaluation point.
 * result - Array in which to store the result. Each worker indexes this array
 *   by get_global_id(0) to determine where to store its result.
 */
__kernel void sphereMapTo3D(double radius, double3 center,
  __global const double2 *evalPts, __global double *result)
{
  // Compute the evaluation point, normalized into the base square (unit square
  // centered at (0.5, 0.5)).
  size_t pixelIdx = get_global_id(0);
  double2 evalPt = evalPts[pixelIdx];
  normalizeTexPt2D(&evalPt);
  
  // Compute the Cartesian position corresponding to the evaluation point.
  double3 evalPtCart = texSphericalToCartesian(evalPt, 0.5) + center;
  
  vstore3(evalPtCart, pixelIdx, result);
}
