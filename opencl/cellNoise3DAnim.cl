#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "texCoordTransforms.clh"

void computeBoxBounds(const uint numBoxesH, const uint numPtsPerBox,
  const uint ptIdx, double3 *lowBounds, double3 *highBounds)
{
  double boxWidth = 1.0 / numBoxesH;
  uint boxIdx = ptIdx / numPtsPerBox;
  uint boxX = boxIdx % numBoxesH;
  uint boxY = (boxIdx / numBoxesH) % numBoxesH;
  uint boxZ = (boxIdx / numBoxesH) / numBoxesH;
  if (lowBounds) {
    *lowBounds = (double3) (boxX, boxY, boxZ) * boxWidth;
  }
  if (highBounds) {
    *highBounds = (double3) (boxX, boxY, boxZ) * boxWidth + boxWidth;
  }
}

/*
 * Updates positions of cell points for 3D cellular noise using the specified
 * velocities and accelerations.
 * numBoxesH - Number of grid box spaces lying along each axis.
 * numPtsPerBox - Number of cell points in each grid box.
 * maxSpeed - Maximum allowed speed for cell points, in space units per frame.
 * cellPts - Array containing the cell center points, grouped by grid box.
 *   Updated positions will be stored here.
 * cellVels - Array containing the cell point velocities, grouped by grid box.
 *   Updated velocities will be stored here. Units are space units per frame.
 * cellPtAccels - Array containing the cell point accelerations, grouped by grid
 *   box, in spherical coordinates. Units are space units per frame squared.
 */
__kernel void cellNoise3DAnimUpdate(const uint numBoxesH,
  const uint numPtsPerBox, const double maxSpeed, __global double *cellPts,
  __global double *cellPtVels, __global const double *cellPtAccels)
{
  size_t ptIdx = get_global_id(0);
  // Convert acceleration to Cartesian coordinates.
  double3 accelSpherical = vload3(ptIdx, cellPtAccels);
  double3 accelCart = sphericalToCartesian(accelSpherical);
  
  // Compute new velocity and position. Since the time units are frames, we can
  // ignore delta time in the calculations.
  double3 newVel = vload3(ptIdx, cellPtVels) + accelCart;
  double speedSquared = newVel.x * newVel.x + newVel.y * newVel.y
    + newVel.z * newVel.z;
  if (speedSquared > maxSpeed * maxSpeed) {
    newVel *= maxSpeed / sqrt(speedSquared);
  }
  double3 newPos = vload3(ptIdx, cellPts) + newVel;
  
  // Clamp the position and velocity based on the grid box boundaries.
  double3 lowBounds;
  double3 highBounds;
  computeBoxBounds(numBoxesH, numPtsPerBox, ptIdx, &lowBounds, &highBounds);
  if (newPos.x < lowBounds.x) {
    newPos.x = lowBounds.x;
    newVel.x = 0;
  }
  else if (newPos.x > highBounds.x) {
    newPos.x = highBounds.x;
    newVel.x = 0;
  }
  if (newPos.y < lowBounds.y) {
    newPos.y = lowBounds.y;
    newVel.y = 0;
  }
  else if (newPos.y > highBounds.y) {
    newPos.y = highBounds.y;
    newVel.y = 0;
  }
  if (newPos.z < lowBounds.z) {
    newPos.z = lowBounds.z;
    newVel.z = 0;
  }
  else if (newPos.z > highBounds.z) {
    newPos.z = highBounds.z;
    newVel.z = 0;
  }
  
  // Store results.
  vstore3(newVel, ptIdx, cellPtVels);
  vstore3(newPos, ptIdx, cellPts);
}
