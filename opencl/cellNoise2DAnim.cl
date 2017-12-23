#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "texCoordTransforms.cl"

void computeBoxBounds(const uint numBoxesH, const uint numPtsPerBox,
  const uint ptIdx, double2 *lowBounds, double2 *highBounds)
{
  double boxWidth = 1.0 / numBoxesH;
  uint boxIdx = ptIdx / numPtsPerBox;
  uint boxX = boxIdx % numBoxesH;
  uint boxY = boxIdx / numBoxesH;
  if (lowBounds) {
    *lowBounds = (double2) (boxX, boxY) * boxWidth;
  }
  if (highBounds) {
    *highBounds = (double2) (boxX, boxY) * boxWidth + boxWidth;
  }
}

/*
 * Updates positions of cell points for 2D cellular noise using the specified
 * velocities and accelerations.
 * numBoxesH - Number of grid box spaces lying along each axis.
 * numPtsPerBox - Number of cell points in each grid box.
 * maxSpeed - Maximum allowed speed for cell points, in space units per frame.
 * cellPts - Array containing the cell center points, grouped by grid box.
 *   Updated positions will be stored here.
 * cellVels - Array containing the cell point velocities, grouped by grid box.
 *   Updated velocities will be stored here. Units are space units per frame.
 * cellPtAccels - Array containing the cell point accelerations, grouped by grid
 *   box, in magnitude-angle coordinates. Units are space units per frame
 *   squared.
 */
__kernel void cellNoise2DAnimUpdate(const uint numBoxesH,
  const uint numPtsPerBox, const double maxSpeed, __global double2 *cellPts,
  __global double2 *cellPtVels, __global const double2 *cellPtAccels)
{
  size_t ptIdx = get_global_id(0);
  // Convert acceleration to Cartesian coordinates.
  double2 accel = circularToCartesian(cellPtAccels[ptIdx]);
  
  // Compute new velocity and position. Since the time units are frames, we can
  // ignore delta time in the calculations.
  double2 newVel = cellPtVels[ptIdx] + accel;
  double speedSquared = newVel.x * newVel.x + newVel.y * newVel.y;
  if (speedSquared > maxSpeed * maxSpeed) {
    newVel *= maxSpeed / sqrt(speedSquared);
  }
  double2 newPos = cellPts[ptIdx] + newVel;
  
  // Clamp the position and velocity based on the grid box boundaries.
  double2 lowBounds;
  double2 highBounds;
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
  
  // Store results.
  cellPtVels[ptIdx] = newVel;
  cellPts[ptIdx] = newPos;
}
