#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "random.clh"
#include "texCoordTransforms.clh"

void computeBoxBounds(uint numBoxesH, uint numPtsPerBox, uint ptIdx,
  double2 *lowBounds, double2 *highBounds)
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
 * Randomly initializes positions and velocities of cell points for 2D cellular
 * noise.
 * seedBase - Random seed. This will be combined with the worker ID to generate
 *   a separate seed for each worker.
 * numBoxesH - Number of grid box spaces lying along each axis.
 * numPtsPerBox - Number of cell points in each grid box.
 * maxSpeed - Maximum allowed speed for cell points, in space units per frame.
 * cellPts - Array containing the cell center points, grouped by grid box.
 *   Initial positions will be stored here.
 * cellPtVels - Array containing the cell point velocities, grouped by grid box.
 *   Initial velocities will be stored here. Units are space units per frame.
 */
__kernel void cellNoise2DAnimInit(uint seedBase, uint numBoxesH,
  uint numPtsPerBox, double maxSpeed, __global double2 *cellPts,
  __global double2 *cellPtVels)
{
  size_t ptIdx = get_global_id(0);
  uint seed = initWorkerSeed(seedBase, ptIdx);
  
  // Generate random initial position.
  double2 lowBounds, highBounds;
  computeBoxBounds(numBoxesH, numPtsPerBox, ptIdx, &lowBounds, &highBounds);
  double2 pos = (double2) (
    randDoubleInRange(&seed, lowBounds.x, highBounds.x),
    randDoubleInRange(&seed, lowBounds.y, highBounds.y));
  cellPts[ptIdx] = pos;
  
  // Generate random initial velocity.
  double2 vel = randVecWithMagnitude2D(&seed,
    randDoubleInRange(&seed, 0, maxSpeed));
  cellPtVels[ptIdx] = vel;
}

/*
 * Updates positions of cell points for 2D cellular noise using random
 * acceleration.
 * seedBase - Random seed. This will be combined with the worker ID to generate
 *   a separate seed for each worker.
 * numBoxesH - Number of grid box spaces lying along each axis.
 * numPtsPerBox - Number of cell points in each grid box.
 * maxSpeed - Maximum allowed speed for cell points, in space units per frame.
 * maxAccel - Maximum allowed acceleration for cell points, in space units per
 *   frame squared.
 * cellPts - Array containing the cell center points, grouped by grid box.
 *   Updated positions will be stored here.
 * cellPtVels - Array containing the cell point velocities, grouped by grid box.
 *   Updated velocities will be stored here. Units are space units per frame.
 */
__kernel void cellNoise2DAnimUpdate(uint seedBase, uint numBoxesH,
  uint numPtsPerBox, double maxSpeed, double maxAccel,
  __global double2 *cellPts, __global double2 *cellPtVels)
{
  size_t ptIdx = get_global_id(0);
  uint seed = initWorkerSeed(seedBase, ptIdx);
  
  // Generate random acceleration.
  double2 accel = randVecWithMagnitude2D(&seed,
    randDoubleInRange(&seed, 0, maxAccel));
  
  // Compute new velocity and position. Since the time units are frames, we can
  // ignore delta time in the calculations.
  double2 newVel = cellPtVels[ptIdx] + accel;
  double speedSquared = newVel.x * newVel.x + newVel.y * newVel.y;
  if (speedSquared > maxSpeed * maxSpeed) {
    newVel *= maxSpeed / sqrt(speedSquared);
  }
  double2 newPos = cellPts[ptIdx] + newVel;
  
  // Clamp the position and velocity based on the grid box boundaries.
  double2 lowBounds, highBounds;
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
