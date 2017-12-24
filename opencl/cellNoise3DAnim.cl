#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "random.clh"
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
 * Randomly initializes positions and velocities of cell points for 3D cellular
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
__kernel void cellNoise3DAnimInit(uint seedBase, uint numBoxesH,
  uint numPtsPerBox, double maxSpeed, __global double *cellPts,
  __global double *cellPtVels)
{
  size_t ptIdx = get_global_id(0);
  uint seed = seedBase ^ ptIdx;
  
  // Generate random initial position.
  double3 lowBounds, highBounds;
  computeBoxBounds(numBoxesH, numPtsPerBox, ptIdx, &lowBounds, &highBounds);
  double3 pos = (double3) (
    randDoubleInRange(&seed, lowBounds.x, highBounds.x),
    randDoubleInRange(&seed, lowBounds.y, highBounds.y),
    randDoubleInRange(&seed, lowBounds.z, highBounds.z));
  vstore3(pos, ptIdx, cellPts);
  
  // Generate random initial velocity.
  double3 vel = randVecWithMagnitude3D(&seed,
    randDoubleInRange(&seed, 0, maxSpeed));
  vstore3(vel, ptIdx, cellPtVels);
}

/*
 * Updates positions of cell points for 3D cellular noise using the specified
 * velocities and accelerations.
 * seedBase - Random seed. This will be combined with the worker ID to generate
 *   a separate seed for each worker.
 * numBoxesH - Number of grid box spaces lying along each axis.
 * numPtsPerBox - Number of cell points in each grid box.
 * maxSpeed - Maximum allowed speed for cell points, in space units per frame.
 * maxAccel - Maximum allowed acceleration for cell points, in space units per
 *   frame squared.
 * cellPts - Array containing the cell center points, grouped by grid box.
 *   Updated positions will be stored here.
 * cellVels - Array containing the cell point velocities, grouped by grid box.
 *   Updated velocities will be stored here. Units are space units per frame.
 */
__kernel void cellNoise3DAnimUpdate(uint seedBase, uint numBoxesH,
  uint numPtsPerBox, double maxSpeed, double maxAccel, __global double *cellPts,
  __global double *cellPtVels)
{
  size_t ptIdx = get_global_id(0);
  uint seed = seedBase ^ ptIdx;
  
  // Generate random acceleration.
  double3 accel = randVecWithMagnitude3D(&seed,
    randDoubleInRange(&seed, 0, maxAccel));
  
  // Compute new velocity and position. Since the time units are frames, we can
  // ignore delta time in the calculations.
  double3 newVel = vload3(ptIdx, cellPtVels) + accel;
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
