#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "random.clh"

/*
 * Randomly initializes gradients for 3D Perlin noise.
 * seedBase - Random seed. This will be combined with the worker ID to generate
 *   a separate seed for each worker.
 * gradients - Initial gradients will be stored here.
 */
__kernel void perlinNoise3DAnimInit(uint seedBase, __global double *gradients) {
  size_t gradientIdx = get_global_id(0);
  uint seed = initWorkerSeed(seedBase, gradientIdx);
  
  // Generate random initial gradient.
  double3 gradient = randVecWithMagnitude3D(&seed, 1);
  vstore3(gradient, gradientIdx, gradients);
}

/*
 * Updates gradients for 3D Perlin noise.
 * seedBase - Random seed. This will be combined with the worker ID to generate
 *   a separate seed for each worker.
 * gradients - Array of gradients. Will be updated with the new gradients.
 */
__kernel void perlinNoise3DAnimUpdate(uint seedBase, __global double *gradients)
{
  // TODO: The new gradients should be partly based on the old gradients.
  
  size_t gradientIdx = get_global_id(0);
  uint seed = initWorkerSeed(seedBase, gradientIdx);
  
  // Compute the new gradient.
  double3 gradient = randVecWithMagnitude3D(&seed, 1);
  vstore3(gradient, gradientIdx, gradients);
}
