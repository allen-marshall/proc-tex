#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*
 * Normalizes a coordinate into the unit (hyper-)square with one corner at the
 * origin and all other coordinates positive. (e.g. the unit square centered at
 * (0.5, 0.5) for 2 dimensions)
 */
double normalizeTexCoord(double coord) {
  // Convert to nonnegative number with same modulus so that fmod's rounding
  // toward zero gives us what we want.
  if (coord < 0) {
    coord -= trunc(coord) - 1;
  }
  return fmod(coord, 1);
}

/* Normalizes 2D coordinates using normalizeTexCoord. */
void normalizeTexPt2D(double2 *pt) {
  pt->x = normalizeTexCoord(pt->x);
  pt->y = normalizeTexCoord(pt->y);
}

/* Normalizes 3D coordinates using normalizeTexCoord. */
void normalizeTexPt3D(double3 *pt) {
  pt->x = normalizeTexCoord(pt->x);
  pt->y = normalizeTexCoord(pt->y);
  pt->z = normalizeTexCoord(pt->z);
}

/*
 * Computes the absolute delta x and delta y between two (normalized) 2D points,
 * assuming spatial looping in the base square (useful for generating seamless
 * textures).
 */
double2 loopingDelta2D(double2 pt0, double2 pt1) {
  double2 result = fabs(pt1 - pt0);
  result = min(result, 1 - result);
  return result;
}

/*
 * Computes the absolute delta x, delta y, and delta z between two (normalized)
 * 3D points, assuming spatial looping in the base cube (useful for generating
 * seamless textures).
 */
double3 loopingDelta3D(double3 pt0, double3 pt1) {
  double3 result = fabs(pt1 - pt0);
  result = min(result, 1 - result);
  return result;
}

/*
 * Converts a (normalized) texture point for a sphere-mapped texture into
 * Cartesian coordinates.
 */
double3 sphericalToCartesian(double2 pt, double radius) {
  double yaw = (pt.x - 0.5) * 2.0 * M_PI;
  double pitch = pt.y * M_PI;
  double sinPitch = sin(pitch);
  return (double3) (cos(yaw) * sinPitch, sin(yaw) * sinPitch, cos(pitch))
    * radius;
}
