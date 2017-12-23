import math

def vec3d_from_spherical(yaw, pitch, magnitude):
  """Converts a 3D vector from spherical to Cartesian coordinates."""
  sin_pitch = math.sin(pitch)
  vec_x = magnitude * math.cos(yaw) * sin_pitch
  vec_y = magnitude * math.sin(yaw) * sin_pitch
  vec_z = magnitude * math.cos(pitch)
  return [vec_x, vec_y, vec_z]
