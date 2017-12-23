import math

def vec2d_from_circular(magnitude, direction):
  """Converts a 2D vector from magnitude/direction to Cartesian coordinates."""
  vec_x = magnitude * math.cos(direction)
  vec_y = magnitude * math.sin(direction)
  return [vec_x, vec_y]
