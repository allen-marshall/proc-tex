import math

def make_vec2d(x, y):
  """Makes a 2D vector from its Cartesian coordinates."""
  return [x, y]

def vec2d_from_magnitude_direction(magnitude, direction):
  """Converts a 2D vector from magnitude/direction to Cartesian coordinates."""
  vec_x = magnitude * math.cos(direction)
  vec_y = magnitude * math.sin(direction)
  return make_vec2d(vec_x, vec_y)

def add(vec0, vec1):
  """Adds two 2D vectors."""
  return make_vec2d(vec0[0] + vec1[0], vec0[1] + vec1[1])

def scale(scalar, vec):
  """Scales a 2D vector by a scalar."""
  return make_vec2d(scalar * vec[0], scalar * vec[1])

def magnitude(vec):
  """Returns the L2 norm of a 2D vector."""
  return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])