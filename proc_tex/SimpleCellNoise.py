import random

from proc_tex.texture_base import TimeSpaceTexture2D

_SPACE_BOTTOM = 0
_SPACE_TOP = 1
_SPACE_LEFT = 0
_SPACE_RIGHT = 1
_SPACE_WIDTH = _SPACE_RIGHT - _SPACE_LEFT
_SPACE_HEIGHT = _SPACE_TOP - _SPACE_BOTTOM
_NUM_POINTS = 20
_IMAGE_MODE = 'F'

class SimpleCellNoise(TimeSpaceTexture2D):
  """A simple cellular noise generator."""
  def __init__(self):
    super(SimpleCellNoise, self).__init__(_IMAGE_MODE)
    
    # Generate some random points.
    self.points = set()
    for idx in range(_NUM_POINTS):
      point_x = random.uniform(_SPACE_LEFT, _SPACE_RIGHT)
      point_y = random.uniform(_SPACE_BOTTOM, _SPACE_TOP)
      self.points.add((point_x, point_y))
  
  def evaluate(self, x, y):
    # Compute distance metric to the closest point.
    # TODO: Improve efficiency here.
    min_dist = None
    for point in self.points:
      new_dist = self.dist_metric_to_point(x, y, point[0], point[1])
      if (min_dist is None) or (new_dist < min_dist):
        min_dist = new_dist
    
    # TODO: Implement a better way of scaling the numbers to a visible range.
    return min_dist * 3000
  
  def dist_metric_to_point(self, pixel_x, pixel_y, point_x, point_y):
    delta_x = point_x - pixel_x
    delta_y = point_y - pixel_y
    
    # Take spatial looping into account so we can generate a seamless repeating
    # texture.
    delta_x %= _SPACE_WIDTH
    delta_x = min(abs(delta_x), _SPACE_WIDTH - abs(delta_x))
    delta_y %= _SPACE_HEIGHT
    delta_y = min(abs(delta_y), _SPACE_HEIGHT - abs(delta_y))
    
    # Use squared L2-norm as the distance metric.
    return delta_x * delta_x + delta_y * delta_y
