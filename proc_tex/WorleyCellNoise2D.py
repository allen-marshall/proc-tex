import math
import random

from proc_tex.texture_base import TimeSpaceTexture2D
import proc_tex.geom
import proc_tex.vec_2d

_IMAGE_MODE = 'F'

def _default_metric(delta_x, delta_y):
  return delta_x * delta_x + delta_y * delta_y

class _CellPoint:
  def __init__(self, pos, vel):
    self.pos = pos
    self.vel = vel

class _GridSquare:
  def __init__(self, bounds, points):
    self.bounds = bounds
    self.points = set(points)

class WorleyCellNoise2D(TimeSpaceTexture2D):
  """A cellular noise generator based on Worley's grid-based optimization.
  Animation causes the cell points to move randomly."""
  def __init__(self, num_grid_columns, num_grid_rows, min_pts_per_square,
    max_pts_per_square, metric = _default_metric,
    point_max_speed=0.1, point_max_accel=0.1):
    """Initializer.
    metric should be a function that takes an (absolute) delta x and delta y and
    returns the distance metric for a point that is that displacement from its
    nearest cell point. The metric is assumed to be monotonically nondecreasing
    with respect to both delta x and delta y, and to be isotropic (switching
    delta x and delta y should make no difference).
    point_max_speed is in space units per frame. point_max_accel is in space
    units per frame squared."""
    super(WorleyCellNoise2D, self).__init__(_IMAGE_MODE)
    
    self.num_grid_columns = num_grid_columns
    self.num_grid_rows = num_grid_rows
    self.square_width = 1 / num_grid_columns
    self.square_height = 1 / num_grid_rows
    self.metric = metric
    self.point_max_speed = point_max_speed
    self.point_max_accel = point_max_accel
    
    if min_pts_per_square <= 0:
      raise ValueError("Must have at least one point per square.")
    
    # Generate the grid squares and cell points.
    num_grid_squares = num_grid_columns * num_grid_rows
    self.grid_squares = []
    for square_x in range(num_grid_columns):
      self.grid_squares.append([None] * num_grid_rows)
      for square_y in range(num_grid_rows):
        square_bounds = self._grid_coords_to_bounds((square_x, square_y))
        num_pts = random.randint(min_pts_per_square, max_pts_per_square)
        pts_set = set()
        for pt_idx in range(num_pts):
          init_vel_mag = random.uniform(0, point_max_speed)
          init_vel_dir = random.uniform(0, math.tau)
          init_vel = proc_tex.vec_2d.vec2d_from_magnitude_direction(
            init_vel_mag, init_vel_dir)
          init_x = random.uniform(square_bounds.left, square_bounds.right)
          init_y = random.uniform(square_bounds.bottom, square_bounds.top)
          init_pos = proc_tex.vec_2d.make_vec2d(init_x, init_y)
          point = _CellPoint(init_pos, init_vel)
          pts_set.add(point)
        square = _GridSquare(square_bounds, pts_set)
        self.grid_squares[square_x][square_y] = square
  
  def evaluate(self, x, y):
    # Convert the point to its corresponding point in the base square
    # (0, 1, 1, 0). This allows calculations to assume the position is in the
    # base square, which improves efficiency.
    pos = self._pos_to_base_square(proc_tex.vec_2d.make_vec2d(x, y))
    
    # Compute distance metric to the closest point.
    min_dist = None
    for square in self._nearby_squares(pos):
      for point in square.points:
        new_dist = self._metric_from_cell_point(pos, point)
        if (min_dist is None) or (new_dist < min_dist):
          min_dist = new_dist
    
    return min_dist
  
  def step_frame(self):
    for square in self.grid_squares:
      for point in square.points:
        # Choose random acceleration.
        accel_mag = random.uniform(0, self.point_max_accel)
        accel_dir = random(0, math.tau)
        accel = proc_tex.vec_2d.vec2d_from_magnitude_direction(accel_mag,
          accel_dir)
        
        # Compute new velocity.
        point.vel = proc_tex.vec_2d.add(accel, point.vel)
        speed = numpy.norm(point.vel)
        if speed > self.point_max_speed:
          point.vel = proc_tex.vec_2d.scale(self.point_max_speed / speed,
            point.vel)
        
        # Compute new position.
        point.pos = proc_tex.vec_2d.add(point.vel, point.pos)
        
        # Clamp position and velocity based on the square boundary.
        if point.pos[0] < square.bounds.left:
          point.pos[0] = square.bounds.left
          point.vel[0] = max(0, point.vel[0])
        if point.pos[0] > square.bounds.right:
          point.pos[0] = square.bounds.right
          point.vel[0] = min(0, point.vel[0])
        if point.pos[1] < square.bounds.bottom:
          point.pos[1] = square.bounds.bottom
          point.vel[1] = max(0, point.vel[1])
        if point.pos[1] > square.bounds.top:
          point.pos[1] = square.bounds.top
          point.vel[1] = min(0, point.vel[1])
  
  def _pos_to_base_square(self, pos):
    # Converts a point into the base square (0, 1, 1, 0), taking spatial looping
    # into account.
    return proc_tex.vec_2d.make_vec2d(pos[0] % 1, pos[1] % 1)
  
  def _grid_coords_to_base_square(self, grid_coords):
    # Assumes the grid coordinates are at most one unit outside the base square
    # in x and/or y.
    new_x = grid_coords[0]
    new_y = grid_coords[1]
    
    if new_x < 0:
      new_x += self.num_grid_columns
    elif new_x >= self.num_grid_columns:
      new_x -= self.num_grid_columns
    if new_y < 0:
      new_y += self.num_grid_rows
    elif new_y >= self.num_grid_rows:
      new_y -= self.num_grid_rows
    
    return (new_x, new_y)
  
  def _grid_coords_to_bounds(self, coords):
    # Assumes the grid coordinates are inside the base square.
    left = coords[0] * self.square_width
    right = left + self.square_width
    bottom = coords[1] * self.square_height
    top = bottom + self.square_height
    return proc_tex.geom.Rectangle(left, top, right, bottom)
  
  def _grid_coords_to_grid_square(self, grid_coords):
    actual_coords = self._grid_coords_to_base_square(grid_coords)
    return self.grid_squares[actual_coords[0]][actual_coords[1]]
  
  def _position_to_grid_square_coords(self, pos):
    grid_x = int(round((pos[0] // self.square_width) % self.num_grid_columns))
    grid_y = int(round((pos[1] // self.square_height) % self.num_grid_columns))
    return (grid_x, grid_y)
  
  def _nearby_squares(self, pos):
    center = self._position_to_grid_square_coords(pos)
    other_coords = [(center[0] + 1, center[1]), (center[0] - 1, center[1]),
      (center[0], center[1] + 1), (center[0], center[1] - 1),
      (center[0] + 1, center[1] + 1), (center[0] + 1, center[1] - 1),
      (center[0] - 1, center[1] + 1), (center[0] - 1, center[1] - 1)]
    return [self.grid_squares[center[0]][center[1]]] + [self._grid_coords_to_grid_square(coords) for coords in other_coords]
  
  def _looping_delta(self, coord0, coord1):
    # Computes absolute difference in x or y, taking spatial looping into
    # account. Assumes both coordinates are inside the base square
    # (0, 1, 1, 0).
    delta = abs(coord0 - coord1)
    if delta <= 0.5:
      return delta
    else:
      return 1 - delta
  
  def _metric_from_cell_point(self, eval_pos, cell_pt):
    # Assumes eval_pos is inside the base square (0, 1, 1, 0).
    delta_x = self._looping_delta(eval_pos[0], cell_pt.pos[0])
    delta_y = self._looping_delta(eval_pos[1], cell_pt.pos[1])
    
    return self.metric(delta_x, delta_y)
