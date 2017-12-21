import math
import random
import time

import numpy
import pyopencl

from proc_tex.texture_base import TimeSpaceTexture2D
import proc_tex.geom
import proc_tex.vec_2d

_NUM_CHANNELS = 1
_DTYPE = numpy.float64
_NUM_SPACE_DIMS = 2

class DistanceMetrics:
  METRIC_L2_NORM = 0
  METRIC_L2_NORM_SQUARED = 1
  METRIC_DEFAULT = METRIC_L2_NORM

class OpenCLCellNoise2D(TimeSpaceTexture2D):
  """A cellular noise generator based on Worley's grid-based optimization.
  Animation causes the cell points to move randomly."""
  def __init__(self, cl_context, num_grid_rows, num_grid_cols, pts_per_square,
    metric = DistanceMetrics.METRIC_DEFAULT, point_max_speed=0.01,
    point_max_accel=0.005):
    """Initializer.
    cl_context - The PyOpenCL context to use for computation.
    num_grid_rows - The number of rows in the grid. Should be at least 1.
    num_grid_cols - The number of columns in the grid. Should be at least 1.
    pts_per_square - The number of cell points per grid square. Should be at
      least 1.
    metric - One of the constants from DistanceMetrics that specifies the
      distance metric to use.
    point_max_speed - Maximum point speed, in space units per frame.
    point_max_accel - Maximum point acceleration, in space units per frame
      squared."""
    super(OpenCLCellNoise2D, self).__init__(_NUM_CHANNELS, _DTYPE,
      _NUM_SPACE_DIMS)
    
    if pts_per_square <= 0:
      raise ValueError("Must have at least one point per grid square.")
    
    self.cl_context = cl_context
    self.num_grid_cols = num_grid_cols
    self.num_grid_rows = num_grid_rows
    self.square_width = 1 / num_grid_cols
    self.square_height = 1 / num_grid_rows
    self.pts_per_square = pts_per_square
    self.metric = metric
    self.point_max_speed = point_max_speed
    self.point_max_accel = point_max_accel
    
    # Generate the Numpy array of cell points.
    num_grid_squares = num_grid_cols * num_grid_rows
    self.cell_pts = numpy.empty((num_grid_squares * pts_per_square, 2),
      dtype=numpy.float64)
    self.cell_vels = numpy.empty_like(self.cell_pts)
    for square_x in range(num_grid_rows):
      for square_y in range(num_grid_cols):
        square_bounds = self._grid_coords_to_bounds((square_x, square_y))
        for square_pt_idx in range(pts_per_square):
          pt_idx = (square_y * num_grid_rows + square_x) * pts_per_square + square_pt_idx
          init_vel_mag = random.uniform(0, point_max_speed)
          init_vel_dir = random.uniform(0, math.tau)
          init_vel = proc_tex.vec_2d.vec2d_from_magnitude_direction(
            init_vel_mag, init_vel_dir)
          init_x = random.uniform(square_bounds.left, square_bounds.right)
          init_y = random.uniform(square_bounds.bottom, square_bounds.top)
          self.cell_pts[pt_idx] = (init_x, init_y)
          self.cell_vels[pt_idx] = init_vel
  
  def evaluate(self, eval_pts):
    # TODO: Figure out how to make this work with multiple devices
    # simultaneously. Might require splitting up the tasks.
    
    # Create Numpy array for the results.
    result_shape = eval_pts.shape[:-1] + (_NUM_CHANNELS,)
    result_array = numpy.empty(result_shape, dtype=_DTYPE)
    result_size_bytes = result_array.nbytes
    
    # Make sure eval_pts has the required memory layout.
    eval_pts = numpy.ascontiguousarray(eval_pts)
    
    # Create buffers for the OpenCL kernels.
    eval_pts_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=eval_pts)
    cell_pts_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.cell_pts)
    result_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.WRITE_ONLY, result_size_bytes)
    
    with open('opencl/cellNoise2D.cl', 'r', encoding='utf-8') as program_file:
      cl_program = pyopencl.Program(self.cl_context, program_file.read()) \
        .build()
    
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      cl_program.cellNoise2D(cl_queue, (result_array.size,), None,
        numpy.uint32(self.num_grid_cols), numpy.uint32(self.num_grid_rows),
        numpy.uint32(self.pts_per_square), numpy.uint32(self.metric),
        cell_pts_buffer, eval_pts_buffer, result_buffer)
      
      pyopencl.enqueue_copy(cl_queue, result_array, result_buffer)
    
    return result_array
    
  
  def step_frame(self):
    # TODO: Implement this.
    pass
  
  def _grid_coords_to_bounds(self, coords):
    # Assumes the grid coordinates are inside the base square.
    left = coords[0] * self.square_width
    right = left + self.square_width
    bottom = coords[1] * self.square_height
    top = bottom + self.square_height
    return proc_tex.geom.Rectangle(left, top, right, bottom)
