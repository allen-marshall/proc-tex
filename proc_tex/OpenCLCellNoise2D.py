import math
import random
import time

import numpy
import pyopencl

from proc_tex.texture_base import TimeSpaceTexture
import proc_tex.geom
import proc_tex.vec_2d
import proc_tex.vec_3d

_NUM_CHANNELS = 1
_DTYPE = numpy.float64
_NUM_SPACE_DIMS = 2

# TODO: Avoid duplicating this in multiple files.
class DistanceMetrics:
  METRIC_L2_NORM = 0
  METRIC_L2_NORM_SQUARED = 1
  METRIC_DEFAULT = METRIC_L2_NORM

class OpenCLCellNoise2D(TimeSpaceTexture):
  """Computes 2D cellular noise.
  Uses a modified version of Worley's grid-based cellular noise algorithm.
  Animation causes the cell points to move randomly."""
  def __init__(self, cl_context, num_boxes_h, pts_per_box,
    metric = DistanceMetrics.METRIC_DEFAULT, point_max_speed=0.01,
    point_max_accel=0.005):
    """Initializer.
    cl_context - The PyOpenCL context to use for computation.
    num_boxes_h - The width and height (both the same) of the grid, in number of
      grid boxes. Should be at least 1.
    pts_per_box - The number of cell points per grid box. Should be at least 1.
    metric - One of the constants from DistanceMetrics that specifies the
      distance metric to use.
    point_max_speed - Maximum point speed, in space units per frame.
    point_max_accel - Maximum point acceleration, in space units per frame
      squared."""
    super(OpenCLCellNoise2D, self).__init__(_NUM_CHANNELS, _DTYPE,
      _NUM_SPACE_DIMS)
    
    if pts_per_box <= 0:
      raise ValueError("Must have at least one point per grid box.")
    
    self.cl_context = cl_context
    self.num_boxes_h = num_boxes_h
    self.box_width = 1 / num_boxes_h
    self.pts_per_box = pts_per_box
    self.metric = metric
    self.point_max_speed = point_max_speed
    self.point_max_accel = point_max_accel
    
    # Precompile the OpenCL programs.
    with open('opencl/cellNoise2D.cl', 'r', encoding='utf-8') as program_file:
      self.cl_program_noise = pyopencl.Program(self.cl_context, program_file.read()) \
        .build(options=['-I', 'opencl/include/'])
    with open('opencl/cellNoise2DAnim.cl', 'r', encoding='utf-8') as program_file:
      self.cl_program_anim = pyopencl.Program(self.cl_context, program_file.read()) \
        .build(options=['-I', 'opencl/include/'])
    
    # Generate the Numpy array of cell points.
    num_grid_boxes = num_boxes_h * num_boxes_h
    self.cell_pts = numpy.empty((num_grid_boxes * pts_per_box, 2),
      dtype=numpy.float64)
    self.cell_vels = numpy.empty_like(self.cell_pts)
    for box_x in range(num_boxes_h):
      for box_y in range(num_boxes_h):
        box_bounds = self._grid_coords_to_bounds((box_x, box_y))
        for box_pt_idx in range(pts_per_box):
          pt_idx = (box_y * num_boxes_h + box_x) * pts_per_box + box_pt_idx
          init_vel_mag = random.uniform(0, point_max_speed)
          init_vel_dir = random.uniform(0, math.tau)
          init_vel = proc_tex.vec_2d.vec2d_from_circular(init_vel_mag,
            init_vel_dir)
          init_x = random.uniform(box_bounds[0,0], box_bounds[0,1])
          init_y = random.uniform(box_bounds[1,0], box_bounds[1,1])
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
    
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      self.cl_program_noise.cellNoise2D(cl_queue, (result_array.size,), None,
        numpy.uint32(self.num_boxes_h), numpy.uint32(self.pts_per_box),
        numpy.uint32(self.metric), cell_pts_buffer, eval_pts_buffer,
        result_buffer)
      
      pyopencl.enqueue_copy(cl_queue, result_array, result_buffer)
    
    return result_array
    
  
  def step_frame(self):
    # Compute a random acceleration for each cell point.
    cell_accels = numpy.random.sample(self.cell_pts.shape)
    cell_accels[:,0] *= self.point_max_accel
    cell_accels[:,1] *= math.tau
    
    # Create buffers for the OpenCL kernels.
    cell_pts_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.cell_pts)
    cell_vels_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.cell_vels)
    cell_accels_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=cell_accels)
    
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      self.cl_program_anim.cellNoise2DAnimUpdate(cl_queue,
        (self.cell_pts.shape[0],), None, numpy.uint32(self.num_boxes_h),
        numpy.uint32(self.pts_per_box), numpy.float64(self.point_max_speed),
        cell_pts_buffer, cell_vels_buffer, cell_accels_buffer)
      
      pyopencl.enqueue_copy(cl_queue, self.cell_pts, cell_pts_buffer)
      pyopencl.enqueue_copy(cl_queue, self.cell_vels, cell_vels_buffer)
  
  def _grid_coords_to_bounds(self, coords):
    # Assumes the grid coordinates are inside the base cube.
    left = coords[0] * self.box_width
    right = left + self.box_width
    bottom = coords[1] * self.box_width
    top = bottom + self.box_width
    return numpy.array(((left, right), (bottom, top)))
