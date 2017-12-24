import math
import random
import time

import numpy
import pyopencl

from proc_tex.texture_base import TimeSpaceTexture
import proc_tex.dist_metrics

_NUM_CHANNELS = 1
_DTYPE = numpy.float64
_NUM_SPACE_DIMS = 2

# TODO: Implement animation.

class OpenCLSpherePerlinNoise3D(TimeSpaceTexture):
  """Computes sphere-mapped 3D Perlin noise."""
  def __init__(self, cl_context, num_boxes_h):
    """Initializer.
    cl_context - The PyOpenCL context to use for computation.
    num_boxes_h - The width, height, and depth (all the same) of the grid, in
      number of grid boxes. Should be at least 1."""
    super(OpenCLSpherePerlinNoise3D, self).__init__(_NUM_CHANNELS, _DTYPE,
      _NUM_SPACE_DIMS)
    
    self.cl_context = cl_context
    self.num_boxes_h = num_boxes_h
    self.box_width = 1 / num_boxes_h
    
    # Precompile the OpenCL programs.
    with open('opencl/spherePerlinNoise3D.cl', 'r', encoding='utf-8') as program_file:
      self.cl_program_noise = pyopencl.Program(self.cl_context, program_file.read()) \
        .build(options=['-I', 'opencl/include/'])
    with open('opencl/perlinNoise3DAnim.cl', 'r', encoding='utf-8') as program_file:
      self.cl_program_anim = pyopencl.Program(self.cl_context, program_file.read()) \
        .build(options=['-I', 'opencl/include/'])
    
    # Generate the Numpy array of gradients.
    seed = random.randrange(0, 2 ** 32)
    num_grid_boxes = num_boxes_h * num_boxes_h * num_boxes_h
    self.gradients = numpy.empty((num_grid_boxes, 3), dtype=numpy.float64)
    gradients_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.gradients)
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      self.cl_program_anim.perlinNoise3DAnimInit(cl_queue,
        (self.gradients.shape[0],), None, numpy.uint32(seed), gradients_buffer)
      
      pyopencl.enqueue_copy(cl_queue, self.gradients, gradients_buffer)
  
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
    gradients_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.gradients)
    result_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.WRITE_ONLY, result_size_bytes)
    
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      self.cl_program_noise.spherePerlinNoise3D(cl_queue, (result_array.size,),
        None, numpy.uint32(self.num_boxes_h), gradients_buffer, eval_pts_buffer,
        result_buffer)
      
      pyopencl.enqueue_copy(cl_queue, result_array, result_buffer)
    
    return result_array
  
  def step_frame(self):
    seed = random.randrange(0, 2 ** 32)
    gradients_buffer = pyopencl.Buffer(self.cl_context,
      pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=self.gradients)
    with pyopencl.CommandQueue(self.cl_context) as cl_queue:
      self.cl_program_anim.perlinNoise3DAnimUpdate(cl_queue,
        (self.gradients.shape[0],), None, numpy.uint32(seed), gradients_buffer)
      
      pyopencl.enqueue_copy(cl_queue, self.gradients, gradients_buffer)
