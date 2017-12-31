import numpy
import pyopencl

from proc_tex.texture_base import TransformedTexture

def tex_3d_to_sphere_map(src, cl_context, radius=numpy.float64(0.25),
  center=numpy.array((0, 0, 0), dtype=numpy.float64)):
  """Converts a 3D texture to a 2D sphere-mapped texture.
  src - 3D source texture to convert.
  cl_context - OpenCL context for the computation.
  radius - Radius of the sphere, in the source texture's texture space.
  center - Center of the sphere, in the source texture's texture space.
  Returns: The transformed texture."""
  
  # Precompile the OpenCl program.
  with open('opencl/sphereMap.cl', 'r', encoding='utf-8') as program_file:
    cl_program_map = pyopencl.Program(cl_context, program_file.read()) \
      .build(options=['-I', 'opencl/include/'])
  
  def space_transform(eval_pts):
    # Make sure eval_pts has the required memory layout.
    eval_pts = numpy.ascontiguousarray(eval_pts).astype(numpy.float64)
    
    # Set up OpenCL buffers.
    eval_pts_buffer = pyopencl.Buffer(cl_context,
      pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
      hostbuf=eval_pts)
    result_shape = eval_pts.shape[:-1] + (3,)
    result_array = numpy.empty(result_shape, dtype=numpy.float64)
    result_size_bytes = result_array.nbytes
    result_buffer = pyopencl.Buffer(cl_context, pyopencl.mem_flags.WRITE_ONLY,
      result_size_bytes)
    
    with pyopencl.CommandQueue(cl_context) as cl_queue:
      cl_program_map.sphereMapTo3D(cl_queue, (result_array.size // 3,),
        None, radius, center, eval_pts_buffer, result_buffer)
      
      pyopencl.enqueue_copy(cl_queue, result_array, result_buffer)
    
    return [result_array]
  
  def tex_transform(src_vals):
    return src_vals[0]
  
  return TransformedTexture(src.num_channels, 2, [src], space_transform,
    tex_transform)
