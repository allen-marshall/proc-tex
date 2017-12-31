import random

import cv2
import numpy
import pyopencl

from proc_tex.texture_base import ScalarConstantTexture
from proc_tex.OpenCLCellNoise3D import OpenCLCellNoise3D
from proc_tex.OpenCLGridNoise3D import OpenCLGridNoise3D
from proc_tex.OpenCLPerlinNoise3D import OpenCLPerlinNoise3D
from proc_tex.texture_transforms import tex_concat_channels, tex_scale_to_region, tex_space_offset_by_texture, tex_to_dtype
from proc_tex.texture_transforms_opencl import tex_3d_to_sphere_map

if __name__ == '__main__':
  random.seed(345)
  numpy.random.seed(345)
  
  cl_context = pyopencl.create_some_context()
  
  # Create a noise-based offset texture.
  def make_offset_channel():
    return tex_scale_to_region(OpenCLPerlinNoise3D(cl_context, 10), -0.05,
      0.05)
  offset_noise = tex_concat_channels(
    [make_offset_channel(), make_offset_channel(), make_offset_channel()])
  
  # Combine cellular noise textures.
  cell_noise = ScalarConstantTexture(1, 3, 0)
  cell_noise_params = [(5, 1, 1), (4, 1, -1), (7, 1, 0.5), (6, 1, -0.5),
    (9, 1, 0.25), (8, 1, -0.25), (11, 1, 0.125), (10, 1, -0.125)]
  for params in cell_noise_params:
    new_cell_noise = OpenCLCellNoise3D(cl_context, params[0], params[1])
    cell_noise += params[2] * tex_scale_to_region(new_cell_noise, -0.5, 0.5)
  
  # Apply offset noise to cellular noise.
  warped_noise = tex_space_offset_by_texture(cell_noise, offset_noise)
  
  # Sphere-map the texture.
  sphere_mapped_noise = tex_3d_to_sphere_map(warped_noise, cl_context)
  
  # Make image.
  texture = tex_to_dtype(tex_scale_to_region(sphere_mapped_noise), numpy.uint16,
    scale=65535)
  eval_pts = texture.gen_eval_pts((2048, 2048), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  
  # Save image.
  cv2.imwrite('./example.png', image)
