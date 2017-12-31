import random

import cv2
import numpy
import pyopencl

from proc_tex.texture_base import ScalarConstantTexture
from proc_tex.OpenCLCellNoise3D import OpenCLCellNoise3D
from proc_tex.OpenCLGridNoise3D import OpenCLGridNoise3D
from proc_tex.OpenCLPerlinNoise3D import OpenCLPerlinNoise3D
from proc_tex.texture_transforms import tex_scale_to_region, tex_to_dtype
from proc_tex.texture_transforms_opencl import tex_3d_to_sphere_map

if __name__ == '__main__':
  random.seed(234)
  numpy.random.seed(234)
  
  cl_context = pyopencl.create_some_context()
  
  # Combine cellular noise textures.
  texture = ScalarConstantTexture(1, 2, 0)
  cell_noise_params = [(5, 1, 1), (5, 1, -1), (8, 1, 0.5), (8, 1, -0.5), (10, 1, 0.25), (10, 1, -0.25), (12, 1, 0.125), (12, 1, -0.125)]
  for params in cell_noise_params:
    cell_noise = tex_3d_to_sphere_map(
      OpenCLCellNoise3D(cl_context, params[0], params[1]), cl_context)
    texture += params[2] * tex_scale_to_region(cell_noise, -0.5, 0.5)
  
  # Combine Perlin noise textures.
  perlin_noise_params = [(200, 0.05), (100, 0.02)]
  for params in perlin_noise_params:
    perlin_noise = tex_3d_to_sphere_map(
      OpenCLPerlinNoise3D(cl_context, params[0]), cl_context)
    texture += params[1] * tex_scale_to_region(perlin_noise, -0.5, 0.5)
  
  # Combine grid noise textures.
  grid_noise_params = [(2000, 0.01)]
  for params in grid_noise_params:
    grid_noise = tex_3d_to_sphere_map(OpenCLGridNoise3D(cl_context, params[0]),
      cl_context)
    texture += params[1] * tex_scale_to_region(grid_noise, -0.5, 0.5)
  
  texture = tex_to_dtype(tex_scale_to_region(texture), numpy.uint16,
    scale=65535)
  eval_pts = texture.gen_eval_pts((2048, 2048), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  cv2.imwrite('./example.png', image)
  
  # texture.to_video(None, None, 120, 30, './example.webm', pix_fmt='gray16le',
  #   codec_params=['-lossless', '1'], eval_pts=eval_pts)
