import random

import cv2
import numpy
import pyopencl

from proc_tex.texture_base import ScalarConstantTexture
from proc_tex.OpenCLSphereCellNoise3D import OpenCLSphereCellNoise3D
from proc_tex.OpenCLSphereGridNoise3D import OpenCLSphereGridNoise3D
from proc_tex.OpenCLSpherePerlinNoise3D import OpenCLSpherePerlinNoise3D
from proc_tex.texture_transforms import tex_scale_to_region, tex_to_dtype

if __name__ == '__main__':
  random.seed(234)
  numpy.random.seed(234)
  
  cl_context = pyopencl.create_some_context()
  
  # Combine cellular noise textures.
  texture = ScalarConstantTexture(1, 2, 0)
  cell_noise_params = [(5, 1, 1), (5, 1, -1), (8, 1, 0.5), (8, 1, -0.5), (10, 1, 0.25), (10, 1, -0.25), (12, 1, 0.125), (12, 1, -0.125)]
  for params in cell_noise_params:
    cell_noise = OpenCLSphereCellNoise3D(cl_context, params[0], params[1])
    texture += params[2] * tex_scale_to_region(cell_noise, -0.5, 0.5)
  
  # Combine Perlin noise textures.
  perlin_noise_params = [(200, 0.05), (100, 0.02)]
  for params in perlin_noise_params:
    perlin_noise = OpenCLSpherePerlinNoise3D(cl_context, params[0])
    texture += params[1] * tex_scale_to_region(perlin_noise, -0.5, 0.5)
  
  # Combine grid noise textures.
  grid_noise_params = [(2000, 0.01)]
  for params in grid_noise_params:
    grid_noise = OpenCLSphereGridNoise3D(cl_context, params[0])
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
