import random

import cv2
import numpy
import pyopencl

from proc_tex.texture_base import ScalarConstantTexture
from proc_tex.OpenCLSphereCellNoise3D import OpenCLSphereCellNoise3D
from proc_tex.OpenCLSpherePerlinNoise3D import OpenCLSpherePerlinNoise3D
from proc_tex.texture_transforms import tex_scale_to_region, tex_to_dtype

if __name__ == '__main__':
  random.seed(234)
  numpy.random.seed(234)
  
  cl_context = pyopencl.create_some_context()
  
  # Combine cellular noise textures.
  texture = ScalarConstantTexture(1, numpy.float64, 2, 0)
  cell_noise_params = [(5, 1, -1), (8, 1, 0.5), (10, 1, -0.25), (12, 1, 0.125)]
  for params in cell_noise_params:
    cell_noise = OpenCLSphereCellNoise3D(cl_context, params[0], params[1])
    texture += params[2] * tex_scale_to_region(cell_noise)
  
  # Combine Perlin noise textures.
  perlin_noise_params = [(50, 0.02), (200, -0.05)]
  for params in perlin_noise_params:
    perlin_noise = OpenCLSpherePerlinNoise3D(cl_context, params[0])
    texture += params[1] * tex_scale_to_region(perlin_noise)
  
  texture = tex_to_dtype(tex_scale_to_region(texture), numpy.uint16,
    scale=65535)
  eval_pts = texture.gen_eval_pts((2048, 2048), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  cv2.imwrite('./example.png', image)
  
  # texture.to_video(None, None, 120, 30, './example.webm', pix_fmt='gray16le',
  #   codec_params=['-lossless', '0'], eval_pts=eval_pts)
