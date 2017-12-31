import cv2
import numpy
import pyopencl

from proc_tex.OpenCLPerlinNoise3D import OpenCLPerlinNoise3D
from proc_tex.texture_transforms import tex_scale_to_region, tex_to_dtype
from proc_tex.texture_transforms_opencl import tex_3d_to_sphere_map

if __name__ == '__main__':
  cl_context = pyopencl.create_some_context()
  texture = tex_3d_to_sphere_map(OpenCLPerlinNoise3D(cl_context, 40),
    cl_context)
  texture = tex_to_dtype(tex_scale_to_region(texture), numpy.uint16,
    scale=65535)
  eval_pts = texture.gen_eval_pts((1024, 1024), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  cv2.imwrite('./example.png', image)
  
  # texture.to_video(None, None, 120, 30, './example.webm', pix_fmt='gray16le',
  #   codec_params=['-lossless', '0'], eval_pts=eval_pts)
