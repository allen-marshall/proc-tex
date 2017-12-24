import cv2
import numpy
import pyopencl

from proc_tex.OpenCLSpherePerlinNoise3D import OpenCLSpherePerlinNoise3D
import proc_tex.texture_transforms
from proc_tex.texture_transforms import UnaryTransformedTexture

if __name__ == '__main__':
  cl_context = pyopencl.create_some_context()
  texture = UnaryTransformedTexture(
    lambda src_vals: proc_tex.texture_transforms.tex_to_dtype(
      proc_tex.texture_transforms.tex_scale_to_region(src_vals), numpy.uint16,
      scale=65535),
    OpenCLSpherePerlinNoise3D(cl_context, 40))
  eval_pts = texture.gen_eval_pts((1024, 1024), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  cv2.imwrite('./example.png', image)
  
  # texture.to_video(None, None, 120, 30, './example.webm', pix_fmt='gray16le',
  #   codec_params=['-lossless', '0'], eval_pts=eval_pts)
