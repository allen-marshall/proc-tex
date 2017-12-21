import cv2
import numpy
import pyopencl

from proc_tex.OpenCLCellNoise2D import OpenCLCellNoise2D

if __name__ == '__main__':
  cl_context = pyopencl.create_some_context()
  texture = OpenCLCellNoise2D(cl_context, 4, 4, 1)
  eval_pts = texture.gen_eval_pts((1024, 1024), numpy.array([[0,1], [0,1]]))
  image = texture.to_image(None, None, eval_pts=eval_pts)
  cv2.imshow('image', image)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
