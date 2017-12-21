import cv2
import numpy
import pyopencl

from proc_tex.OpenCLCellNoise2D import OpenCLCellNoise2D

if __name__ == '__main__':
  cl_context = pyopencl.create_some_context()
  texture = OpenCLCellNoise2D(cl_context, 4, 4, 1)
  image = texture.to_image((1024, 1024), numpy.array([[0,2], [0,2]]))
  cv2.imshow('image', image)
  cv2.waitKey(0)
  
  # texture.to_video(1024, 1024, proc_tex.geom.Rectangle(0, 2, 2, 0), 120, 30, './example.avi')
  
  cv2.destroyAllWindows()
