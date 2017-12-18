import math

class Rectangle:
  """Simple representation of a rectangle."""
  def __init__(self, left, top, right, bottom):
    self.left = left
    self.right = right
    self.top = top
    self.bottom = bottom

def convert_pillow_rect(rect):
  """Converts a rectangle from the Pillow API into a Rectangle object."""
  return Rectangle(rect[0], rect[1], rect[2], rect[3])
