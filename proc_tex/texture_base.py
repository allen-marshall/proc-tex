import PIL.Image

class TimeSpaceTexture2D:
  """Base class for 2D animated textures.
  Textures of this type support two dimensions of space and one of time. This
  class can also be used for non-animated textures by simply not overriding
  step_frame."""
  def __init__(self, mode):
    """Initializes the Pillow image mode.
    mode is a Pillow image mode that specifies the channels in the texture."""
    self.mode = mode
  
  def evaluate(self, x, y):
    """Returns the pixel value for the current frame at the specified location.
    The format of the pixel value should match the subclass's Pillow mode.
    Subclasses should typically override this. Subclasses are not required to
    return a consistent value for multiple evaluations of the same point (this
    helps with the implementation of certain types of noise). Therefore, callers
    that need consistency should cache the value if it is needed more than once.
    Default implementation returns zero."""
    return 0
  
  def step_frame(self):
    """Moves to the next frame.
    This can alter the values returned by evaluate. Subclasses that require
    animation should typically override this."""
    pass
  
  def to_image(self, width, height, space_bounds):
    """Generates a Pillow image of the current frame.
    width and height specify the number of pixels. space_bounds should be a Pillow
    rectangle specifying the spatial coordinates to use for evaluating. Note that
    the rectangle should assume positive y is up, not down."""
    space_left = space_bounds[0]
    space_right = space_bounds[2]
    space_bottom = space_bounds[3]
    space_top = space_bounds[1]
    space_width = space_right - space_left
    space_height = space_top - space_bottom
    if space_width <= 0 or space_height <= 0:
      raise ValueError("Nonpositive space width or height")
      
    image = PIL.Image.new(self.mode, (width, height))
    
    # Evaluate the texture for each pixel.
    for x in range(width):
      for y in range(height):
        x_factor = (x + 0.5) / width
        y_factor = (y + 0.5) / height
        space_x = (1 - x_factor) * space_left + x_factor * space_right
        space_y = (1 - y_factor) * space_bottom + y_factor * space_top
        pixel = self.evaluate(space_x, space_y)
        image.putpixel((x, height-(1+y)), pixel)
    
    return image
  
  # TODO: Write a method for making a video from multiple frames.
