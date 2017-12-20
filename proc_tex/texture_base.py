import numpy
import cv2

class TimeSpaceTexture2D:
  """Base class for 2D animated textures.
  Textures of this type support two dimensions of space and one of time. This
  class can also be used for non-animated textures by simply not overriding
  step_frame."""
  def __init__(self, num_channels, dtype):
    """Initializer.
    num_channels specifies the number of channels per pixel. dtype specifies the
    Numpy dtype to use for the data in each channel."""
    self.num_channels = num_channels
    self.dtype = dtype
  
  def evaluate(self, x, y):
    """Returns the pixel value for the current frame at the specified location.
    The pixel value should be an iterable of size self.num_channels. Subclasses
    should typically override this. Subclasses are not required to return a
    consistent value for multiple evaluations of the same point (this helps with
    the implementation of certain types of noise). Therefore, callers that need
    consistency should cache the value if it is needed more than once, or ensure
    that they only use subclasses with consistent behavior. Subclasses may
    change the interface of this method to allow for optimizations; subclasses
    that do so should also override to_image with custom logic to handle the
    changed interface. Default implementation returns zero for each channel."""
    return [0] * self.num_channels
  
  def step_frame(self):
    """Moves to the next frame.
    This can alter the values returned by evaluate. Subclasses that require
    animation should typically override this."""
    pass
  
  def to_image(self, width, height, space_bounds):
    """Generates an OpenCV image of the current frame.
    width and height specify the number of pixels. space_bounds should be a
    Rectangle specifying the spatial coordinates to use for evaluating. Note
    that the rectangle should assume positive y is up, not down."""
    space_width = space_bounds.right - space_bounds.left
    space_height = space_bounds.top - space_bounds.bottom
    if space_width <= 0 or space_height <= 0:
      raise ValueError('Nonpositive space width or height')
    
    image = numpy.zeros((height, width, self.num_channels), self.dtype)
    
    # Evaluate the texture for each pixel.
    for x in range(width):
      for y in range(height):
        x_factor = (x + 0.5) / width
        y_factor = (y + 0.5) / height
        space_x = (1 - x_factor) * space_bounds.left + x_factor * space_bounds.right
        space_y = (1 - y_factor) * space_bounds.bottom + y_factor * space_bounds.top
        pixel = self.evaluate(space_x, space_y)
        for channel_idx in range(len(pixel)):
          image[y][x][channel_idx] = pixel[channel_idx]
    
    return image
  
  def to_video(self, width, height, space_bounds, num_frames,
    frames_per_second, filename):
    """Generates an OpenCV video starting at the current frame.
    Arguments shared with to_image have similar meaning as there. This method
    has the side effect of moving the current frame forward by
    num_frames."""
    try:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      video = cv2.VideoWriter(filename, fourcc, frames_per_second,
        (width, height))
      for frame_idx in range(num_frames):
        frame = self.to_image(width, height, space_bounds)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        video.write(frame)
        print('Frame written')
        self.step_frame()
    finally:
      if video is not None:
        video.release()

class OpenCLTimeSpaceTexture2D(TimeSpaceTexture2D):
  """TimeSpaceTexture2D with OpenCL support.
  To support OpenCL, the evaluate method has a different interface than in
  TimeSpaceTexture2D."""
  def __init__(self, num_channels, dtype, cl_context):
    """Initializer.
    cl_context - The PyOpenCL context to use for OpenCL computations."""
    super(OpenCLTimeSpaceTexture2D, self).__init__(num_channels, dtype)
    self.cl_context = cl_context
  
  def evaluate(self, eval_pts):
    """Gets the pixel values for the current frame at the specified locations.
    Default implementation returns zero for each channel.
    eval_pts - Numpy array of points to evaluate. The last dimension size should
      be 2 since the points are 2D vectors."""
    return [0] * self.num_channels
  
  def to_image(self, width, height, space_bounds, eval_pts=None):
    """See superclass.
    eval_pts - Precomputed Numpy array of the points to evaluate. Should be
      consistent with the points implied by width, height, and space_bounds.
      This is useful for making videos where every frame uses the same
      evaluation points."""
    space_width = space_bounds.right - space_bounds.left
    space_height = space_bounds.top - space_bounds.bottom
    if space_width <= 0 or space_height <= 0:
      raise ValueError('Nonpositive space width or height')
    
    image = numpy.zeros((height, width, self.num_channels), self.dtype)
    
    # Create a Numpy array of the points at which to evaluate the texture.
    if eval_pts is None:
      eval_pts = self._gen_eval_pts(width, height, space_bounds)
    
    return self.evaluate(eval_pts)
  
  def to_video(self, width, height, space_bounds, num_frames,
    frames_per_second, filename):
    # Precompute the evaluation points so we don't have to recompute them every
    # frame.
    eval_pts = self._gen_eval_pts(width, height, space_bounds)
    try:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      video = cv2.VideoWriter(filename, fourcc, frames_per_second,
        (width, height))
      for frame_idx in range(num_frames):
        frame = self.to_image(width, height, space_bounds, eval_pts=eval_pts)
        video.write(frame)
        print('Frame written')
        self.step_frame()
    finally:
      if video is not None:
        video.release()
  
  def _gen_eval_pts(self, num_w_pixels, num_h_pixels, space_bounds):
    space_width = space_bounds.right - space_bounds.left
    space_height = space_bounds.top - space_bounds.bottom
    pixel_space_width = space_width / num_w_pixels
    pixel_space_height = space_height / num_h_pixels
    eval_pts = numpy.empty((num_h_pixels, num_w_pixels, 2), dtype=numpy.float64)
    for x in range(num_w_pixels):
      for y in range(num_h_pixels):
        eval_pts[y,x] = (x, (num_h_pixels - 1) - y)
    eval_pts += 0.5
    eval_pts[:,:,0] *= pixel_space_width
    eval_pts[:,:,1] *= pixel_space_height
    
    return eval_pts
