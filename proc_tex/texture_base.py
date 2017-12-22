import subprocess

import numpy

class TimeSpaceTexture:
  """Base class for still or animated textures.
  Textures of this type support two dimensions of space and one of time. This
  class can be used for non-animated textures by simply not overriding
  step_frame."""
  def __init__(self, num_channels, dtype, num_space_dims):
    """Initializer.
    num_channels - The number of channels per pixel.
    dtype - The Numpy dtype to use for the data in each channel.
    num_space_dims - The number of spatial dimensions expected in an image
      generated from the texture. Typically 2."""
    self.num_channels = num_channels
    self.dtype = dtype
    self.num_space_dims = num_space_dims
  
  def evaluate(self, eval_pts):
    """Gets the pixel values at the specified locations. Subclasses should
    typically override this. Subclasses are not required to return a consistent
    value when evaluated multiple times at the same point (this helps with the
    implementation of e.g. white noise). Therefore, callers that need
    consistency should cache the values if they are needed more than once, or
    ensure that they only use subclasses with consistent behavior. Default
    implementation returns all zeros.
    eval_pts - A Numpy array of evaluation points. The size of the last
      dimension should equal the number of dimensions in an evaluation point
      (e.g., dimension size 2 for a 2D texture).
    returns: A Numpy array of evaluation results. This should have the same
      shape as eval_pts, except with the last dimension size changed to the
      number of channels supported by this texture."""
    return numpy.zeros(eval_pts.size[:-1] + (self.num_channels,))
  
  def step_frame(self):
    """Moves internal state to the next frame.
    For animated textures, this will typically result in evaluate changing what
    values it returns. Subclasses that require animation should typically
    override this. Default implementation does nothing."""
    pass
  
  def to_image(self, pixel_dims, space_bounds, eval_pts=None):
    """Generates a Numpy array representing an image of the current frame.
    Assuming the texture's number of channels, channel dtype, and number of
    spatial dimensions are supported by OpenCV, the image should be compatible
    with PyOpenCV functions. (Presumably the number of spatial dimensions must
    be 2 for this to work.)
    pixel_dims - Numpy shape object specifying the number of pixels in each
      dimension. E.g., for a 2D image pixel_dims[0] is the width and
      pixel_dims[1] the height.
    space_bounds - An indexable object of shape (self.num_space_dims, 2).
      self.num_space_dims is the number of spatial dimensions supported by this
      texture (typically 2). space_bounds[i,0] and space_bounds[i,1] give the
      lower and upper bounds (respectively) on dimension i.
    eval_pts - Optional precomputed Numpy array of evaluation points. If this is
      provided, pixel_dims and space_bounds are ignored. This can be useful when
      a texture is evaluated repeatedly at the same evaluation points, e.g. when
      making a video."""
    # Generate evaluation points.
    if eval_pts is None:
      eval_pts = self.gen_eval_pts(pixel_dims, space_bounds)
    
    return self.evaluate(eval_pts)
  
  def to_video(self, pixel_dims, space_bounds, num_frames, frames_per_second,
    filename, pix_fmt, codec='libvpx-vp9', codec_params=[], eval_pts=None):
    """Generates a video starting at the current frame.
    This method has the side effect of moving the current frame forward by
    num_frames. Since FFmpeg requires the number of spatial dimensions to be 2,
    this method also requires that.
    pixel_dims - See to_image.
    space_bounds - See to_image.
    num_frames - Number of frames to include in the video.
    frames_per_second - Number of frames per second to use in the video.
    filename - Location at which to store the video.
    pix_fmt - Input pixel format string to pass to FFmpeg.
    codec - Video codec string to pass to FFmpeg.
    codec_params - Extra codec parameters to pass to FFmpeg.
    eval_pts - See to_image."""
    if self.num_space_dims != 2:
      raise ValueError(
        'Cannot make videos with number of dimensions other than 2.')
    
    # Precompute the evaluation points so we don't have to recompute them every
    # frame.
    if eval_pts is None:
      eval_pts = self.gen_eval_pts(pixel_dims, space_bounds)
    
    # Start the FFmpeg process.
    video_size_arg = '{}x{}'.format(eval_pts.shape[0], eval_pts.shape[1])
    global_args = ['-y']
    input_args = ['-f', 'rawvideo', '-pixel_format', pix_fmt,
      '-video_size', video_size_arg, '-framerate', str(frames_per_second),
      '-i', '-',]
    output_args = ['-r', str(frames_per_second),
      '-codec:v', codec] + codec_params + ['-frames:v', str(num_frames),
      '-s', video_size_arg, filename]
    ffmpeg_process = None
    try:
      ffmpeg_process = subprocess.Popen(
        ['ffmpeg'] + global_args + input_args + output_args,
        stdin=subprocess.PIPE)
      
      # Generate frames.
      for frame_idx in range(num_frames):
        frame = self.to_image(pixel_dims, space_bounds, eval_pts=eval_pts)
        
        ffmpeg_process.stdin.write(frame.tobytes())
        
        self.step_frame()
    
    finally:
      if ffmpeg_process is not None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
  
  def gen_eval_pts(self, pixel_dims, space_bounds):
    """Generates a Numpy array of evaluation points.
    The resulting array is suitable for passing to to_image as the optional
    eval_pts parameter.
    pixel_dims - See to_image.
    space_bounds - See to_image."""
    # convert arguments to Numpy arrays so we can use the arithmetic operations.
    pixel_dims = numpy.array(pixel_dims)
    space_bounds = numpy.array(space_bounds)
    
    space_widths = numpy.array(
      [space_bounds[dim,1] - space_bounds[dim,0] for dim in range(self.num_space_dims)])
    
    pixel_dims_ranges = [numpy.arange(dim) for dim in pixel_dims]
    
    # Initialize each evaluation point to match its pixel coordinates.
    eval_pts = numpy.asarray(numpy.meshgrid(*pixel_dims_ranges)).astype(
      numpy.float64)
    eval_pts = numpy.rollaxis(eval_pts, 0, len(eval_pts.shape))
    
    # Convert from pixel coordinates to texture space coordinates.
    eval_pts += 0.5
    scale = space_widths / pixel_dims
    offset = space_bounds[:,0]
    eval_pts = eval_pts * scale + offset
    
    return eval_pts
