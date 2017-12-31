import subprocess

import numpy

class Texture:
  """Base class for still or animated textures.
  This class can be used for non-animated textures by simply not overriding
  step_frame."""
  def __init__(self, num_channels, num_space_dims, anim_synch_textures=[]):
    """Initializer.
    num_channels - The number of channels per pixel.
    num_space_dims - The number of spatial dimensions expected in an image
      generated from the texture. Typically 2.
    anim_synch_textures - List of textures that should be animated along with
      this texture. Can be useful for animated textures that are derived from
      other animated textures."""
    self.num_channels = num_channels
    self.num_space_dims = num_space_dims
    self.anim_synch_textures = anim_synch_textures
    self.curr_frame = max(
      [0] + [texture.curr_frame for texture in anim_synch_textures])
  
  def evaluate(self, eval_pts):
    """Gets the pixel values at the specified locations. Subclasses should
    typically override this. Default implementation returns all zeros.
    eval_pts - A Numpy array of evaluation points. The size of the last
      dimension should equal the number of dimensions expected in an evaluation
      point (e.g., dimension size 2 for a 2D texture).
    returns: A Numpy array of evaluation results. This should have the same
      shape as eval_pts, except with the last dimension size changed to the
      number of channels supported by this texture."""
    return numpy.zeros(eval_pts.shape[:-1] + (self.num_channels,))
  
  def set_frame(self, frame_idx):
    """Moves internal state to the specified frame.
    Does not support going back before the current frame.
    frame_idx - Index of the frame to go to."""
    while self.curr_frame < frame_idx:
      self.step_frame()
      self.curr_frame += 1
    
    # Move any animation-synchronized textures along with this texture.
    for texture in self.anim_synch_textures:
      texture.set_frame(frame_idx)
  
  def step_frame(self):
    """Moves internal state to the next frame.
    For animated textures, this will typically result in the evaluate method
    changing what values it returns. Subclasses that require animation should
    typically override this. Default implementation does nothing."""
    pass
  
  def to_image(self, pixel_dims, space_bounds, eval_pts=None):
    """Generates a Numpy array representing an image of the current frame.
    Assuming the texture's number of channels, channel dtype, and number of
    spatial dimensions are supported by OpenCV, the image should be compatible
    with PyOpenCV functions.
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
      start_frame = self.curr_frame
      for frame_idx in range(num_frames):
        self.set_frame(start_frame + frame_idx)
        
        frame = self.to_image(pixel_dims, space_bounds, eval_pts=eval_pts)
        
        ffmpeg_process.stdin.write(frame.tobytes())
    
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
  
  def __add__(self, other):
    """Texture addition.
    See BinaryCombinedTexture for restrictions on what can be added."""
    return _SimpleBinaryCombinedTexture(self, other, lambda x,y: x + y)
  
  def __radd__(self, other):
    """Right-hand side texture addition.
    See BinaryCombinedTexture for restrictions on what can be added."""
    return _SimpleBinaryCombinedTexture(other, self, lambda x,y: x + y)
  
  def __sub__(self, other):
    """Texture subtraction.
    See BinaryCombinedTexture for restrictions on what can be subtracted."""
    return _SimpleBinaryCombinedTexture(self, other, lambda x,y: x - y)
  
  def __rsub__(self, other):
    """Right-hand side texture subtraction.
    See BinaryCombinedTexture for restrictions on what can be subtracted."""
    return _SimpleBinaryCombinedTexture(other, self, lambda x,y: x - y)
  
  def __neg__(self):
    return 0 - self
  
  def __mul__(self, other):
    """Texture multiplication.
    See BinaryCombinedTexture for restrictions on what can be multiplied."""
    return _SimpleBinaryCombinedTexture(self, other, lambda x,y: x * y)
  
  def __rmul__(self, other):
    """Right-hand side texture multiplication.
    See BinaryCombinedTexture for restrictions on what can be multiplied."""
    return _SimpleBinaryCombinedTexture(other, self, lambda x,y: x * y)

class ScalarConstantTexture(Texture):
  """Texture that outputs a constant scalar value on each channel."""
  
  def __init__(self, num_channels, num_space_dims, value):
    """Initializer.
    value - The constant scalar value to output."""
    super(ScalarConstantTexture, self).__init__(num_channels, num_space_dims)
    self.value = value
  
  def evaluate(self, eval_pts):
    return numpy.full(eval_pts.shape[:-1] + (self.num_channels,), self.value)

class TransformedTexture(Texture):
  """Class for applying transformation functions to source texture(s)."""
  
  def __init__(self, num_channels, num_space_dims, src_textures,
    space_transform, tex_transform, anim_synch_textures=[]):
    """Initializer.
    src_textures - Iterable of source textures to which transformations will be
      applied.
    space_transform - A function that transforms the evaluation points before
      they are passed to the source textures. Takes a single argument: a Numpy
      array of evaluation points with number of space dimensions equal to
      num_space_dims. Returns an iterable of transformed Numpy arrays, one for
      each source texture. The ith returned array must have number of space
      dimensions matching that of the ith source texture.
    tex_transform - A function that transforms the texture outputs after they
      are obtained from the source textures. Takes a single argument: an
      iterable of the Numpy arrays representing the texture output of each
      source texture. Returns a Numpy array representing the final texture
      output. The returned array must have the appropriate number of channels
      for the texture.
    anim_synch_textures - See superclass. src_textures get added
      automatically."""
    super(TransformedTexture, self).__init__(num_channels, num_space_dims,
      anim_synch_textures + src_textures)
    self.src_textures = src_textures
    self.space_transform = space_transform
    self.tex_transform = tex_transform
  
  def evaluate(self, eval_pts):
    transformed_eval_pts = self.space_transform(eval_pts)
    src_outputs = [src_texture.evaluate(pts) for src_texture, pts in zip(self.src_textures, transformed_eval_pts)]
    return self.tex_transform(src_outputs)

class _SimpleBinaryCombinedTexture(TransformedTexture):
  """Simple texture transformation for implementing overloaded operators."""
  
  def __init__(self, src0, src1, combination):
    """Combines two textures according to the specified combination function.
    src0 and src1 should have the same number of space dimensions and channels
    if both are textures. At least one of the two must be a texture object, but
    one can be a scalar.
    src0 - First texture to combine, or a scalar to combine with src1.
    src1 - Second texture to combine, or a scalar to combine with src0.
    combination - A combining function that takes two Numpy arrays of evaluated
      texture points and combines them into one.
    """
    if not (isinstance(src0, Texture) or isinstance(src1, Texture)):
      raise ValueError('Expected at least one source texture.')
    
    # Convert scalars to constant-valued textures.
    if not isinstance(src0, Texture):
      src0 = ScalarConstantTexture(src1.num_channels, src1.num_space_dims, src0)
    if not isinstance(src1, Texture):
      src1 = ScalarConstantTexture(src0.num_channels, src0.num_space_dims, src1)
    
    def space_transform(eval_pts):
      return [eval_pts, eval_pts]
    def tex_transform(src_vals):
      return combination(src_vals[0], src_vals[1])
    
    super(_SimpleBinaryCombinedTexture, self).__init__(src0.num_channels,
      src0.num_space_dims, [src0, src1], space_transform, tex_transform)
