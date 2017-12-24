import numpy

from proc_tex.texture_base import TimeSpaceTexture

class UnaryTransformedTexture(TimeSpaceTexture):
  """Class for applying a transformation function to a single source texture."""
  
  def __init__(self, transform, src_texture):
    """Initializer.
    transform - A function that takes a Numpy array of evaluated texture values
      and returns the transformed array. The transformed array should have the
      same dimensions, except possibly for the last dimension which represents
      the number of channels.
    src_texture - The texture to which the transform will be applied."""
    super(UnaryTransformedTexture, self).__init__(src_texture.num_channels,
      src_texture.dtype, src_texture.num_space_dims)
    self.transform = transform
    self.src_texture = src_texture
    self.curr_frame = src_texture.curr_frame
  
  def evaluate(self, eval_pts):
    return self.transform(self.src_texture.evaluate(eval_pts))
  
  def set_frame(self, frame_idx):
    self.src_texture.set_frame(frame_idx)

def _tex_scale_to_region_transform(src_vals, min_value, max_value):
  """Transform function for tex_scale_to_region."""
  src_min_value = src_vals.min()
  src_max_value = src_vals.max()
  src_delta = src_max_value - src_min_value
  delta = max_value - min_value
  
  if src_delta == 0:
    return numpy.full_like(src_vals, min_value + delta / 2)
  else:
    scale = delta / src_delta
    offset = min_value - (src_min_value * scale)
    return src_vals * scale + offset

def tex_scale_to_region(src, min_value=0, max_value=1):
  """Scales and offsets a floating point texture's values to the given range.
  Each frame will be scaled and offset so that its minimum texture value is
  min_value and its maximum is max_value. (All channels receive the same scale
  and offset in each frame.) If this is not possible because the texture values
  are all the same, all the values will be moved to halfway between min_value
  and max_value. Note that each frame in a video is handled separately, so the
  transform may not be uniform across video frames.
  src - The source texture to transform.
  min_value - The desired minimum texture value.
  max_value - The desired maximum texture value.
  Returns: The transformed texture."""
  return UnaryTransformedTexture(
    lambda src_vals: _tex_scale_to_region_transform(src_vals, min_value, max_value),
    src)

def _tex_to_dtype_transform(src_vals, dtype, scale):
  """Transform function for tex_to_dtype."""
  return (src_vals * scale).astype(dtype)

def tex_to_dtype(src, dtype, scale=1):
  """Converts a texture to the given dtype for each channel.
  src - The source texture to transform.
  scale - Value by which to multiply before the conversion. This can be useful
    for converting floating point images in the range [0, 1] into integer
    images in the range [0, 2^b - 1].
  Returns: The transformed texture."""
  return UnaryTransformedTexture(
    lambda src_vals: _tex_to_dtype_transform(src_vals, dtype, scale),
    src)
