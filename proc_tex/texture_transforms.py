import numpy

from proc_tex.texture_base import TransformedTexture

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
  def space_transform(eval_pts):
    return [eval_pts]
  def tex_transform(src_vals):
    src_vals = src_vals[0]
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
  
  return TransformedTexture(src.num_channels, src.num_space_dims, [src],
    space_transform, tex_transform, src.allow_anim)

def tex_to_dtype(src, dtype, scale=1):
  """Converts a texture to the given dtype for each channel.
  src - The source texture to transform.
  dtype - The dtype to convert to.
  scale - Value by which to multiply before the conversion. This can be useful
    for converting floating point images in the range [0, 1] into integer
    images in the range [0, 2^b - 1].
  Returns: The transformed texture."""
  def space_transform(eval_pts):
    return [eval_pts]
  def tex_transform(src_vals):
    return (src_vals[0] * scale).astype(dtype)
  
  return TransformedTexture(src.num_channels, src.num_space_dims, [src],
    space_transform, tex_transform, src.allow_anim)
