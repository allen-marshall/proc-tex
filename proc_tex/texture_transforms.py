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
    space_transform, tex_transform)

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
    space_transform, tex_transform)

def tex_to_num_channels(src, num_channels):
  """Converts a texture to have the specified number of channels.
  If src has more than num_channels, the higher-numbered channels will be
  discarded. If src has less than num_channels, channels will be repeated in
  round robin order until the required number of channels is reached.
  src - The source texture to transform.
  num_channels - The desired number of channels.
  Returns: The transformed texture."""
  def space_transform(eval_pts):
    return [eval_pts]
  def tex_transform(src_vals):
    src_vals = src_vals[0]
    curr_num_channels = src_vals.shape[-1]
    
    # Repeat channels in round robin order if we have too few.
    if curr_num_channels < num_channels:
      num_channels_to_add = num_channels - curr_num_channels
      pad_widths = ((0, 0),) * (len(src_vals.shape) - 1) + ((0, num_channels_to_add),)
      return numpy.pad(src_vals, pad_widths, 'wrap')
    
    # Discard channels if we have too many.
    elif curr_num_channels > num_channels:
      index = (slice(None),) * (len(src_vals.shape) - 1) + (slice(0, num_channels),)
      return src_vals[index]
    
    # Do nothing if we already have the desired number of channels.
    else:
      return src_vals

def tex_concat_channels(src_textures):
  """Concatenates the channels of multiple source textures.
  The resulting texture has a number of channels equal to
  sum([src.num_channels for src in src_textures]).
  src_textures - The textures to concatenate. All source textures should have
    the same number of space dimensions.
  Returns: The transformed texture."""
  if len(src_textures) == 0:
    raise ValueError('Must have at least one source texture')
  
  def space_transform(eval_pts):
    return [eval_pts] * len(src_textures)
  def tex_transform(src_vals):
    return numpy.concatenate(src_vals, axis=-1)
  
  new_num_channels = sum([src.num_channels for src in src_textures])
  
  return TransformedTexture(new_num_channels, src_textures[0].num_space_dims,
    src_textures, space_transform, tex_transform)
