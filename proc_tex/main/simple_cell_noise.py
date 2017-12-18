from proc_tex.SimpleCellNoise import SimpleCellNoise

if __name__ == '__main__':
  texture = SimpleCellNoise()
  image = texture.to_image(256, 256, (0, 1, 1, 0))
  image.show()
