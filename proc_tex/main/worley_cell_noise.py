from proc_tex.WorleyCellNoise2D import WorleyCellNoise2D

if __name__ == '__main__':
  texture = WorleyCellNoise2D(10, 10, 1, 1)
  image = texture.to_image(512, 512, (0, 1, 1, 0))
  image.show()
