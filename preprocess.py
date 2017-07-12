import PIL
from PIL import Image
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt




WIDTH = 84
im = Image.open("im.jpg")
data = np.asarray( im, dtype='uint8' )



# print data.shape
# data = np.repeat(data.reshape(WIDTH, WIDTH, 1), 3, axis=2)
# print data.shape
plt.imshow(data, cmap = plt.get_cmap('gray'))
plt.show()


# for x in range(im.size[0]):
#     for y in range(im.size[1]):
#         a = int(round(luminance(pixelMap[x, y])))
#         pixelMap[x, y] = (a, a, a)


im.save("im.jpg")
im.close()
