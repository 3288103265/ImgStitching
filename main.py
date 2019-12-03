import json
import os

import cv2
import numpy as np

from blend import ImageInfo, blendImages
from feature import get_mapping
from warp import warp_spherical


def load_images(dirpath):
    """
    Load images from the dirpath
    :param dirpath containing a series of images
    :return: list of cv2 images
    """
    if not dirpath:
        return
    files = sorted(os.listdir(dirpath))
    files = [
        f for f in files
        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.ppm')
    ]
    images = [cv2.imread(os.path.join(dirpath, i)) for i in files]
    print('Load {0} images successfully!'.format(len(images)))
    return images


def warp_images(images, f=595, k1=-0.15, k2=0.0):
    """
    warp images to spherical coordinates.
    :return: warpped images list
    """
    return [
        warp_spherical(img, f, k1, k2) for img in images
    ]


imgs = load_images('panorama')
warpped = warp_images(imgs)


t = np.eye(3)
info = []
T = []
for i in range(len(warpped)-1):
    print('Computing mapping from {0} to {1}'.format(i, i+1))
    info.append(ImageInfo('', warpped[i], np.linalg.inv(t)))
    tmp = get_mapping(warpped[i], warpped[i+1])
    print(tmp)
    t = tmp.dot(t)
    T.append(tmp)

info.append(ImageInfo('', warpped[len(warpped)-1], np.linalg.inv(t)))
print('Computing mapping from {0} to {1}'.format(len(warpped)-1, 0))
tmp = get_mapping(warpped[len(warpped)-1], warpped[0])
print(tmp)
t = tmp.dot(t)
info.append(ImageInfo('', warpped[0], np.linalg.inv(t)))

# json.dump(T, fp=open('transform_output.json','w'), indent=4)
print('Blending Images.')
panorama = blendImages(info, blendWidth=50, is360=True)
cv2.imwrite('Panorama.jpg', panorama)
