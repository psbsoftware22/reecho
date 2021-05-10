import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_lfw_dataset(RAW_IMAGES_NAME, dx=80, dy=80, dimx=45, dimy=45, grayScale=False):

    all_photos = []

    print('Data load initiated') 

    for subdir, dirs, files in os.walk(RAW_IMAGES_NAME):
        for file in files:
            img = np.array(Image.open(os.path.join(subdir, file)))
            if (grayScale):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop only faces and resize it
            img = img[dy:-dy, dx:-dx]
            img = cv2.resize(img, (dimx, dimy))

            all_photos.append(img)
                

    all_photos = np.stack(all_photos).astype('uint8')

    return all_photos

def img_preprocessing(RAW_IMAGES_NAME, zoom=80, resolution=[32,32], grayScale=False):
    X = _load_lfw_dataset(RAW_IMAGES_NAME, dx=zoom, dy=zoom, dimx=resolution[0], dimy=resolution[1], grayScale=grayScale)

    X = X.astype('float32') / 255.0 - 0.5
    print(X.max(), X.min())
        
    return X

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()
    
'''
if __name__ == "__main__":
    RAW_IMAGES_NAME = "reecho/data/lfw"
    X = img_preprocessing(RAW_IMAGES_NAME)'''
    
