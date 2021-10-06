import os
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2


class ImageLoader():
    def __init__(self):
        self.images = np.empty((0, 300, 300, 3), dtype=np.uint8)

    def load(self, input_dataset_dir):
        for image_name in os.listdir(input_dataset_dir):
            image = cv2.imread(os.path.join(input_dataset_dir, image_name))
            image = make_image_square(image)
            # changing image size to achieve similar image sizes to make image batch
            image = cv2.resize(image, (300, 300))
            # apprending to batch
            self.images = np.append(self.images, [image], axis=0)

        return self.images


def make_image_square(image):
    # dodawanie marginesów
    size = max(image.shape)
    # będziemy dodawać wzdłuż osi 0 (pionowej) jeśli obraz ma pion krótszy od poziomu, i na odwrót
    if np.argmax(image.shape):
        axis = 0
        margin_height = int((size - image.shape[0])/2)
        margin = np.zeros((margin_height, size, 3), dtype=np.uint8)
    else:
        axis = 1
        margin_width = int((size - image.shape[1])/2)
        margin = np.zeros((size, margin_width, 3), dtype=np.uint8)

    new_image = np.concatenate((margin, image, margin), axis=axis)
    return new_image


def save_images(images, output_dir):
    for i, image in enumerate(images):
        cv2.imwrite(f'{output_dir}/image_{i}.png', image)


def augument_and_append(images, augmentation_fcn):
    # applies augmentation function to images and appends generated images to the old ones
    images_aug = augmentation_fcn(images=images)
    images = np.append(images, images_aug, axis=0)
    return images


def augmentation_pipeline(images):
    aug_fcn = iaa.Fliplr()
    images = augument_and_append(images, aug_fcn)
    aug_fcn = iaa.Snowflakes(flake_size=(0.1, 0.3), speed=(0.01, 0.03))
    images = augument_and_append(images, aug_fcn)

    return images


# variables
input_dataset_dir = './dataset_dir'
output_dataset_dir = './dataset_mltpl'

# images loading
loader = ImageLoader()
images = loader.load(input_dataset_dir)

images = augmentation_pipeline(images)

save_images(images, output_dataset_dir)
