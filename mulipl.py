import os
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2


class ImageLoader():
    def __init__(self):
        self.images = np.empty((0, 300, 300, 3), dtype=np.uint8)

    def load(self, input_dataset_dir):
        print('Loading images...')
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
    print('Saving images...')
    for i, image in enumerate(images):
        #image = cv2.resize(image, (128, 128))
        cv2.imwrite(f'{output_dir}/image_{i}.png', image)


def augument_and_append(images, *augmentation_fcns):
    # applies augmentation function to images and appends generated images to the old ones
    # inserting more than one function causes parallel generation and appending
    new_images = images
    for augmentation_fcn in augmentation_fcns:
        augmented_images = augmentation_fcn(images=images)
        new_images = np.append(new_images, augmented_images, axis=0)
    return new_images


def augmentation_pipeline(images):
    print('Generating new images...')
    
    # flip
    aug_fcn1 = iaa.Fliplr()
    images = augument_and_append(images, aug_fcn1)

    # lightening and darkening
    aug_fcn1 = iaa.Add(50)
    aug_fcn2 = iaa.Add(-50)
    images = augument_and_append(images, aug_fcn1, aug_fcn2)

    # colorizing
    aug_fcn1 = iaa.WithChannels(0, iaa.Add((-50, -30)))
    aug_fcn2 = iaa.WithChannels(0, iaa.Add((50, 30)))
    aug_fcn3 = iaa.WithChannels(1, iaa.Add((-50, -30)))
    aug_fcn4 = iaa.WithChannels(1, iaa.Add((50, 30)))
    aug_fcn5 = iaa.WithChannels(2, iaa.Add((-50, -30)))
    aug_fcn6 = iaa.WithChannels(2, iaa.Add((50, 30)))
    images = augument_and_append(images, aug_fcn1, aug_fcn2, aug_fcn3, aug_fcn4, aug_fcn5, aug_fcn6)

    # contrast change
    aug_fcn1 = iaa.GammaContrast((1.0, 2.0))
    images = augument_and_append(images, aug_fcn1)

    # histagram equalization (whatever it is)
    aug_fcn1 = iaa.BlendAlpha((0.4, 0.6), iaa.AllChannelsHistogramEqualization())
    images = augument_and_append(images, aug_fcn1)

    # perspective change
    aug_fcn1 = iaa.PerspectiveTransform(scale=(0.07, 0.07))
    images = augument_and_append(images, aug_fcn1)

    # rotation
    aug_fcn1 = iaa.Affine(rotate=(15, 20), scale=(1.15, 1.25))
    aug_fcn2 = iaa.Affine(rotate=(-15, -20), scale=(1.15, 1.25))
    images = augument_and_append(images, aug_fcn1, aug_fcn2)

    return images


# variables
input_dataset_dir = './dataset_dir'
output_dataset_dir = './dataset_mltpl'

# images loading
loader = ImageLoader()
images = loader.load(input_dataset_dir)

images = augmentation_pipeline(images)

save_images(images, output_dataset_dir)
