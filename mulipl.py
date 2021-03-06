import os
import imgaug.augmenters as iaa
import numpy as np
import cv2
from tqdm import tqdm


class ImageLoader:
    def __init__(self):
        self.process_size = 256
        self.images = np.empty((0, self.process_size, self.process_size, 3), dtype=np.uint8)

    def load(self, input_dataset_dir, n_images):
        print('Loading images...')
        # tqdm jest po to, by rysować ładne skale ładowania
        for image_name in tqdm(os.listdir(input_dataset_dir)[:n_images]):
            image = cv2.imread(os.path.join(input_dataset_dir, image_name))
            image = make_image_square(image)
            # changing image size to achieve similar image sizes to make image batch
            image = cv2.resize(image, (self.process_size, self.process_size))
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
    for i, image in tqdm(enumerate(images)):
        image = cv2.resize(image, (128, 128))
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

    # colorizing
    #print('operation 1/7...')
    #aug_fcn1 = iaa.WithChannels(0, iaa.Add((-50, -30)))
    #aug_fcn2 = iaa.WithChannels(0, iaa.Add((50, 30)))
    #aug_fcn3 = iaa.WithChannels(1, iaa.Add((-50, -30)))
    #aug_fcn4 = iaa.WithChannels(1, iaa.Add((50, 30)))
    #aug_fcn5 = iaa.WithChannels(2, iaa.Add((-50, -30)))
    #aug_fcn6 = iaa.WithChannels(2, iaa.Add((50, 30)))
    #images = augument_and_append(images, aug_fcn1, aug_fcn2, aug_fcn3, aug_fcn4, aug_fcn5, aug_fcn6)

    # rotation
    print('operation 2/7...')
    aug_fcn1 = iaa.Affine(rotate=(15, 20), scale=(1.15, 1.25))
    aug_fcn2 = iaa.Affine(rotate=(-15, -20), scale=(1.15, 1.25))
    images = augument_and_append(images, aug_fcn1, aug_fcn2)

    # lightening and darkening
    print('operation 3/7...')
    aug_fcn1 = iaa.Add(50)
    aug_fcn2 = iaa.Add(-50)
    images = augument_and_append(images, aug_fcn1, aug_fcn2)

    # histagram equalization (whatever it is)
    print('operation 4/7...')
    aug_fcn1 = iaa.BlendAlpha((0.4, 0.6), iaa.AllChannelsHistogramEqualization())
    images = augument_and_append(images, aug_fcn1)

    # flip
    print('operation 5/7...')
    aug_fcn1 = iaa.Fliplr()
    images = augument_and_append(images, aug_fcn1)

    # contrast change
    print('operation 6/7...')
    aug_fcn1 = iaa.GammaContrast((1.0, 2.0))
    images = augument_and_append(images, aug_fcn1)

    # perspective change
    print('operation 7/7...')
    aug_fcn1 = iaa.PerspectiveTransform(scale=(0.07, 0.07))
    images = augument_and_append(images, aug_fcn1)

    return images



# variables
input_dataset_dir = './img_align_celeba'
output_dataset_dir = './celeba_mltpl'
n_images = 10000

# images loading
loader = ImageLoader()
images = loader.load(input_dataset_dir, n_images)

#images = augmentation_pipeline(images)

save_images(images, output_dataset_dir)
