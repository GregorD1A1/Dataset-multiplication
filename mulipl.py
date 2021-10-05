import os
import cv2


input_dataset_dir = './dataset_dir'
output_dataset_dir = './dataset_mltpl'

for image_name in os.listdir(input_dataset_dir):
    image = cv2.imread(f'{input_dataset_dir}/{image_name}')
    # flipCode 1 for horizontal flip
    flipped = cv2.flip(image, flipCode=1)
    cv2.imwrite(f'{output_dataset_dir}/flipped_{image_name}', flipped)

