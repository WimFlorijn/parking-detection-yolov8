import os
import albumentations

os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm

images_dir, masks_dir = 'images', 'masks'

data_dir = 'trainset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'train')

image_shape = (1024, 2048)

preprocess_input = sm.get_preprocessing('efficientnetb5')

augmenter = albumentations.Compose([
    albumentations.OneOf([
        albumentations.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50,p=1),
        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4,p=1),
        albumentations.FancyPCA(p=1),
        albumentations.ColorJitter(p=1),
        albumentations.RGBShift(p=1)
    ], p=.5),
    albumentations.HorizontalFlip(p=.2),
    albumentations.RandomResizedCrop(scale=(0.5, 1.0), height=image_shape[0], width=image_shape[1], p=.3),
    albumentations.OneOf([
        albumentations.RandomRain(p=1),
        albumentations.RandomSnow(p=1),
        albumentations.RandomSunFlare(p=1)
    ], p=.35),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=.35),
])


def data_generator(data_path, seed=1):
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    image_generator = image_data_generator.flow_from_directory(
        data_path, classes=[images_dir], class_mode=None,
        color_mode='rgb', target_size=image_shape, batch_size=1,
        seed=seed)

    mask_generator = image_data_generator.flow_from_directory(
        data_path, classes=[masks_dir], class_mode=None,
        color_mode='grayscale', target_size=image_shape, batch_size=1,
        seed=seed)

    for x, y in zip(image_generator, mask_generator):
        x = np.squeeze(x).astype(np.uint8)
        y = np.squeeze(y).astype(np.uint8)

        yield x, y


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


validation_generator_instance = data_generator(test_dir, seed=42)

while True:
    image, mask = next(validation_generator_instance)

    images = []
    for x in range(4):
        augmented = augmenter(image=image, mask=mask)
        image1, mask1 = augmented['image'], augmented['mask']

        image1 = preprocess_input(image1)
        mask1 = (mask1 / 255).astype(np.float32)

        img = normalize(np.squeeze(image1))
        mask_filter = np.squeeze(mask1) > .5

        alpha = 0.4
        img[mask_filter] = img[mask_filter] * (1 - alpha) + np.array([0, 1, 1]) * alpha
        images.append(np.copy(img))

    plt.imshow(np.vstack((np.hstack(images[:2]), np.hstack(images[2:]))))
    plt.show()
