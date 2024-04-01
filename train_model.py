import os

os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import albumentations
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm

image_shape = (1024, 1024)

images_dir, masks_dir = 'images', 'masks'

data_dir = 'trainset'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

augmenter = albumentations.Compose([
    albumentations.OneOf([
        albumentations.HueSaturationValue(p=1),
        albumentations.RandomBrightnessContrast(p=1),
        albumentations.FancyPCA(p=1),
        albumentations.ColorJitter(p=1),
        albumentations.RGBShift(p=1)
    ], p=.5),
    albumentations.HorizontalFlip(p=.2),
    albumentations.RandomResizedCrop(height=image_shape[0], width=image_shape[1], p=.3),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=.35),
])

preprocess_input = sm.get_preprocessing('efficientnetb5')


def data_generator(data_path, augmentation=False, seed=1):
    image_data_generator = tf.keras.preprocessing.image1.ImageDataGenerator()

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

        if augmentation:
            augmented = augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        x = preprocess_input(x)
        x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        y = (y / 255).astype(np.float32)

        yield x, y


train_generator_instance = data_generator(train_dir, augmentation=True, seed=42)
validation_generator_instance = data_generator(validation_dir, augmentation=False, seed=42)

model = sm.Unet('efficientnetb5', classes=1, activation='sigmoid')

optim = tf.keras.optimizers.Adam(1e-4)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [
    sm.metrics.IOUScore(threshold=.5),
    sm.metrics.FScore(threshold=.5),
]

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

for _ in range(2):
    image, _ = next(train_generator_instance)
    plt.imshow(normalize(np.squeeze(image[0])))
    plt.show()

model.compile(optim, total_loss, metrics)

weights_checkpoint_dir = os.path.join(data_dir, 'weights-new.h5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_checkpoint_dir, monitor='loss', save_best_only=True, mode='min')

callbacks = [
    model_checkpoint,
]

model.load_weights('weights.h5')
model.fit(
    train_generator_instance,
    steps_per_epoch=270,
    validation_data=validation_generator_instance,
    validation_steps=30,
    epochs=200,
    verbose=1,
    shuffle=True,
    callbacks=callbacks)
