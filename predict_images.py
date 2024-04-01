import os
import cv2
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm

images_dir = 'images'

data_dir = 'trainset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'predict')

image_shape = (1024, 2048)

preprocess_input = sm.get_preprocessing('efficientnetb5')


def data_generator(data_path, seed=1):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        data_path, classes=[images_dir], class_mode=None,
        color_mode='rgb', target_size=image_shape, batch_size=1,
        seed=seed)

    for x in image_generator:
        yield np.expand_dims(preprocess_input(np.squeeze(x).astype(np.uint8)), axis=0)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


validation_generator_instance = data_generator(test_dir, seed=42)

model = sm.Unet('efficientnetb5', classes=1, activation='sigmoid')
model.load_weights(os.path.join(data_dir, 'weights-new.h5'))

optim = tf.keras.optimizers.Adam(1e-4)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [
    sm.metrics.IOUScore(threshold=.5),
    sm.metrics.FScore(threshold=.5),
]

model.compile(optim, total_loss, metrics)

while True:
    image = next(validation_generator_instance)
    mask_out = model.predict(image)

    img = normalize(np.squeeze(image[0]))
    mask_filter = np.squeeze(mask_out[0]) > .5

    alpha = 0.4
    img[mask_filter] = img[mask_filter] * (1 - alpha) + np.array([0, 1, 1]) * alpha

    plt.imshow(img)
    plt.show()

    cv2.imwrite('out.png', cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR))
