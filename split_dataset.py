import os
import shutil
import numpy as np

images_dir, masks_dir = 'images', 'masks'
data_dir = r'C:\Users\FlorijnWim\PycharmProjects\parking-garages\trainset\data'

train_dir = r'C:\Users\FlorijnWim\PycharmProjects\parking-garages\trainset\train'
validation_dir = r'C:\Users\FlorijnWim\PycharmProjects\parking-garages\trainset\validation'

try:
    shutil.rmtree(train_dir)
except:
    pass
os.mkdir(train_dir)

try:
    shutil.rmtree(validation_dir)
except:
    pass
os.mkdir(validation_dir)

split = .9
file_names = list(os.listdir(os.path.join(data_dir, masks_dir)))
split_index = int(split * len(file_names))

try:
    shutil.rmtree(os.path.join(train_dir, images_dir))
except:
    pass
os.mkdir(os.path.join(train_dir, images_dir))

try:
    shutil.rmtree(os.path.join(train_dir, masks_dir))
except:
    pass
os.mkdir(os.path.join(train_dir, masks_dir))

np.random.shuffle(file_names)
train_names = file_names[:split_index]

for i, file_name in enumerate(train_names):
    shutil.copyfile(os.path.join(data_dir, images_dir, file_name),
                    os.path.join(train_dir, images_dir, str(i) + file_name))
    shutil.copyfile(os.path.join(data_dir, masks_dir, file_name),
                    os.path.join(train_dir, masks_dir, str(i) + file_name))

try:
    shutil.rmtree(os.path.join(validation_dir, images_dir))
except:
    pass

os.mkdir(os.path.join(validation_dir, images_dir))

try:
    shutil.rmtree(os.path.join(validation_dir, masks_dir))
except:
    pass

os.mkdir(os.path.join(validation_dir, masks_dir))

validation_names = file_names[split_index:]
for i, file_name in enumerate(validation_names):
    shutil.copyfile(os.path.join(data_dir, images_dir, file_name),
                    os.path.join(validation_dir, images_dir, str(i) + file_name))
    shutil.copyfile(os.path.join(data_dir, masks_dir, file_name),
                    os.path.join(validation_dir, masks_dir, str(i) + file_name))
