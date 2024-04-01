import os
import cv2
import json
import numpy as np

from coco import COCO


directory_names = [
    'task_test-upload-2022_06_23_08_20_18-coco 1.0',
    'task_ingangen brt + osm (coco)-2022_06_23_11_09_25-coco 1.0',
]

for directory in directory_names:
    images_directory = os.path.join('downloads', directory, 'images')
    annotations_file_name = os.path.join('downloads', directory, 'annotations', 'instances_default.json')

    with open(annotations_file_name, 'r') as fh:
        data = json.load(fh)
        c = COCO(annotations_file_name)

        id_to_mask = {}
        for image in data['images']:
            if image['file_name'].startswith('OSM'):
                continue

            id_to_mask[image['id']] = np.zeros(shape=[image['height'], image['width'], 1], dtype=np.uint8)

        for annotation in data['annotations']:
            if annotation['image_id'] not in id_to_mask:
                continue

            mask = id_to_mask[annotation['image_id']]
            where_nonzero = np.where(c.annToMask(annotation) > 0)
            mask[where_nonzero[0], where_nonzero[1]] = 255

        for image_id, mask in id_to_mask.items():
            if not len(np.where(mask > 0)[0]):
                continue

            item = next((x for x in data['images'] if x['id'] == image_id))
            all_locations = {item['file_name'] for item in data['images']}

            mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join('trainset', 'data', 'masks', item['file_name']), mask_resized)

            image = cv2.imread(os.path.join(images_directory, item['file_name']))
            image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join('trainset',  'data', 'images', item['file_name']), image_resized)
