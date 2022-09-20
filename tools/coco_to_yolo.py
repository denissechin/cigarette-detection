import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cigarette detection dataset generation')
    parser.add_argument('--annotation-file', type=str, help='Path to annotation .json file',
                        required=True)
    parser.add_argument('--images-path', type=str, help='Path to directory containing images',
                        required=True)
    parser.add_argument('--save-path', type=str, help='Path to original person dataset',
                        required=True)

    args = parser.parse_args()
    dataset_name = os.path.basename(args.save_path)
    Path(args.save_path).mkdir(exist_ok=True, parents=True)

    with open(args.annotation_file) as f:
        data = json.load(f)

    with open(os.path.join(args.save_path, f'{dataset_name}.txt'), 'w') as file:
        file.truncate(0)

    images = {'%g' % x['id']: x for x in data['images']}

    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    for img_id, anns in imgToAnns.items():
        img = images['%g' % img_id]
        h, w, filename = img['height'], img['width'], os.path.splitext(os.path.basename(img['file_name']))[0]
        subdir = os.path.dirname(img['file_name'])
        bboxes = []
        segments = []
        for ann in anns:
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = ann['category_id']  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)

        # Write
        image_path = os.path.join(args.images_path, img['file_name'])
        save_path = os.path.join(args.save_path, 'images', subdir)
        Path(save_path).mkdir(exist_ok=True, parents=True)

        shutil.copyfile(image_path, os.path.join(save_path, os.path.basename(img['file_name'])))

        ann_path = os.path.join(args.save_path, 'labels', subdir)
        Path(ann_path).mkdir(exist_ok=True, parents=True)

        with open(os.path.join(ann_path, f'{filename}.txt'), 'w') as file:
            for i in range(len(bboxes)):
                line = *(bboxes[i]),  # cls, box or segments
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

        with open(os.path.join(args.save_path, f'{dataset_name}.txt'), 'a') as file:
            file.write(f'.{os.sep}images{os.sep}' + img['file_name'] + '\n')
