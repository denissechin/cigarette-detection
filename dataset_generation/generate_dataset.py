import argparse
import os
from pathlib import Path
import random

import cv2
import numpy as np
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from tqdm import tqdm

from generation_modules import HandDetector, LipsDetector, CigaretteAdder
from generation_utils import xyxy2xywh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cigarette detection dataset generation')
    parser.add_argument('--image-path', type=str, help='Path to original person dataset',
                        required=True)
    parser.add_argument('--cigarette-dataset-path', type=str, help='Path to original person dataset',
                        required=True)
    parser.add_argument('--save-path', type=str, help='Path where to store generated images',
                        required=True)
    parser.add_argument('--num-repeats', type=int, help='Number of times cigarette is added separately to each image',
                        required=False, default=1)

    args = parser.parse_args()

    image_path_list = [os.path.join(args.image_path, p) for p in os.listdir(args.image_path)]
    Path(os.path.join(args.save_path, 'train')).mkdir(exist_ok=True, parents=True)

    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='cigarette'))

    hand_detector = HandDetector()
    lips_detector = LipsDetector()
    cigarette_adder = CigaretteAdder(cigarette_path=args.cigarette_dataset_path)

    image_counter = 0

    for image_path in tqdm(image_path_list):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        hand_landmarks = hand_detector.get_landmarks(image)
        lips_landmarks = lips_detector.get_landmarks(image)

        # TODO: add random number of cigarettes to one image

        for bodypart_landmarks in [hand_landmarks, lips_landmarks]:  # For each group[hands, lips]

            if bodypart_landmarks is None:
                continue

            # Repeat num_repeats with random landmark
            for _ in range(args.num_repeats):
                for bodypart, scale, to_flip_list in zip(bodypart_landmarks['landmarks'],
                                                         bodypart_landmarks['scale_factors'],
                                                         bodypart_landmarks['todo_flip']):
                    # For each hand/face draw cigarette once at random landmark point
                    landmark_index = random.randint(0, len(bodypart) - 1)
                    landmark = bodypart[landmark_index]
                    to_flip = to_flip_list[landmark_index]

                    scale *= np.sqrt(w * h / 640 / 480)
                    if scale < 0.1:  # Skip if hand/face is too small
                        continue

                    cigaretted_image, bbox = cigarette_adder.add_cigarette(image, landmark, scale, to_flip)
                    if cigaretted_image is None:
                        continue

                    cigaretted_image = cv2.cvtColor(cigaretted_image, cv2.COLOR_RGB2BGR)
                    filename = os.path.join(args.save_path, 'train', f'{image_counter}.jpg')
                    cv2.imwrite(filename, cigaretted_image)

                    coco_image = CocoImage(file_name=os.path.join('train', f'{image_counter}.jpg'), height=h, width=w)
                    xywh_box = xyxy2xywh(bbox)

                    coco_image.add_annotation(
                        CocoAnnotation(
                            bbox=xywh_box,
                            category_id=0,
                            category_name='cigarette'
                        )
                    )
                    coco.add_image(coco_image)
                    image_counter += 1

    # save annotation file
    coco_json = coco.json
    save_json(data=coco_json, save_path=os.path.join(args.save_path, 'train.json'))
