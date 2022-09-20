import os
import random
from typing import List, Tuple, Union, Dict

import cv2
import mediapipe
import numpy as np

from generation_utils import rotate_cigarette, overlay_image_alpha


class HandDetector:

    def __init__(self):
        self.hands = mediapipe.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        # Landmarks of corresponding index and middle fingers points
        self.index_landmarks = [6, 7]
        self.middle_landmarks = [10, 11]
        self.index_knuckle_index = 5
        self.middle_knuckle_index = 9

    def get_landmarks(self, image: np.ndarray):
        h, w, c = image.shape
        results = self.hands.process(image)

        if not results.multi_hand_landmarks:
            return None

        output_dict = {}
        result_landmarks = []
        scale_factors = []
        to_flip_list = []

        for hand_landmarks in results.multi_hand_landmarks:

            # Compute distance between knuckles to understand if fingers are too far away to hold a cigarette
            index_knuckle = hand_landmarks.landmark[self.index_knuckle_index]
            index_tip = hand_landmarks.landmark[8]
            middle_knuckle = hand_landmarks.landmark[self.middle_knuckle_index]
            knuckle_distance = abs(index_knuckle.x - middle_knuckle.x) + abs(index_knuckle.y - middle_knuckle.y)
            # P.S. Pretty dumb way to compute hand scale, tilted hand will be considered smaller than it is
            # TODO: change to smth 3d
            scale_factor = knuckle_distance * 15

            # Compute simple hand direction, flip cigarette if hand aims left
            # TODO: compute adequate hand direction
            to_flip = not (index_tip.x - index_knuckle.x) > 0

            current_hand_landmarks = []
            current_toflip_list = []

            for index_index, middle_index in zip(self.index_landmarks, self.middle_landmarks):

                index_point = hand_landmarks.landmark[index_index]
                middle_point = hand_landmarks.landmark[middle_index]
                result_x = int((index_point.x + middle_point.x) / 2 * w)
                result_y = int((index_point.y + middle_point.y) / 2 * h)

                l1_dist = (abs(index_point.x - middle_point.x) + abs(index_point.y - middle_point.y))

                # Sophisticated thresholding constant
                if l1_dist < knuckle_distance * 1.5:
                    current_hand_landmarks.append((result_x, result_y))
                    current_toflip_list.append(to_flip)

            if current_hand_landmarks:
                result_landmarks.append(current_hand_landmarks)
                # In order to scale cigarette to hand size
                scale_factors.append(scale_factor)
                to_flip_list.append(current_toflip_list)

        if not result_landmarks:
            return None

        output_dict['landmarks'] = result_landmarks
        output_dict['scale_factors'] = scale_factors
        output_dict['todo_flip'] = to_flip_list

        return output_dict


class LipsDetector:

    def __init__(self, check_face_direction=False):
        self.face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # Landmarks of corresponding upper and lower lip points
        self.upper_lip_landmarks = [82, 81, 80, 191, 312, 311, 310, 415]
        self.lower_lip_landmarks = [87, 178, 88, 95, 317, 402, 318, 324]
        self.check_face_direction = check_face_direction

    def get_landmarks(self, image):

        h, w, c = image.shape
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        output_dict = {}
        result_landmarks = []
        scale_factors = []
        to_flip_list = []

        for face_landmarks in results.multi_face_landmarks:

            # Compute face height to image to do relative check if mouth is opened too wide
            uppest_point = face_landmarks.landmark[10]
            lowest_point = face_landmarks.landmark[152]
            face_height = abs(uppest_point.x - lowest_point.x) + abs(uppest_point.y - lowest_point.y)
            scale_factor = face_height * 1.5

            current_face_landmarks = []
            current_flip_list = []

            for upper_lip_index, lower_lip_index in zip(self.upper_lip_landmarks, self.lower_lip_landmarks):

                upper_point = face_landmarks.landmark[upper_lip_index]
                lower_point = face_landmarks.landmark[lower_lip_index]
                result_x = int((upper_point.x + lower_point.x) / 2 * w)
                result_y = int((upper_point.y + lower_point.y) / 2 * h)

                l1_dist = (abs(upper_point.x - lower_point.x) + abs(upper_point.y - lower_point.y)) / face_height

                # Sophisticated thresholding constant
                if l1_dist < 0.02:
                    current_face_landmarks.append((result_x, result_y))
                    # Workaround to adjust cigarette direction, should be swapped for face direction computing
                    # Flip cigarette if landmark is on left side of face
                    to_flip = True if upper_lip_index in (82, 81, 80, 191) else False
                    current_flip_list.append(to_flip)

            if current_face_landmarks:
                result_landmarks.append(current_face_landmarks)
                scale_factors.append(scale_factor)
                to_flip_list.append(current_flip_list)

        if not result_landmarks:
            return None

        output_dict['landmarks'] = result_landmarks
        output_dict['scale_factors'] = scale_factors
        output_dict['todo_flip'] = to_flip_list

        if self.check_face_direction:
            # TODO: compute face direction and add to output dict xyz angle and adapt cigarette direction
            pass

        return output_dict


class CigaretteAdder:

    def __init__(self, cigarette_path='cigarette_dataset'):
        self.cigarettes = []
        for image_path in os.listdir(cigarette_path):
            cig_image = cv2.imread(os.path.join(cigarette_path, image_path), cv2.IMREAD_UNCHANGED)
            cig_image = cv2.cvtColor(cig_image, cv2.COLOR_BGRA2RGBA)
            cig_image = cv2.resize(cig_image, (150, 15))
            self.cigarettes.append(cig_image)

        self.rotate_p = 1
        self.rotate_limit = 45
        self.flip_p = 0.5

    def add_cigarette(self, image: np.ndarray, landmark: Tuple[int, int], scale: float, toflip: bool):

        random_cigarette = random.choice(self.cigarettes)
        cig_info = self.transform_cigarette(random_cigarette, scale, toflip)
        overlayed_image, bbox = self.overlay_cigarette(image, landmark, cig_info)

        if overlayed_image is None:
            return None, None

        return overlayed_image, bbox

    def transform_cigarette(self, image: np.ndarray, scale: float, toflip: bool):

        h, w, _ = image.shape
        transformed_image = image.copy()
        bias = list(map(lambda x: int(x*scale), [5, 7]))
        flipped_flag = False

        transformed_image = cv2.resize(transformed_image, dsize=None, fx=scale, fy=scale)

        if random.random() < self.rotate_p:
            transformed_image, rotated_bias = rotate_cigarette(transformed_image, self.rotate_limit, bias)
            bias[1] = rotated_bias[1]

        if toflip or (toflip is None and random.random() < self.flip_p):  # Flip according to keypoint direction
            transformed_image = cv2.flip(transformed_image, 1)
            flipped_flag = True

        # TODO NUMBER ONE: add perspective transform

        return {
            'image': transformed_image,
            'attachment_bias': bias,
            'flipped': flipped_flag,
        }

    @staticmethod
    def overlay_cigarette(image: np.ndarray, landmark: Tuple[int, int], cigarette: Dict[str, Union[np.ndarray, int, bool]]):

        current_landmark = list(landmark)
        h, w, _ = image.shape
        cig_h, cig_w, _ = cigarette['image'].shape

        # move attachment x point if cigarette was flipped
        if cigarette['flipped']:
            current_landmark[0] -= cig_w

        # move attachment y point according to rotation
        current_landmark[1] -= cigarette['attachment_bias'][1]

        overlayed_image = overlay_image_alpha(image, cigarette['image'], current_landmark)

        bbox = [max(0, current_landmark[0]),
                max(0, current_landmark[1]),
                min(w, (current_landmark[0] + cig_w)),
                min(h, (current_landmark[1] + cig_h))]

        bbox = np.array(bbox)

        return overlayed_image, bbox
