import cv2
import dlib
import numpy as np
from math import atan2, degrees


def get_face_landmarks(image_path, predictor_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    faces = detector(gray)

    landmarks_list = []
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(81)]
        landmarks_list.append(landmarks)
    return landmarks_list


def calculate_head_pose(landmarks):
    eye_left = landmarks[45]
    eye_right = landmarks[36]

    dx = eye_right[0] - eye_left[0]
    dy = eye_right[1] - eye_left[1]

    angle = degrees(atan2(dy, dx))
    return angle


def overlay_glasses(person_image, glasses_image, landmarks_list):
    person = cv2.imread(person_image)

    glasses = cv2.imread(glasses_image, cv2.IMREAD_UNCHANGED)

    for landmarks in landmarks_list:
        eye_left = landmarks[36:39]
        eye_right = landmarks[42:45]

        eye_left_center = np.mean(eye_left, axis=0).astype(int)
        eye_right_center = np.mean(eye_right, axis=0).astype(int)
        glasses_width = int(np.linalg.norm(eye_left_center - eye_right_center) * 2.4)
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

        eye_left_center = np.mean(eye_left, axis=0).astype(int)
        eye_right_center = np.mean(eye_right, axis=0).astype(int)
        x_offset = int((eye_left_center[0] + eye_right_center[0]) / 2) - int(glasses_width / 2.2)
        y_offset = int((eye_left_center[1] + eye_right_center[1]) / 2) - int(glasses_height / 2.1)

        head_pose_angle = calculate_head_pose(landmarks)
        rotated_glasses = cv2.rotate(glasses_resized, cv2.ROTATE_180)

        M = cv2.getRotationMatrix2D((rotated_glasses.shape[1] / 2, rotated_glasses.shape[0] / 2), -head_pose_angle, 1)
        rotated_glasses = cv2.warpAffine(rotated_glasses, M, (rotated_glasses.shape[1], rotated_glasses.shape[0]))

        for i in range(rotated_glasses.shape[0]):
            for j in range(rotated_glasses.shape[1]):
                if rotated_glasses[i, j, 3] != 0:
                    person[y_offset + i, x_offset + j, :3] = rotated_glasses[i, j, :3]

    return person


if __name__ == "__main__":
    person_image_path = "samples_photos/three_female.png"
    glasses_image_path = "samples_photos/glassess.png"
    predictor_path = "C:/Users/AdeelCyber/Downloads/shape_predictor_81_face_landmarks.dat"

    landmarks_list = get_face_landmarks(person_image_path, predictor_path)

    if landmarks_list:
        result = overlay_glasses(person_image_path, glasses_image_path, landmarks_list)
        cv2.imshow("Result", result)
        cv2.imwrite('edited_pics/tom_cruise_glasses.png', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")
