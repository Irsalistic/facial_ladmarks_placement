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


def overlay_glasses(person, glasses_image, landmarks_list):
    # person = cv2.imread(person)

    glasses = cv2.imread(glasses_image, cv2.IMREAD_UNCHANGED)

    for landmarks in landmarks_list:
        eye_left = landmarks[36:39]
        eye_right = landmarks[42:45]

        eye_left_center = np.mean(eye_left, axis=0).astype(int)
        eye_right_center = np.mean(eye_right, axis=0).astype(int)
        glasses_width = int(np.linalg.norm(eye_left_center - eye_right_center) * 2.3)
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

        eye_left_center = np.mean(eye_left, axis=0).astype(int)
        eye_right_center = np.mean(eye_right, axis=0).astype(int)
        x_offset = int((eye_left_center[0] + eye_right_center[0]) / 2) - int(glasses_width / 2.15)
        y_offset = int((eye_left_center[1] + eye_right_center[1]) / 2) - int(glasses_height / 3)

        head_pose_angle = calculate_head_pose(landmarks)
        rotated_glasses = cv2.rotate(glasses_resized, cv2.ROTATE_180)

        M = cv2.getRotationMatrix2D((rotated_glasses.shape[1] / 2, rotated_glasses.shape[0] / 2), -head_pose_angle, 1)
        rotated_glasses = cv2.warpAffine(rotated_glasses, M, (rotated_glasses.shape[1], rotated_glasses.shape[0]))

        for i in range(rotated_glasses.shape[0]):
            for j in range(rotated_glasses.shape[1]):
                if rotated_glasses[i, j, 3] != 0:
                    person[y_offset + i, x_offset + j, :3] = rotated_glasses[i, j, :3]

    return person


def overlay_nose(person, nose_image, landmarks):
    nose = cv2.imread(nose_image, cv2.IMREAD_UNCHANGED)

    # Dynamically select nose landmarks based on face shape
    nose_points = landmarks[27:36]
    nose_center = np.mean(nose_points, axis=0).astype(int)

    # Accessing individual tuple elements for subtraction
    nose_width = int(np.linalg.norm(np.array(nose_points[0]) - np.array(nose_points[8])) * 1.45)
    nose_height = int(nose_width * nose.shape[0] / nose.shape[1])

    nose_resized = cv2.resize(nose, (nose_width, nose_height))

    # Adjust nose offset based on head pose
    head_pose_angle = calculate_head_pose(landmarks)
    rotated_nose = cv2.rotate(nose_resized, cv2.ROTATE_180)
    M_nose = cv2.getRotationMatrix2D((rotated_nose.shape[1] / 2, rotated_nose.shape[0] / 2), -head_pose_angle, 1)
    rotated_nose = cv2.warpAffine(rotated_nose, M_nose, (rotated_nose.shape[1], rotated_nose.shape[0]))

    nose_offset_x = int(nose_center[0] - nose_width / 2.1)
    nose_offset_y = int(nose_center[1] - nose_height / 1.8)

    for i in range(rotated_nose.shape[0]):
        for j in range(rotated_nose.shape[1]):
            if rotated_nose[i, j, 3] != 0:
                person[nose_offset_y + i, nose_offset_x + j, :3] = rotated_nose[i, j, :3]

    return person


if __name__ == "__main__":
    person_image_path = "samples_photos/images-18.png"
    glasses_image_path = "nose_and_moustache/glassess.png"
    nose_image_path = "nose_and_moustache/nose_mustache.png"
    predictor_path = "C:/Users/AdeelCyber/Downloads/shape_predictor_81_face_landmarks.dat"

    landmarks_list = get_face_landmarks(person_image_path, predictor_path)

    if landmarks_list:
        person = cv2.imread(person_image_path)
        result_with_nose = overlay_nose(person.copy(), nose_image_path, landmarks_list[0])
        # cv2.imshow('image with just nose', result_with_nose)

        result_with_glassess = overlay_glasses(result_with_nose, glasses_image_path, landmarks_list)
        cv2.imshow("Result with glassess", result_with_glassess)
        # cv2.imwrite('edited_pics/tomcruise_with_moustache.png', result_with_glassess)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")

# if __name__ == "__main__":
#     person_image_path = "samples_photos/mrbean.png"
#     glasses_image_path = "samples_photos/glasses_moutache.png"
#     nose_image_path = "nose_and_moustache/nose.png"
#     mustache_image_path = "nose_and_moustache/mustache.png"
#     predictor_path = "C:/Users/AdeelCyber/Downloads/shape_predictor_81_face_landmarks.dat"
#
#     landmarks_list = get_face_landmarks(person_image_path, predictor_path)
#
#     if landmarks_list:
#         result = overlay_glasses(person_image_path, glasses_image_path, landmarks_list)
#         cv2.imshow("Result", result)
#         # cv2.imwrite('edited_pics/driver_with_glasses.png', result)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("No face detected in the image.")
