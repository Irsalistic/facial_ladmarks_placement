import cv2
import mediapipe as mp
import numpy as np
import os

os.environ['DLIB_USE_CUDA'] = '0'


def fill_eyebrows_with_templates(image_path, left_eyebrow_template_path, right_eyebrow_template_path):
    # Load the image
    img = cv2.imread(image_path)
    left_eyebrow = cv2.imread(left_eyebrow_template_path, cv2.IMREAD_UNCHANGED)
    right_eyebrow = cv2.imread(right_eyebrow_template_path, cv2.IMREAD_UNCHANGED)

    # Extraction of points for the left and right eyebrows using provided indices
    left_eyebrow_points = [46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107]
    right_eyebrow_points = [276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336]

    # left_eyebrow_inpaint_points = [70, 156, 156, 124, 124, 225, 225, 30, 30, 29, 29, 27, 27, 222, 222, 28, 28,
    #                                56, 56, 221, 9, 9, 70, 63, 63, 105, 105, 66, 66, 107, 107, 9]

    left_eyebrow_inpaint_points = [70, 156, 156, 139, 139, 124, 124, 113, 113, 225, 225, 224, 224, 223, 223,
                                   222, 222, 221, 9, 9, 70, 63, 63, 105, 105, 66, 66, 107, 107, 9]

    # right_eyebrow_inpaint_points = [300, 383, 383, 276, 276, 445, 445, 444, 444, 259, 259, 257, 257, 258, 258,
    #                                 441, 441, 285, 285, 9, 300, 293, 293, 334, 334, 296, 296, 336, 336, 9]

    right_eyebrow_inpaint_points = [300, 383, 353, 353, 342, 342, 445, 444, 444, 443, 443, 442, 442,
                                    441, 441, 413, 413, 417, 417, 8, 8, 9, 300, 293, 293, 334, 334, 296, 296, 336, 336,
                                    9]

    # Initialize Mediapipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1)

    # Find face locations and facial landmarks using Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            img_height, img_width, img_channel = img.shape

            left_eyebrow_pts = np.array(
                [(int(landmarks.landmark[i].x * img_width), int(landmarks.landmark[i].y * img_height)) for i
                 in left_eyebrow_points], np.int32)
            # print(left_eyebrow_pts)

            right_eyebrow_pts = np.array(
                [(int(landmarks.landmark[i].x * img_width), int(landmarks.landmark[i].y * img_height)) for i
                 in right_eyebrow_points], np.int32)
            # print(right_eyebrow_pts)
            left_eyebrow_inpaint_pts = np.array(
                [(int(landmarks.landmark[i].x * img_width), int(landmarks.landmark[i].y * img_height)) for i
                 in left_eyebrow_inpaint_points], np.int32)

            right_eyebrow_inpaint_pts = np.array(
                [(int(landmarks.landmark[i].x * img_width), int(landmarks.landmark[i].y * img_height)) for i
                 in right_eyebrow_inpaint_points], np.int32)

            # Calculate the bounding boxes for the left and  right eyebrows
            left_eyebrow_rect = cv2.boundingRect(left_eyebrow_pts)
            right_eyebrow_rect = cv2.boundingRect(right_eyebrow_pts)
            lx, ly, lw, lh = left_eyebrow_rect[0], left_eyebrow_rect[1], left_eyebrow_rect[2], left_eyebrow_rect[3]
            rx, ry, rw, rh = right_eyebrow_rect[0], right_eyebrow_rect[1], right_eyebrow_rect[2], right_eyebrow_rect[3]

            # Use in paint to remove eyebrows
            inpaint_mask = np.zeros_like(img, dtype=np.uint8)
            cv2.fillPoly(inpaint_mask, [left_eyebrow_inpaint_pts], (255, 255, 255))
            cv2.fillPoly(inpaint_mask, [right_eyebrow_inpaint_pts], (255, 255, 255))

            img = cv2.inpaint(img, inpaint_mask[:, :, 0], 50, cv2.INPAINT_TELEA)
            cv2.imshow('before smoothing', img)

            # Smoothing the left eyebrow region
            left_eyebrow_smoothed = cv2.GaussianBlur(img[ly:ly + lh, lx:lx + lw], (3, 3), 0)

            # Smoothing the right eyebrow region
            right_eyebrow_smoothed = cv2.GaussianBlur(img[ry:ry + rh, rx:rx + rw], (3, 3), 0)

            # Replacing the inpainted eyebrow regions with the smoothed versions
            img[ly:ly + lh, lx:lx + lw] = left_eyebrow_smoothed

            img[ry:ry + rh, rx:rx + rw] = right_eyebrow_smoothed
            cv2.imshow('after smoothing', img)
            # Resize the eyebrow templates
            resized_left_eyebrow_template = cv2.resize(left_eyebrow, (lw, lh), interpolation=cv2.INTER_AREA, )

            resized_right_eyebrow_template = cv2.resize(right_eyebrow, (rw, rh), interpolation=cv2.INTER_AREA, )

            # Get the regions of interest (ROIs) for the eyebrows in the inpainted image
            left_roi = img[ly:ly + lh, lx:lx + lw]
            right_roi = img[ry:ry + rh, rx:rx + rw]

            # Putting templates on the roi area
            alpha_channel_left = resized_left_eyebrow_template[:, :, 3] / 255.0
            img_alpha_left = left_roi * (1 - alpha_channel_left[:, :, np.newaxis])
            resized_left_eyebrow = resized_left_eyebrow_template[:, :, :3] * (
                alpha_channel_left[:, :, np.newaxis]
            )
            roi_with_left_eyebrow = img_alpha_left + resized_left_eyebrow
            #
            alpha_channel_right = resized_right_eyebrow_template[:, :, 3] / 255.0
            img_alpha_right = right_roi * (1 - alpha_channel_right[:, :, np.newaxis])
            resized_right_eyebrow = resized_right_eyebrow_template[:, :, :3] * (
                alpha_channel_right[:, :, np.newaxis]
            )
            roi_with_right_eyebrow = img_alpha_right + resized_right_eyebrow

            #  Update the original inpainted image with the modified eyebrow regions
            img[ly:ly + lh, lx:lx + lw] = roi_with_left_eyebrow
            img[ry:ry + rh, rx:rx + rw] = roi_with_right_eyebrow

            left_centre = lw // 2, lh // 2
            right_centre = rw // 2, rh // 2

            # Create masks for the resized left and right eyebrow templates
            # first converting them to gray mode
            left_gray = cv2.cvtColor(resized_left_eyebrow_template, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(resized_right_eyebrow_template, cv2.COLOR_BGR2GRAY)

            # Now Creating a black and white mask using thresh hold binary
            _, left_mask = cv2.threshold(left_gray, 128, 255, cv2.THRESH_BINARY)
            _, right_mask = cv2.threshold(right_gray, 128, 255, cv2.THRESH_BINARY)

            # Performing seamless cloning with masks
            left_cloned = cv2.seamlessClone(resized_left_eyebrow_template, img, left_mask, left_centre,
                                            cv2.NORMAL_CLONE)
            right_cloned = cv2.seamlessClone(resized_right_eyebrow_template, img, right_mask, right_centre,
                                             cv2.NORMAL_CLONE)

            # Blending the cloned eyebrows with the original image using masks
            img[ly:ly + lh, lx:lx + lw] = ((1 - left_mask[:, :, np.newaxis]) * img[ly:ly + lh, lx:lx + lw] +
                                           left_mask[:, :, np.newaxis] * left_cloned[ly:ly + lh, lx:lx + lw])
            img[ry:ry + rh, rx:rx + rw] = ((1 - right_mask[:, :, np.newaxis]) * img[ry:ry + rh, rx:rx + rw] +
                                           right_mask[:, :, np.newaxis] * right_cloned[ry:ry + rh, rx:rx + rw])

    # Display the result
    cv2.imshow('img', img)
    # cv2.imwrite('edited_pics/tom_cruise_eyebrows_without_cloning.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'samples_photos/tom_cruise.png'
left_eyebrow_path = 'samples_photos/right1-eyebrow.png'
right_eyebrow_path = 'samples_photos/left1-eyebrow.png'

# passing the paths to the function
fill_eyebrows_with_templates(image_path, left_eyebrow_path, right_eyebrow_path)

# fill_eyebrows_with_templates('samples_photos/tom_cruise.png', 'eyebrows/natural_left.png'
#                              , 'eyebrows/natural_right.png')
