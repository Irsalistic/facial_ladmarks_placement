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

    # Initialize Mediapipe Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4, min_detection_confidence=0.1)

    # Find face locations and facial landmarks using Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Extraction of points for the left and right eyebrows using provided indices
            left_eyebrow_pts = np.array(
                [(int(landmarks.landmark[idx].x * img.shape[1]), int(landmarks.landmark[idx].y * img.shape[0])) for idx
                 in
                 [46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107]], np.int32)

            right_eyebrow_pts = np.array(
                [(int(landmarks.landmark[idx].x * img.shape[1]), int(landmarks.landmark[idx].y * img.shape[0])) for idx
                 in
                 [276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336]], np.int32)

            # Calculate the bounding boxes for the left and right eyebrows
            left_eyebrow_rect = cv2.boundingRect(left_eyebrow_pts)
            right_eyebrow_rect = cv2.boundingRect(right_eyebrow_pts)
            lx, ly, lw, lh = left_eyebrow_rect[0], left_eyebrow_rect[1], left_eyebrow_rect[2], left_eyebrow_rect[3]
            rx, ry, rw, rh = right_eyebrow_rect[0], right_eyebrow_rect[1], right_eyebrow_rect[2], right_eyebrow_rect[3]

            # Use inpaint to remove eyebrows
            inpaint_mask = np.zeros_like(img, dtype=np.uint8)
            cv2.fillPoly(inpaint_mask, [left_eyebrow_pts], (255, 255, 255))
            cv2.fillPoly(inpaint_mask, [right_eyebrow_pts], (255, 255, 255))

            img = cv2.inpaint(img, inpaint_mask[:, :, 0], 50, cv2.INPAINT_TELEA)

            # Smoothing the left eyebrow region
            left_eyebrow_smoothed = cv2.GaussianBlur(img[ly:ly + lh, lx:lx + lw], (5, 5), 0)

            # Smoothing the right eyebrow region
            right_eyebrow_smoothed = cv2.GaussianBlur(img[ry:ry + rh, rx:rx + rw], (5, 5), 0)

            # Replace the inpainted eyebrow regions with the smoothed versions
            img[ly:ly + lh, lx:lx + lw] = left_eyebrow_smoothed
            img[ry:ry + rh, rx:rx + rw] = right_eyebrow_smoothed

            # Resize the eyebrow templates
            resized_left_eyebrow_template = cv2.resize(left_eyebrow, (lw, lh), interpolation=cv2.INTER_AREA)
            resized_right_eyebrow_template = cv2.resize(right_eyebrow, (rw, rh), interpolation=cv2.INTER_AREA)

            # Adjust the eyebrow templates for a more realistic appearance
            realistic_left_eyebrow = adjust_eyebrow_template(resized_left_eyebrow_template)
            realistic_right_eyebrow = adjust_eyebrow_template(resized_right_eyebrow_template)

            # Get the regions of interest (ROIs) for the eyebrows in the inpainted image
            left_roi = img[ly:ly + lh, lx:lx + lw]
            right_roi = img[ry:ry + rh, rx:rx + rw]

            # Putting templates on the roi area
            alpha_channel_left = realistic_left_eyebrow[:, :, 3] / 255.0
            img_alpha_left = left_roi * (1 - alpha_channel_left[:, :, np.newaxis])
            resized_left_eyebrow = realistic_left_eyebrow[:, :, :3] * (
                    alpha_channel_left[:, :, np.newaxis]
            )
            roi_with_left_eyebrow = img_alpha_left + resized_left_eyebrow

            alpha_channel_right = realistic_right_eyebrow[:, :, 3] / 255.0
            img_alpha_right = right_roi * (1 - alpha_channel_right[:, :, np.newaxis])
            resized_right_eyebrow = realistic_right_eyebrow[:, :, :3] * (
                    alpha_channel_right[:, :, np.newaxis]
            )
            roi_with_right_eyebrow = img_alpha_right + resized_right_eyebrow

            # Update the original inpainted image with the modified eyebrow regions
            img[ly:ly + lh, lx:lx + lw] = roi_with_left_eyebrow
            img[ry:ry + rh, rx:rx + rw] = roi_with_right_eyebrow

    # Display the result
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adjust_eyebrow_template(eyebrow_template):
    # Apply adjustments to make the eyebrow template more realistic
    # You can implement color matching, texture matching, feathering, etc.

    # Example: Adjusting brightness and contrast
    eyebrow_template = cv2.convertScaleAbs(eyebrow_template, alpha=1.2, beta=20)

    # Example: Applying feathering
    feathering_mask = create_feathering_mask(eyebrow_template)
    eyebrow_template[:, :, 3] = feathering_mask

    return eyebrow_template


def create_feathering_mask(eyebrow_template):
    # Example: Create a feathering mask
    alpha_channel = eyebrow_template[:, :, 3]
    feathering_mask = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
    feathering_mask = np.clip(feathering_mask, 0, 255).astype(np.uint8)
    return feathering_mask


# Example usage
image_path = 'samples_photos/tom_cruise.png'
left_eyebrow_path = 'samples_photos/right1-eyebrow.png'
right_eyebrow_path = 'samples_photos/left1-eyebrow.png'

fill_eyebrows_with_templates(image_path, left_eyebrow_path, right_eyebrow_path)
