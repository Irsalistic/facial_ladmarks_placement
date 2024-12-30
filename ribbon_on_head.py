import cv2
import dlib
import math


def rotate_image(image, angle):
    center = tuple(map(lambda x: x // 2, image.shape[:2][::-1]))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    return result


# face landmark model of dlib 81
predictor_path = "C:/Users/AdeelCyber/Downloads/shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ribbon_image
hat_path = "samples_photos/rebear1.png"
hat = cv2.imread(hat_path, -1)

# person_image
img_path = "samples_photos/messi.png"
actual_imag = cv2.imread(img_path)

# Step 1: Detect faces in the image
faces = detector(actual_imag)

for face in faces:
    # Step 2: Detect the landmark points in the face
    landmarks = predictor(actual_imag, face)

    # head bounding box coordinates
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # head position based on landmarks
    head_position = (landmarks.part(71).x, landmarks.part(71).y)

    # Estimate head pose using other landmarks here we are targeting the eyes
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # angle of head pose
    angle = -math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Hat placement process
    w_hat = int(w * 2)
    h_hat = int(w_hat / 3)
    hatResize = cv2.resize(hat, (w_hat, h_hat))
    hatOriginal = hatResize[:, :, 0:3]
    maskHat = hatResize[:, :, 3]

    y1 = int(head_position[1] - h_hat)
    y2 = (y1 + h_hat)
    x1 = int(head_position[0] - w_hat / 2)
    x2 = x1 + w_hat

    # Adjust hat placement if it goes beyond image boundaries
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if x2 > actual_imag.shape[1]:
        x2 = actual_imag.shape[1]

    hatROI = actual_imag[y1:y2, x1:x2]

    # Resize the hat and mask to match the ROI dimensions
    hatOriginal = cv2.resize(hatOriginal, (x2-x1,y2-y1))
    maskHat = cv2.resize(maskHat, (x2 - x1, y2 - y1))

    # Rotate the hat based on the estimated head pose
    hatOriginal = rotate_image(hatOriginal, angle)
    maskHat = rotate_image(maskHat, angle)

    # Create masked images
    maskedHatImage = cv2.merge((maskHat, maskHat, maskHat))
    hatImage = cv2.bitwise_and(hatOriginal, maskedHatImage)
    hatROIImage = cv2.bitwise_and(hatROI, cv2.bitwise_not(maskedHatImage))

    # Combine the hat with the original image
    finalHat = cv2.bitwise_or(hatROIImage, hatImage)
    actual_imag[y1:y2, x1:x2] = finalHat

    # Display the image with the hat
    cv2.imshow("Image with Hat", actual_imag)
    cv2.imwrite('edited_pics/messi_with_ribbon.png',actual_imag)

# Wait for a key press and then close all OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()
