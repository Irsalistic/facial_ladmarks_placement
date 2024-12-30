import cv2 as cv
import numpy as np

# Read the image
img = cv.imread('samples_photos/tom_cruise.png')

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply a threshold to create a binary mask
_, mask = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

# Display the binary mask
cv.imshow('Binary Mask', mask)
cv.waitKey(0)
cv.destroyAllWindows()
