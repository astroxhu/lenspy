import cv2
import numpy as np

# Load the image
image = cv2.imread('rf5018_6.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to make it binary
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the image
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on their area (you would need to determine a suitable threshold)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

# Calculate the diameter of each contour
diameters = [np.sqrt(cv2.contourArea(cnt) / np.pi) * 2 for cnt in contours]

print(diameters)
