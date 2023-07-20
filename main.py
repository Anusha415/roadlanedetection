import cv2
import numpy as np

def detect_lines(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Perform Hough line transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    
    # Draw detected lines on the image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine the original image with the line image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return result

# Read the input image
image = cv2.imread('road_image.jpg')

# Detect lines
result = detect_lines(image)

# Display the result
cv2.imshow('Road Lines Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
