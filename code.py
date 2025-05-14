# Air Canvas using OpenCV and Mediapipe

import cv2              # Importing OpenCV library for computer vision tasks
import numpy as np      # Importing NumPy for numerical operations
from collections import deque  # Importing deque from collections to handle drawing points

# Initialize deques to store points for different colors
bpoints = [deque(maxlen=1024)]  # Blue points
gpoints = [deque(maxlen=1024)]  # Green points
rpoints = [deque(maxlen=1024)]  # Red points
ypoints = [deque(maxlen=1024)]  # Yellow points

# Indexes to keep track of points in different color arrays
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for image dilation
kernel = np.ones((5, 5), np.uint8)

# List of colors in BGR format
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default color index

# Set up the canvas window with color buttons
paintWindow = np.zeros((471, 636, 3)) + 255  # White canvas
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue button
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)  # Green button
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)  # Red button
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow button

# Adding text labels on the canvas buttons
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Creating a window named 'Paint'
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural (mirror-like) interaction
    frame = cv2.flip(frame, 1)
    
    # Apply background subtraction
    fgmask = bg_subtractor.apply(frame)
    
    # Apply some morphological operations to clean up the mask
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw color buttons on the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Process the largest contour (assuming it's the hand)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only process if the contour is large enough
        if cv2.contourArea(largest_contour) > 1000:
            # Get the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                center = (center_x, center_y)
                
                # Draw a circle at the center
                cv2.circle(frame, center, 3, (0, 255, 0), -1)
                
                # Check if the center is in the button area
                if center[1] <= 65:
                    if 40 <= center[0] <= 140:  # Clear Button
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0
                        paintWindow[67:, :, :] = 255
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0  # Blue
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1  # Green
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2  # Red
                    elif 505 <= center[0] <= 600:
                        colorIndex = 3  # Yellow
                else:
                    # Append points to the corresponding deque based on selected color
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Display the frame and paint window
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
