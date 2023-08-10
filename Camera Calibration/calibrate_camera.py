import numpy as np 
import cv2 as cv
import glob

chessboardSize = (8, 6) 
frameSize = (1280, 720)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv. TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Define the size of each square in meters or any desired unit
square_size = 0.025  # Replace with the actual size of each square
objp *= square_size

objPoints = []
imgPoints = [] # 2d points in image plane.
images = glob.glob('calibration_images/*.jpg')

for image in images:
    # print(image)
    img = cv.imread(image)
    frameSize = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
    # If corners are found, refine and process them
    if ret:
        # print("into loop")
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(1000)

# cv.destroyAllWindows()

# Camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

# Print and save camera matrix and distortion coefficients
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(distortion_coeffs)

np.save('camera_matrix.npy', camera_matrix)
np.save('distortion_coeffs.npy', distortion_coeffs)
