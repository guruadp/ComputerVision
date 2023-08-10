import cv2
import numpy as np

# Load camera parameters obtained from calibration

# camera_matrix = np.load('camera_matrix.npy')
# dist_coeffs = np.load('distortion_coeffs.npy')

camera_matrix = np.array([[947.12934065, 0, 631.00626976],   # Focal length (fx, fy) and principal point (cx, cy)
                          [0, 954.34655281, 371.80479122],   # (cx, cy) is the center of the image
                          [0, 0, 1]])

# Example distortion coefficients
dist_coeffs = np.array([-0.17328115, 0.94817042, 0.01218803, -0.0018911, -1.26428073])  # k1, k2, p1, p2, k3q

# Load the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

# Create the detector parameters
parameters = cv2.aruco.DetectorParameters_create()

# Open the camera
cap = cv2.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Calculate the distance to the marker
            distance = tvecs[i][0][2]  # Z-coordinate of the marker's position

            print(f"ArUco ID {ids[i][0]} - Distance: {distance} units")

        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('ArUco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()