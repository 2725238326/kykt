import glob
from pathlib import Path

import numpy as np
import cv2 as cv

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,6,0)
# 修改1：这里按你现在这块棋盘的内角点数写成 (6,7)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# 修改2：原代码是 glob.glob('*.jpg')，现在图片在 Picture 文件夹里
script_dir = Path(__file__).resolve().parent
images = sorted(script_dir.glob("Picture/*.jpg"), key=lambda p: int(p.stem))

gray = None
used_images = []

for fname in images:
    img = cv.imread(str(fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    # 修改3：这里和 objp 同步，改成 (6,7)
    ret, corners = cv.findChessboardCorners(gray, (6, 7), None)

    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        used_images.append(fname)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (6, 7), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(200)
        print(f"detected: {fname.name}")
    else:
        print(f"failed: {fname.name}")

cv.destroyAllWindows()

if not objpoints:
    raise RuntimeError("No chessboard corners were detected. Please check pattern size or images.")

# 修改4：老师说要写一个 K，这里直接把 camera matrix 记成 K
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nCalibration finished.")
print(f"valid images: {len(used_images)}")
print(f"RMS reprojection error: {ret}")
print("\nK =")
print(K)
print("\ndist =")
print(dist)

# 修改5：增加一段畸变矫正，方便直接输出结果图
img = cv.imread(str(used_images[0]))
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, K, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
if w > 0 and h > 0:
    dst = dst[y : y + h, x : x + w]

result_path = script_dir / "calibresult.png"
cv.imwrite(str(result_path), dst)
print(f"\nundistorted image saved to: {result_path}")
print(f"undistorted source image: {used_images[0].name}")

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))
