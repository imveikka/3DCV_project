import numpy as np
import cv2 as cv
from PIL import Image


def load_img(path):
    img = Image.open(path)
    w, h = img.size
    new_size = (w // 4, h // 4)
    img.thumbnail(new_size)
    return img


def select_points(img):

    """
    Function to select points from an image.
    
    If you selected all the points you wanted, press ESC or close the window.
    """

    points = []

    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    Y = img.shape[0] - 1

    # load reference
    reference = load_img('./data/CALIB_ALEX_VEIKKA_SETUP.jpg')
    reference = np.array(reference)
    reference = cv.cvtColor(reference, cv.COLOR_BGR2RGB)

    # clip reference
    reference = reference[:, 350:]
    reference = reference[:, :650]
    
    img = np.concat((img, reference), 1)

    def click_event(event, x, y, flags, param):
        nonlocal points
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(param, (x, y), 4, (255, 0, 255), -1)
            cv.imshow("Image", param)
            points.append((x, Y - y))

    cv.imshow("Image", img)
    cv.setMouseCallback("Image", click_event, img)

    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:  # ESC key to break
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:  # Check if window is closed
            break
    cv.destroyAllWindows()

    return np.stack(points)


def calibrate(points2d, points3d, normalize=False):
    # Implement a direct linear transformation (DLT) algorithm to calibrate the camera.

    # Convert the 2D & 3D points to homogeneous coordinates
    points2d = np.hstack((points2d, np.ones((points2d.shape[0], 1))))
    points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))

    if normalize:
        # Nomalize the 2D points
        x_mean = np.mean(points2d[:, 0])
        y_mean = np.mean(points2d[:, 1])
        d_bar = np.mean(
            np.sqrt((points2d[:, 0] - x_mean) ** 2 + (points2d[:, 1] - y_mean) ** 2)
        )
        S = np.sqrt(2) / d_bar
        T = np.array([[S, 0, -S * x_mean], [0, S, -S * y_mean], [0, 0, 1]])
        points2d = np.dot(T, points2d.T).T

        # Normalize the 3D points
        x_mean = np.mean(points3d[:, 0])
        y_mean = np.mean(points3d[:, 1])
        z_mean = np.mean(points3d[:, 2])
        d_bar = np.mean(np.sqrt((points3d[:, 0] - x_mean) ** 2 + (points3d[:, 1] - y_mean) ** 2 + (points3d[:, 2] - z_mean) ** 2))
        S = np.sqrt(3) / d_bar
        U = np.array(
            [
                [S, 0, 0, -S * x_mean],
                [0, S, 0, -S * y_mean],
                [0, 0, S, -S * z_mean],
                [0, 0, 0, 1],
            ]
        )
        points3d = np.dot(U, points3d.T).T

    # Create the matrix A
    A = np.zeros((2 * points2d.shape[0], 12))
    for i in range(points2d.shape[0]):
        A[2 * i, 0:4] = points3d[i, :]
        A[2 * i, 8:12] = -points2d[i, 0] * points3d[i, :]
        A[2 * i + 1, 4:8] = points3d[i, :]
        A[2 * i + 1, 8:12] = -points2d[i, 1] * points3d[i, :]

    # Solve the linear system of equations Am = 0
    _, _, V = np.linalg.svd(A)
    m = V[-1, :].reshape((3, 4))

    if normalize:
        # Denormalize the camera matrix
        m = np.linalg.inv(T) @ m @ U
    return m