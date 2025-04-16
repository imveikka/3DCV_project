import numpy as np
import cv2 as cv
from PIL import Image
from scipy.linalg import rq
from pathlib import Path
from matplotlib.axes import Axes
import skimage


def load_img(path: str | Path) -> Image:
    img = Image.open(path)
    w, h = img.size
    new_size = (w // 4, h // 4)
    img.thumbnail(new_size)
    return img


def select_points(img: Image) -> np.array:

    """
    Function to select points from an image.
    
    If you selected all the points you wanted, press ESC or close the window.
    """

    points = []
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # load reference
    reference = Image.open('./data/CALIB_ALEX_VEIKKA_SETUP.jpg')
    reference = np.array(reference)
    reference = cv.cvtColor(reference, cv.COLOR_RGB2BGR)

    # resize
    img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)
    reference = cv.resize(reference, (0, 0), fx=0.25, fy=0.25)
    
    # clip reference
    reference = reference[:, 350:]
    reference = reference[:, :650]

    # canvas
    img = np.concat((img, reference), 1)

    def click_event(event, x, y, flags, param):
        nonlocal points
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(param, (x, y), 4, (255, 0, 255), -1)
            cv.imshow("Image", param)
            points.append((x, y))

    cv.imshow("Image", img)
    cv.setMouseCallback("Image", click_event, img)

    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:  # ESC key to break
            break
        if cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:  # Check if window is closed
            break
    cv.destroyAllWindows()

    return np.stack(points) * 4


def calibrate(points2d: np.array, points3d: np.array) -> np.array:

    """
    Performs direct linear transform (DLT)
    """

    n = len(points3d)
    biased = np.concat((points3d, np.ones((n, 1))), 1)

    top_left = np.concat((biased, np.zeros((n, 4))), 1)
    bot_left = np.roll(top_left, 4, axis=1)

    top_right = -biased * points2d[:, 0:1]
    bot_right = -biased * points2d[:, 1:2]

    system = np.concat(
        (np.concat((top_left, top_right), 1), 
         np.concat((bot_left, bot_right), 1)), 0
    )

    _, _, Vh = np.linalg.svd(system)
    m = Vh[-1]
    M = m.reshape(3, 4) / m[-1]

    return M


def calibrate_norm(points2d: np.array, points3d: np.array) -> np.array:

    """DLT woth normalization"""

    x_hat, y_hat = points2d.mean(0)
    d_hat = np.linalg.norm(points2d - np.array([x_hat, y_hat]), 1).mean()

    X_hat, Y_hat, Z_hat = points3d.mean(0)
    D_hat = np.linalg.norm(points3d - np.array([X_hat, Y_hat, Z_hat]), 1).mean()

    root2 = np.sqrt(2)
    root3 = np.sqrt(3)

    T = np.array([[root2 / d_hat, 0, -root2 * x_hat / d_hat],
                  [0, root2 / d_hat, -root2 * y_hat / d_hat],
                  [0, 0, 1]])

    U = np.array([[root3 / D_hat, 0, 0, -root3 * X_hat / D_hat],
                  [0, root3 / D_hat, 0, -root3 * Y_hat / D_hat],
                  [0, 0, root3 / D_hat, -root3 * Z_hat / D_hat],
                  [0, 0, 0, 1]]) 

    normalized_2d_pts = points2d * np.diag(T)[:-1] + T[:-1, -1]
    normalized_3d_pts = points3d * np.diag(U)[:-1] + U[:-1, -1]

    M = calibrate(normalized_2d_pts, normalized_3d_pts)
    denormalized_M = np.linalg.inv(T) @ M @ U

    return denormalized_M


def extract_params(M: np.array) -> tuple[np.array]:

    """
    Reference: https://www.youtube.com/watch?v=2XM2Rb2pfyQ
    """

    K, R = rq(M[:, :3])
    t = np.linalg.inv(K) @ M[:, 3:4]

    intrinsic_params = np.pad(K, ((0, 0), (0, 1)))
    extrinsic_params = np.concat((R, t), 1)
    extrinsic_params = np.concat((extrinsic_params, [[0, 0, 0, 1]]))

    return intrinsic_params, extrinsic_params


def plot_frame(extrinsic: np.array, ax: Axes, name: str = "",
               s: int = 10, l: int = 10) -> None:

    """
    Function that plots the world frames from
    extrinsic matrix: 4x4 matrix
    E = [R t]
        [0 1]
    Note: RR^t = I

    Reference: https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
    """

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    center = -R.T @ t
    Ux, Uy, Uz = (R.T @ np.eye(3)).T

    ax.scatter(*center, c='k', s=s, marker='x', label='camera position')

    ax.quiver(*center, *Ux, color="r", length=l,)
    ax.quiver(*center, *Uy, color="g", length=l,)
    ax.quiver(*center, *Uz, color="b", length=l,)

    ax.text(*(center + Ux * l), f"{name}X")
    ax.text(*(center + Uy * l), f"{name}Y")
    ax.text(*(center + Uz * l), f"{name}Z")


def locate_bot(img: Image) -> list[np.array]:

    """
    Locates robot purple disk and two yellow bars.
    Returns points in the image plane.

    [disk, bar1, bar2]
    """

    img = np.array(img)

    # threshold color in HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask_y = cv.inRange(hsv, (20, 40, 120), (40, 255, 255))
    mask_m = cv.inRange(hsv, (140, 40, 30), (160, 255, 255))
    mask = mask_m | mask_y

    # simplify mask
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((7, 7)))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, np.ones((11, 11)))

    labels = skimage.measure.label(closed)
    regions = skimage.measure.regionprops(labels)

    # filter out 3 largest contours based on area (ideally disk and bars)
    areas = np.array([r.area for r in regions])
    idx = areas.argsort()[::-1]
    coords = [regions[i].coords[:, ::-1] for i in idx[:3]]

    return coords


def locate_rest(img: Image) -> list[np.array]:

    """
    Locates red, green and blue cubes and goals from the image.
    Returns points from the image plane.

    [[red_cube, red_goal], [green_cube, green_goal], [blue_cube, blue_goal]]
    """

    img = np.array(img)

    # threshold color in HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    red = cv.inRange(hsv, (0, 90, 50), (10, 255, 255))
    red += cv.inRange(hsv, (170, 90, 50), (179, 255, 255))
    green = cv.inRange(hsv, (40, 30, 30), (70, 255, 255))
    blue = cv.inRange(hsv, (110, 30, 30), (130, 255, 255))

    # simplify masks
    red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((7, 7)))
    red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((11, 11)))
    green = cv.morphologyEx(green, cv.MORPH_OPEN, np.ones((7, 7)))
    green = cv.morphologyEx(green, cv.MORPH_CLOSE, np.ones((11, 11)))
    blue = cv.morphologyEx(blue, cv.MORPH_OPEN, np.ones((7, 7)))
    blue = cv.morphologyEx(blue, cv.MORPH_CLOSE, np.ones((11, 11)))

    coords = []
    for color in (red, green, blue):
        labels = skimage.measure.label(color)
        regions = skimage.measure.regionprops(labels)
        # filter out 2 largest contours based on area (ideally cube and goal)
        areas = np.array([r.area for r in regions])
        idx = areas.argsort()[::-1]
        regions = [regions[i] for i in idx[:2]]
        y, x = map(int, regions[0].centroid)
        if color[y, x] != 0: # cube
            coord = [r.coords[:, ::-1] for r in regions]
        else: # goal
            coord = [r.coords[:, ::-1] for r in regions[::-1]]
        coords.append(coord)

    return coords
