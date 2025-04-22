import numpy as np
import cv2 as cv
from PIL import Image
import pandas as pd
from scipy.linalg import rq
from pathlib import Path
from matplotlib.axes import Axes
import skimage
# You can import any other files that you use.
# Please also provide pip requirements file for any additional libraries that you might need


def calibrate(imgs):
    """Calibrate the camera using images of a scene.

    Args:
        imgs (list<PIL.Image>): a list of PIL images to be used for calibration

    Returns:
        results of calibration that could be used for finding 3D positions of robot, blocks and target positions.
        They could, for example, contain camera frame, projection matrix, etc.
    """    

    points = map(select_points, imgs)
    points = np.concatenate(list(points))

    reference = pd.read_csv('./data/3d_points.csv', index_col=0)
    reference = reference.iloc[:, -3:].to_numpy()

    calib = calibrate_norm(points, reference)
    
    return calib


def move_block(blocks, img, calib):
    """Returns the commands to move the specified blocks to their target position.

    Args:
        blocks (list<str>): a list of string values that determine which blocks you should move 
                            and in which order. For example, if blocks = ["red", "green", "blue"] 
                            that means that you need to first move red block. 
                            Your function should at minimum work for the following values of blocks:
                            blocks = ["red"], blocks = ["green"], blocks = ["blue"].
        img (PIL.Image): a PIL image containing the current arrangement of robot and blocks.
        calib : calibration results from function calibrate.

    Returns:
        str: robot commands separated by ";". 
             An example output: "go(20); grab(); turn(90);  go(-10); let_go()"
    """    

    disk, bar1, bar2 = locate_bot(img)
    bar = np.concatenate((bar1, bar2))

    disk = img_to_world(disk, calib, 9).mean(0)
    bar = img_to_world(bar, calib, 8.5).mean(0)

    command = ''

    for color in blocks:

        if color == 'red':
            block, goal = locate_red(img)
        elif color == 'blue':
            block, goal = locate_blue(img)
        elif color == 'green':
            block, goal = locate_green(img)
        else:
            raise Exception(f'Block color "{color}" not valid.')

        block = img_to_world(block, calib, 4).mean(0)
        goal = img_to_world(goal, calib, 0).mean(0)

        r_disk_bar = bar - disk
        r_disk_block = block - disk
        r_block_goal = goal - block
        distance_bar = np.linalg.norm(r_disk_bar)
        distance_block = np.linalg.norm(r_disk_block)
        distance_goal = np.linalg.norm(r_block_goal)

        sign_block = -np.sign(r_disk_bar[0] * r_disk_block[1] - r_disk_bar[1] * r_disk_block[0])
        rotate_block = sign_block * np.acos((r_disk_bar @ r_disk_block) / (distance_bar * distance_block)) * 180 / np.pi

        sign_goal = -np.sign(r_disk_block[0] * r_block_goal[1] - r_disk_block[1] * r_block_goal[0])
        rotate_goal = sign_goal * np.acos((r_disk_block @ r_block_goal) / (distance_block * distance_goal)) * 180 / np.pi

        command += f'turn({rotate_block:.0f});go({distance_block:.0f});grab();turn({rotate_goal:.0f});go({distance_goal - 12:.0f});let_go();go(-20);'

        disk = goal - 32 * r_block_goal / distance_goal
        bar = goal

    return command


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
    img = np.concatenate((img, reference), 1)

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


def calibrate_default(points2d: np.array, points3d: np.array) -> np.array:

    """
    Performs direct linear transform (DLT)
    """

    n = len(points3d)
    biased = np.concatenate((points3d, np.ones((n, 1))), 1)

    top_left = np.concatenate((biased, np.zeros((n, 4))), 1)
    bot_left = np.roll(top_left, 4, axis=1)

    top_right = -biased * points2d[:, 0:1]
    bot_right = -biased * points2d[:, 1:2]

    system = np.concatenate(
        (np.concatenate((top_left, top_right), 1), 
         np.concatenate((bot_left, bot_right), 1)), 0
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

    M = calibrate_default(normalized_2d_pts, normalized_3d_pts)
    denormalized_M = np.linalg.inv(T) @ M @ U

    return denormalized_M


def extract_params(M: np.array) -> tuple[np.array]:

    """
    Reference: https://www.youtube.com/watch?v=2XM2Rb2pfyQ
    """

    K, R = rq(M[:, :3])
    t = np.linalg.inv(K) @ M[:, 3:4]

    intrinsic_params = np.pad(K, ((0, 0), (0, 1)))
    extrinsic_params = np.concatenate((R, t), 1)
    extrinsic_params = np.concatenate((extrinsic_params, [[0, 0, 0, 1]]))

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


def locate_red(img: Image) -> list[np.array]:

    """
    Locates red cube and goal from the image.
    Returns points from the image plane.

    [cube, goal]
    """

    img = np.array(img)

    # threshold color in HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    red = cv.inRange(hsv, (0, 90, 50), (10, 255, 255))
    red += cv.inRange(hsv, (170, 90, 50), (179, 255, 255))

    # simplify masks
    red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((7, 7)))
    red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((11, 11)))

    labels = skimage.measure.label(red)
    regions = skimage.measure.regionprops(labels)

    # filter out 2 largest contours based on area (ideally cube and goal)
    areas = np.array([r.area for r in regions])
    idx = areas.argsort()[::-1]
    regions = [regions[i] for i in idx[:2]]
    y, x = map(int, regions[0].centroid)
    if red[y, x] != 0: # cube
        coords = [r.coords[:, ::-1] for r in regions]
    else: # goal
        coords = [r.coords[:, ::-1] for r in regions[::-1]]

    return coords


def locate_green(img: Image) -> list[np.array]:

    """
    Locates green cube and goal from the image.
    Returns points from the image plane.

    [cube, goal]
    """

    img = np.array(img)

    # threshold color in HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    green = cv.inRange(hsv, (40, 30, 30), (70, 255, 255))

    # simplify masks
    green = cv.morphologyEx(green, cv.MORPH_OPEN, np.ones((7, 7)))
    green = cv.morphologyEx(green, cv.MORPH_CLOSE, np.ones((11, 11)))

    labels = skimage.measure.label(green)
    regions = skimage.measure.regionprops(labels)

    # filter out 2 largest contours based on area (ideally cube and goal)
    areas = np.array([r.area for r in regions])
    idx = areas.argsort()[::-1]
    regions = [regions[i] for i in idx[:2]]
    y, x = map(int, regions[0].centroid)
    if green[y, x] != 0: # cube
        coords = [r.coords[:, ::-1] for r in regions]
    else: # goal
        coords = [r.coords[:, ::-1] for r in regions[::-1]]

    return coords


def locate_blue(img: Image) -> list[np.array]:

    """
    Locates blue cube and goal from the image.
    Returns points from the image plane.

    [cube, goal]
    """

    img = np.array(img)

    # threshold color in HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    blue = cv.inRange(hsv, (110, 30, 30), (130, 255, 255))

    # simplify masks
    blue = cv.morphologyEx(blue, cv.MORPH_OPEN, np.ones((7, 7)))
    blue = cv.morphologyEx(blue, cv.MORPH_CLOSE, np.ones((11, 11)))

    labels = skimage.measure.label(blue)
    regions = skimage.measure.regionprops(labels)

    # filter out 2 largest contours based on area (ideally cube and goal)
    areas = np.array([r.area for r in regions])
    idx = areas.argsort()[::-1]
    regions = [regions[i] for i in idx[:2]]
    y, x = map(int, regions[0].centroid)
    if blue[y, x] != 0: # cube
        coords = [r.coords[:, ::-1] for r in regions]
    else: # goal
        coords = [r.coords[:, ::-1] for r in regions[::-1]]

    return coords


def img_to_world(points, calib, z):
    n = len(points)
    A = np.linalg.inv(calib[:, [0, 1, 3]])
    b = calib[:, 2:3]
    points = np.concatenate((points.T, [[1] * n]), 0)
    out = A @ (points - z * b)
    return (out[:-1] / out[-1]).T
