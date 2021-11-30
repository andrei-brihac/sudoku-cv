import cv2 as cv
import numpy as np
from typing import Tuple
import utils


def manhattan_distance(p1 : Tuple[int, int], p2 : Tuple[int, int]) -> int:
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def euclidean_distance(p1 : Tuple[int, int], p2 : Tuple[int, int]) -> int:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def preprocess_img(img : np.ndarray) -> np.ndarray:
    """
        returns a binary adaptive thresholded image
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (9, 9), 0)
    img_thresh = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    img_inverted = cv.bitwise_not(img_thresh)
    img_dilated = cv.dilate(img_inverted, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    return img_dilated

def get_grid(img : np.ndarray) -> np.ndarray:
    def compare_corners(old_corner, new_corner, is_left, is_top, img_h, img_w):
        """
            returns the point with the smaller manhattan distance to the chosen corner\n
            the chosen corner is based on the is_left and is_top parameters\n
        """
        if not old_corner or not new_corner:
            return new_corner
        img_corner = ((not is_left) * img_w, (not is_top) * img_h)
        old_corner_distance = manhattan_distance(old_corner, img_corner)
        new_corner_distance = manhattan_distance(new_corner, img_corner)
        return new_corner if new_corner_distance < old_corner_distance else old_corner

    img = preprocess_img(img)
    contours = cv.findContours(cv.Canny(img, 85, 255), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    maxarea_contour = max(contours, key = lambda contour : cv.contourArea(contour))
    # using is_left for rows and is_top for columns we get the matrix:
    # [bottom_right, top_right] 
    # [bottom_left, top_left]
    corners = [[None, None], [None, None]]
    img_h, img_w = img.shape
    for point in maxarea_contour:
        x, y = point[0][0], point[0][1]
        is_left = x < img_w // 2
        is_top = y < img_h // 2
        corners[is_left][is_top] = compare_corners(corners[is_left][is_top], (x, y), is_left, is_top, img_h, img_w)
    # flatten the matrix
    bottom_right, top_right, bottom_left, top_left = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    # get shape and corner coordinates of the new cropped image
    new_img_h = max(manhattan_distance(bottom_left, top_left), manhattan_distance(bottom_right, top_right))
    new_img_w = max(manhattan_distance(bottom_left, bottom_right), manhattan_distance(top_left, top_right))
    new_img_corners = np.array([[0, 0], [new_img_w, 0], [new_img_w, new_img_h], [0, new_img_h]], dtype=np.float32)
    # get transform matrix and return the extracted grid
    perspective_transform_matrix = cv.getPerspectiveTransform(corners, new_img_corners)
    return cv.warpPerspective(img, perspective_transform_matrix, (new_img_w, new_img_h))

def get_squares(grid : np.ndarray) -> np.ndarray:
    grid_h, grid_w = grid.shape
    sqr_h, sqr_w = grid_h // 9, grid_w // 9
    for i in range(9):
        for j in range(9):
            yield grid[i*sqr_h:i*sqr_h + sqr_h, j*sqr_w:j*sqr_w + sqr_w]

def square_is_filled(sqr : np.ndarray) -> bool:
    # cut 1/4 around each border to make sure there's no extra lines left
    sqr_h, sqr_w = sqr.shape
    sqr = sqr[sqr_h//4:, 0:sqr_w - sqr_w//4]
    sqr_h, sqr_w = sqr.shape
    sqr = sqr[0:sqr_h - sqr_h//4, sqr_w//4:]
    sqr = cv.erode(sqr, np.ones((3, 3), dtype=np.uint8))
    mean = np.mean(sqr)
    return mean > 20
