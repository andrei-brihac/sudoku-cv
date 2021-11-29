import cv2 as cv
import numpy as np


def show_image(img : np.ndarray, name : str = ''):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocess_img(img : np.ndarray):
    """
        returns an image more suitable for edge extraction
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_median = cv.medianBlur(img_gray, 3)
    img_gaussian = cv.GaussianBlur(img_gray, (0, 0), 3)
    img_sharpened = cv.addWeighted(img_median, 1, img_gaussian, -0.6, 0)
    img_thresh = cv.threshold(img_sharpened, 30, 255, cv.THRESH_BINARY)[1]
    img_erode = cv.erode(img_thresh, np.ones((5, 5), dtype=np.uint8))
    return img_erode

def manhattan_distance(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def extract_sudoku_grid(img : np.ndarray):
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

    contours = cv.findContours(cv.Canny(preprocess_img(img), 85, 255), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    maxarea_contour = max(contours, key = lambda contour : cv.contourArea(contour))
    # using is_left for rows and is_top for columns we get the matrix:
    # [bottom_right, top_right] 
    # [bottom_left, top_left]
    corners = [[None, None], [None, None]]
    img_h, img_w, _ = img.shape
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
    return cv.warpPerspective(img, perspective_transform_matrix, (new_img_h, new_img_w))