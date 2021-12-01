import cv2 as cv
import numpy as np
from typing import Tuple, List

from numpy.lib.arraypad import pad
import utils


def manhattan_distance(p1 : Tuple[int, int], p2 : Tuple[int, int]) -> int:
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def euclidean_distance(p1 : Tuple[int, int], p2 : Tuple[int, int]) -> int:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_grid(img : np.ndarray) -> np.ndarray:
    """
        returns a cropped sudoku image that only contains the sudoku grid\n
        finds the contours with findContours and crops the contour with the maximum area - the sudoku grid - from the main image\n
        the cropping operation is done using warpPerspective:\n 
            - 4 bounding points for the new image are found by getting the four corners of the maxarea contour\n
            - 4 bounding points for the old image are the corners of the main image\n
        the corners of the grid are found by comparing each point in the maxarea contour with the nearest main image corner\n
        for each corner of the main image, the nearest maxarea contour point is taken
    """
    def enhance_grid(img : np.ndarray) -> np.ndarray:
        """
            returns an image more suited for extracting the grid with findContours
        """
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # get grayscale image
        img = cv.GaussianBlur(img, (9, 9), 0)  # blur get rid of noise
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)  # thresh to get rid of bright/dark spots
        img = cv.bitwise_not(img)  # make the foreground white
        img = cv.dilate(img, np.ones((3, 3), dtype=np.uint8))  # dilate to make edges more visible
        return img

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

    contours = cv.findContours(enhance_grid(img.copy()), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    maxarea_contour = max(contours, key = lambda contour : cv.contourArea(contour))
    # using is_left for rows and is_top for columns we get the matrix:
    # [bottom_right, top_right] 
    # [bottom_left, top_left]
    corners = [[None, None], [None, None]]
    img_h, img_w = img.shape[0], img.shape[1]
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

def get_grid_zones(grid : np.ndarray) -> np.ndarray:
    """
        returns a 2D array where arr[i, j] is the jigsaw zone number of the square at row=i, col=j\n
        it does so in multiple steps:\n
            - applies some processing to the grid image to get rid of thin lines and retain thick lines\n
            - each cell in the grid is labeled depending on the zone of its neighbours\n
                - a neighbour is a cell with no border between it and the home cell
                - during the labeling process, equivalent zone indexes are identified and replaced\n
                - using a fill algorithm, cells are relabeled to a number between 1 and 9\n
        see the function definition for more details
    """
    def enhance_thick_lines(grid : np.ndarray) -> np.ndarray:
        """
            returns an image of the grid which retains its thicker lines
        """
        grid = cv.cvtColor(grid, cv.COLOR_BGR2GRAY)  # get grayscale image
        grid = cv.bilateralFilter(grid, 9, 50, 50)  # blur reduce noise while keeping edges
        grid = cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)  # thresh to get rid of bright/dark spots
        grid = cv.bitwise_not(grid)  # make the foreground white
        grid = cv.erode(grid, np.ones((2, 2), dtype=np.uint8), iterations=3)  # erode to get rid of thinner lines
        grid = cv.dilate(grid, np.ones((2, 2), dtype=np.uint8), iterations=1) # dilate to make the remaining lines thicker
        return grid
    
    def check_square_borders(sqr : np.ndarray) -> List[bool]:
        """
            returns a list of booleans for the existence of square borders in order [left, top, right, bottom]
        """
        def get_border_imgs(sqr : np.ndarray) -> List[np.ndarray]:
            """
                returns a list containing the images of the borders in the order [left, top, right, bottom]
            """
            sqr_h, sqr_w = sqr.shape[0], sqr.shape[1]
            left = sqr[sqr_h//4:int(sqr_h*3/4), :sqr_w//4]  # 1/4h:3/4h , 0:1/4w
            top = sqr[:sqr_h//4, sqr_w//4:int(sqr_w*3/4)]  # 0:1/4h , 1/4w:3/4w
            right = sqr[sqr_h//4:int(sqr_h*3/4), int(sqr_w*3/4):]  # 1/4h:3/4h , 3/4w:w 
            bottom = sqr[int(sqr_h*3/4):, sqr_w//4:int(sqr_w*3/4)]  # 3/4h:h, 1/4w:3/4w
            return [left, top, right, bottom]
        
        is_border = []
        for border in get_border_imgs(sqr):
            is_border.append(np.mean(border) > 30)  # check for the existence of a line by using the mean of the binary image
        return is_border
    
    def get_neighbour_positions(sqr : np.ndarray, row : int, col : int) -> List[Tuple[int, int]]:
        """
            returns a list of (x, y) coordinates for neighbours of the same zone of given square at position (row, col)
        """
        left_border, top_border, right_border, bottom_border = check_square_borders(sqr)
        positions = []
        max_row, max_col = 9, 9
        if not left_border and col - 1 >= 0:
            positions.append((row, col - 1))
        if not top_border and row - 1 >= 0:
            positions.append((row - 1, col))
        if not right_border and col + 1 < max_col:
            positions.append((row, col + 1))
        if not bottom_border and row + 1 < max_row:
            positions.append((row + 1, col))
        return positions
    
    grid = enhance_thick_lines(grid)
    grid_zones = np.zeros((9, 9), dtype=np.uint8)
    zone_number = 0
    equivalent_zones = dict()
    for sqr, i, j in get_squares(grid):
        neighbour_zones = set()
        for neighbour_row, neighbour_col in get_neighbour_positions(sqr, i, j):
            neighbour_zones.add(grid_zones[neighbour_row][neighbour_col])
        if len(neighbour_zones) == 1 and 0 in neighbour_zones:  # if it has found a new zone
            zone_number += 1
            grid_zones[i][j] = zone_number
        if len(neighbour_zones) >= 1 and 0 in neighbour_zones:  # remove unexplored neighbours
            neighbour_zones.remove(0)
        if len(neighbour_zones) >= 1:  # found zone from neighbour
            neighbour_zones = list(neighbour_zones)
            grid_zones[i][j] = neighbour_zones[0]
            for k in range(1, len(neighbour_zones)):  # add to zone equivalence table
                equivalent_zones[neighbour_zones[k]] = neighbour_zones[0]
    
     # replace equivalent indexes
    for i in range(9):
        for j in range(9):
            if grid_zones[i][j] in equivalent_zones:
                grid_zones[i][j] = equivalent_zones[grid_zones[i][j]]

    # re-index with fill algorithm
    zone_number = 0
    indexed = np.zeros((9, 9), dtype=np.uint8)
    final_grid_zones = np.zeros((9, 9), dtype=np.uint8)
    d = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(9):
        for j in range(9):
            if indexed[i][j] == 0:
                zone_number += 1
                q = [(i, j)]
                indexed[i][j] = 1
                while len(q) != 0:
                    curr = q[0]
                    final_grid_zones[curr[0]][curr[1]] = zone_number
                    for p in d:
                        dx, dy = curr[0] + p[0], curr[1] + p[1]
                        if dx >= 0 and dx < 9 and dy >= 0 and dy < 9 and indexed[dx][dy] == 0 and grid_zones[dx][dy] == grid_zones[i][j] :
                            q.append((dx, dy))
                            indexed[dx][dy] = 1
                    q.pop(0)
    return final_grid_zones

def get_squares(grid : np.ndarray, pad_h : int = 0, pad_w : int = 0) -> np.ndarray:
    """
        returns a generator for each grid cell image in the sudoku grid
    """
    grid_h, grid_w = grid.shape[0], grid.shape[1]
    sqr_h, sqr_w = grid_h // 9, grid_w // 9
    for i in range(9):
        for j in range(9):
            h_cut = (i*sqr_h, i*sqr_h + sqr_h + pad_h)
            w_cut = (j*sqr_w, j*sqr_w + sqr_w + pad_w)
            yield grid[h_cut[0]:h_cut[1], w_cut[0]:w_cut[1]], i, j

def get_square_cut(sqr : np.ndarray) -> str:
    """
        returns a square cut by 1/4 on all sides
    """
    sqr_h, sqr_w = sqr.shape[0], sqr.shape[1]
    sqr = sqr[sqr_h//4:, 0:sqr_w - sqr_w//4, :]
    sqr_h, sqr_w = sqr.shape[0], sqr.shape[1]
    sqr = sqr[0:sqr_h - sqr_h//4, sqr_w//4:, :]
    return sqr

def process_square(sqr : np.ndarray) -> np.ndarray:
    """
        returns a de-noised and thresholded image of the square
    """
    sqr = cv.cvtColor(sqr, cv.COLOR_BGR2GRAY)
    sqr = cv.GaussianBlur(sqr, (9, 9), 0)
    sqr = cv.bitwise_not(cv.adaptiveThreshold(sqr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2))
    return sqr

def get_square_state(sqr : np.ndarray) -> str:
    """
        returns 'x' if square contains a digit, otherwise returns 'o'\n
        it does so by doing:\n
            - cuts the square 1/4 on each side to make sure there's no grid lines left in the photo\n
            - applies a blur and threshold to get rid of noise and bright/dark spots\n
                - after the processing, the pixels of the digit(if any) have values 255\n
            - if the mean of the image is greater than a fixed value obtained by experimentation, then the square is filled
    """
    sqr = get_square_cut(sqr)    
    sqr = process_square(sqr)
    mean = np.mean(sqr)
    return 'x' if mean > 20 else 'o'

def get_square_digit(sqr : np.ndarray) -> str:
    if get_square_state(sqr.copy()) == 'o':
        return 'o'
    return '-1'
