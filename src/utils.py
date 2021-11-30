import cv2 as cv
import numpy as np
import os
import sudoku
from typing import List

cwd = os.path.abspath(os.getcwd())  # path to root folder
input_path = f'{cwd}/train'
output_path = f'{cwd}/Brihac_Andrei_333'
img_extension = '.jpg'

def show_image(img : np.ndarray, name : str = '') -> None:
    """
        displays image and waits for key input to continue
    """
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_images(type : str) -> List[np.ndarray]:
    """
        returns a generator for each image read from input_path of given type\n
        type - the name of the input folder
    """
    global input_path
    img_names = filter(lambda name : img_extension in name, os.listdir(f'{input_path}/{type}'))
    for img_name in sorted(img_names):
        img = cv.imread(f'{input_path}/{type}/{img_name}')  # complete path to image
        img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
        yield img, img_name

def write_solution(img : str, type : str, name : str) -> None:
    """
        writes to output_path the data extracted from sudoku image of given type
    """
    global output_path
    if type not in os.listdir(output_path):
        os.mkdir(f'{output_path}/{type}')
    name = name.strip(img_extension)
    with open(f'{output_path}/{type}/{name}_predicted.txt', 'w') as f1, open(f'{output_path}/{type}/{name}_bonus_predicted.txt', 'w') as f2:
        grid = sudoku.get_grid(img.copy())
        if type == 'jigsaw':
            grid_zones = sudoku.get_grid_zones(grid.copy())
        i, j = 0, 0
        for square, row, col in sudoku.get_squares(grid):
            if type == 'jigsaw':
                f1.write(str(grid_zones[row][col]))
                f2.write(str(grid_zones[row][col]))
            f1.write(sudoku.get_square_state(square))
            f2.write(sudoku.get_square_digit(square))
            i += 1
            if i >= 9:
                i = 0
                j += 1
                if j < 9:
                    f1.write('\n')
                    f2.write('\n')
