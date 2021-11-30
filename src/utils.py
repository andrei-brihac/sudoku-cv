import cv2 as cv
import numpy as np
import os
import sudoku
from typing import List

cwd = os.path.abspath(os.getcwd())  # path to root folder
input_path = f'{cwd}/grading/fake_test'  # path to input folder
output_path = f'{cwd}/Brihac_Andrei_333'  # path to output folder
img_extension = '.jpg'

def show_image(img : np.ndarray, name : str = '') -> None:
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_images(type : str) -> List[np.ndarray]:
    global cwd, input_path
    img_names = filter(lambda name : img_extension in name, os.listdir(f'{input_path}/{type}'))
    for img_name in sorted(img_names):
        img = cv.imread(f'{input_path}/{type}/{img_name}')  # complete path to image
        img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
        yield img, img_name

def write_solution(img : str, type : str, name : str) -> None:
    global cwd, output_path
    name = name.strip(img_extension)
    with open(f'{output_path}/{type}/{name}_predicted.txt', 'w') as f1, open(f'{output_path}/{type}/{name}_bonus_predicted.txt', 'w') as f2:
        grid = sudoku.get_grid(img)
        i = 0
        j = 0
        for square in sudoku.get_squares(grid):
            f1.write('x' if sudoku.square_is_filled(square) else 'o')
            f2.write('x' if sudoku.square_is_filled(square) else 'o')
            i += 1
            if i >= 9:
                i = 0
                j += 1
                if j < 9:
                    f1.write('\n')
                    f2.write('\n')
