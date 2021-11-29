import cv2 as cv
import numpy as np
import utils

for i in range(1, 20):
    filenumber = '0' + str(i) if i < 10 else str(i)
    test_image = cv.imread(f'/home/charmichles/Desktop/Coding/Git Repos/sudoku-cv/train/classic/{filenumber}.jpg')
    test_image = cv.resize(test_image, (0, 0), fx=0.2, fy=0.2)
    test_image = utils.extract_sudoku_grid(test_image)
    utils.show_image(test_image)
