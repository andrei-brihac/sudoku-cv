# sudoku-cv

Using computer vision algorithms to extract data about sudoku images.

## Requirements

numpy==1.20.2

opencv_python==4.5.3.56

## How to run

Run **main.py** from the root folder using:

`python3 src/main.py`

### Changing tasks and I/O paths.

1. Changing which task to run can be done in **main.py** using the **types** variable. These are also the folder names for the sets of images in the input folder.

2. Changing the paths to input and output folders can be done in **utils.py** using the **cwd**, **input_path**, **output_path** variables.

-   **Note**: you only have to create the output folder, the task folders are created by the script.

3. For grading results, inside the **grade_solutions.py** file change the **cwd**, **predictions_path_root**, **ground_truth_path_root** variables.

## Solution description

The methods used for extracting the data are found in the **src/sudoku.py** folder. For more detailed information about the methods used read the docstrings.

A common method for both tasks is the cropping and straightening of the sudoku grid from the main image. It's done in multiple steps:

1. Preprocessing the main image to bring out the grid edge. (see **enhance_grid**)
2. Using **findContours** to get the contour with the maximum area, which is the grid.
3. Getting the corners of the grid from this maxarea contour. (see **get_grid**)
4. Using **warpPerspective** with the newfound corners to straighten and crop the grid.

### Task 1 - classic sudoku

After getting the image of the sudoku grid, it separates each cell and checks whether it contains a digit by converting the cell to a de-noised binary image and computing its mean. If the value is greater than a threshold obtained by observing the mean value of images with digits (in this case 20), it returns that the cell is filled. Determining the digit is not yet implemented, but you could use template matching or any machine learning model to do it.

### Task 2 - jigsaw sudoku

The added difficulty is finding out which zone each cell belongs to. The jigsaw sudoku images have thicker lines separating the zones so we can do the following:

1. Modify the grid image so that it retains the thicker lines and gets rid of the thinner lines. (see **enhance_thick_lines**).
2. Use an algorithm to label the cells according to their neighbours, where the neighbours are cells not separated by a border from the original cell. (see **get_grid_zones**)
3. Neighbours of the same cell with different zones denote that two previously unconnected zones have been connected.
4. Labels are changed according to these connections and then re-indexed to numbers from 1 to 9 - the total number of zones.

My solution works 100% of the time for classic sudoku and about 50% of the time for jigsaw sudoku. You can use the functions in **src/utils.py** to display images and modify behaviour.

The jigsaw sudoku solution could be improved to give better results by modifying the way in which neighbour cells are identified (current method is checking for borders of a cell with **numpy.mean**, similar to filled cell detection in task 1) and better image processing for removing thin lines and keeping thick lines. Check the methods in **get_grid_zones()** in **src/sudoku.py**.
