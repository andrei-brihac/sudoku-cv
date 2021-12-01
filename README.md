# sudoku-cv

Using computer vision algorithms to extract data about sudoku images.

## Requirements

numpy==1.20.2

opencv_python==4.5.3.56

## How to run

Run **main.py** from the root folder using:

`python3 src/main.py`

### Changing tasks and I/O paths.

1. Changing which task to run can be done in **main.py** using the **types** variable.

2. Changing the paths to input and output folders can be done in **utils.py** using the **cwd**, **input_path**, **output_path** variables.

-   **Note**: you only have to create the output folder, the task folders are created by the script.

3. For grading results, inside the **grade_solutions.py** file change the **cwd**, **predictions_path_root**, **ground_truth_path_root** variables.

## Solution description

### Task 1 - classic sudoku

### Task 2 - jigsaw sudoku
