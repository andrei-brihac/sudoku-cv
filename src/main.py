import utils
import traceback

verbose = True
types = ['classic', 'jigsaw']  # folder names for types of sudoku images

for type in types:
    for img, img_name in utils.get_images(type):
        try:
            if verbose:
                print(f'Processing {type}/{img_name}.') 
            utils.write_solution(img, type, img_name)
        except Exception as e:
            print(f'Error while processing {type}/{img_name}!')
            traceback.print_exception(Exception, e, tb=e.__traceback__)
