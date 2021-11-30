import utils

verbose = True

for type in ['classic']:
    for img, img_name in utils.get_images(type):
        try:
            if verbose:
                print(f'Processing {type}/{img_name}.') 
                utils.write_solution(img, type, img_name)
        except Exception as e:
            print(f'Error while processing {type}/{img_name} : {e}!')
