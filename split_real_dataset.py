import os
import random
from itertools import accumulate
import shutil

def split():

    path = 'datasets/archive' # name of the folder in which is saved the real dataset, put the archive folder downloaded in the datasets one

    real_images = os.listdir(path)
    random.shuffle(real_images)

    num_images = len(real_images)
    test_eval_size = round(0.1*num_images)

    length_to_split = [test_eval_size, test_eval_size, num_images-2*test_eval_size]
    
    # Using zip
    split = [real_images[x - y: x] for x, y in zip(
            accumulate(length_to_split), length_to_split)]

    print("Split length list: ", length_to_split)

    source_folder = "datasets/archive"

    destination_folders = ["datasets/png_images/eval/real/","datasets/png_images/test/real/","datasets/png_images/train/real/"]

    # iterate files
    for i in range(3):
        for file in split[i]:
            # construct full file path
            source = os.path.join(source_folder, file)
            destination = os.path.join(destination_folders[i], file)
            # move file
            shutil.move(source, destination)


def main():

    # split real dataset in order to run the classifiers codes
    # (The FFHQ dataset must be downloaded first from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)
    split()

if __name__ == '__main__':
    main()
