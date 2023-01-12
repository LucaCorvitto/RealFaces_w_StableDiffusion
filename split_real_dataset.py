import os
import random
from itertools import accumulate
import shutil

def main():

    path = 'datasets/archive' # name of the folder in which is saved the real dataset

    real_images = os.listdir(path)
    random.shuffle(real_images)

    num_images = len(real_images)
    test_eval_size = round(0.1*num_images)

    length_to_split = [test_eval_size, test_eval_size, num_images-2*test_eval_size]
    #print(length_to_split)
    
    # Using zip
    split = [real_images[x - y: x] for x, y in zip(
            accumulate(length_to_split), length_to_split)]

    # Printing Output
    #print("Initial list is:", real_images)
    print("Split length list: ", length_to_split)
    #print("List after splitting", split)

    source_folder = "datasets/archive"
    destination_folders = ["datasets/eval/real/","datasets/test/real/","datasets/train/real/"]

    # iterate files
    for i in range(3):
        for file in split[i]:
            # construct full file path
            source = os.path.join(source_folder, file)
            destination = os.path.join(destination_folders[i], file)
            # move file
            shutil.move(source, destination)

if __name__ == '__main__':
    main()
