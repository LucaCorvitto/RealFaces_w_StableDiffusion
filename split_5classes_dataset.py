import os
import random
from itertools import accumulate
import shutil

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'created folder: {path}')
    else:
        print(f'directory {path} already exists')

def split_and_reduce(path):
    for classes, name in [('real/','real'),('fake/','stable')]:
        for folder, num in [('train/', 1000), ('test/', 100), ('eval/', 100)]:
            new_path = path + folder + classes
            lista_cartelle = os.listdir(new_path)
            lista_random = random.sample(lista_cartelle, num)
            dest = 'datasets/5_classes/' + folder + name 
            for photo in lista_random:
                source = new_path + photo
                shutil.copy(source,dest)

def split(name):

    source_folder = 'datasets/' + name

    all_images = os.listdir(source_folder)
    random.shuffle(all_images)

    num_images = len(all_images)
    test_eval_size = round(0.1*num_images) #10%

    length_to_split = [test_eval_size, test_eval_size, num_images-2*test_eval_size]

    splitted = [all_images[x - y: x] for x, y in zip(
            accumulate(length_to_split), length_to_split)]

    destination_folders = [f'datasets/5_classes/eval/{name}', f'datasets/5_classes/test/{name}', f'datasets/5_classes/train/{name}']

    # iterate files
    for i in range(3):
        for file in splitted[i]:
            source = os.path.join(source_folder, file)
            destination = os.path.join(destination_folders[i], file)
            shutil.copy(source, destination)

def main():

    # create directories
    folder = 'datasets/5_classes'
    create_dir(folder)
    for subfolder in ['train', 'eval', 'test']:
        create_dir(f'{folder}/{subfolder}')
        for sub2folder in ['real', 'stable', 'gan', 'gan2', 'gan3']:
            create_dir(f'{folder}/{subfolder}/{sub2folder}')

    path = 'datasets/png_images/'
    split_and_reduce(path)

    for name in ['gan','gan2','gan3']:
        split(name)


if __name__ == '__main__':
    main()
