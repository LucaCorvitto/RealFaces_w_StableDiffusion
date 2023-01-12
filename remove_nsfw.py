import os
from PIL import Image
import shutil

def main():
    for path in ['datasets/eval/fake','datasets/test/fake','datasets/train/fake']:
        images = os.listdir(path)
        del_imgs = 0
        for image_path in images:
            #print(image_path)
            img = Image.open(os.path.join(path, image_path)).convert('L')
            clrs = img.getcolors()
            if len(clrs) == 1:
                del_imgs+=1
                #print(image_path)
                # delete the image from the folder
                os.remove(os.path.join(path, image_path))
                #shutil.move(os.path.join(path, image_path), os.path.join('datasets/nsfw', image_path))
        print(f'deleted images from {path}: {del_imgs}')

if __name__ == '__main__':
    main()
