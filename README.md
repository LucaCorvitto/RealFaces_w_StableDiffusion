# RealFaces_w_StableDiffusion
Git Repo with code and Dataset of the paper **"On the use of Stable Diffusion for creating realistic faces: from generation to detection"**.

This Repo contains the code used to generate the fake Dataset proposed in the paper and the code used for the overall analysis. The other datasets cited and linked are **NOT** proposed by us and the credits go to the original creators.

## Dataset
| Real (FFHQ)   | Stable Diffusion (ours) | GAN    | GAN2   | GAN3  |
| ------------- |:-----------------------:|:------:|:------:|:-----:|
|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/real.png)|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/fake.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan2.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan3.png)|

The fake generated dataset that we propose is available at the drive folder: [Stable Diffusion fakes](https://drive.google.com/drive/folders/10-n9jY3USb5O_2bh4yUpo1IRPWxe1RIA); however new and different images can be generated using the code [main.py](main.py).

The other datasets used in this project for detection and classification purpose were taken from external resources. They are:
* [FFHQ dataset](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) composed by real faces images;
* [StyleGAN dataset](https://iplab.dmi.unict.it/deepfakechallenge/training/1-STYLEGAN.zip) made available for the [Deepfake challenge](https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]);
* [StyleGAN2 dataset](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist) composed by the images generated from the famous website [This Person Does Not Exist](https://thispersondoesnotexist.com/);
* [StyleGAN3 dataset](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/) made available directly from NVIDIA.

## How to run the code
### Fake Images Generation
To generate the images we used [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) from [HuggingFace](https://huggingface.co/). The code is ready to run, since the license of the model does not need to be explicitly accepted through the UI anymore.

### Binary Classifier
#### Fake Dataset
The fake dataset must be downloaded from the link stated [above](#dataset). It is split yet, but each zip must be extracted and put in the folders:
* [train](./datasets/png_images/train)
* [eval](./datasets/png_images/eval)
* [test](./datasets/png_images/test)

each one inside a subfolder named `fake`.

#### Real Dataset
The real dataset must be split, running the [split_real_dataset.py](split_real_dataset.py) file, in order to run the classifiers codes (The FFHQ dataset must be downloaded first from the link stated [above](#dataset), and then moved in the [datasets](datasets) folder, naming the subfolder containing all the images `archive`).

#### Data preprocessing
In order to balance the two datasets, we removed the children in the real one (because we avoided to generate children) using the code available [here](https://www.thepythoncode.com/article/predict-age-using-opencv/).
The [weights](weights) folder is empty. It have to be filled by the model downloadable from the websites cited in the file [remove_children.py](remove_children.py). The algorithm used it is not ours, so the credits go to the original authors.

### Multi-class classifier
To run the 5_classes_classifier one has to download the StyleGAN datasets from the link stated [above](#dataset), then move them in the different folders:
* [gan](datasets/gan)
* [gan2](datasets/gan2)
* [gan3](datasets/gan3)
 
and run the [split_5classes_dataset.py](split_5classes_dataset.py) file.
