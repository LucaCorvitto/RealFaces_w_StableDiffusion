# RealFaces_w_StableDiffusion
Git Repo with code and Dataset of the paper "On the use of Stable Diffusion for creating realistic faces: from generation to detection".

This Repo contains the code used to generate the fake Dataset proposed in the paper and the code used for the overall analysis. The other dataset cited and linked are **NOT** proposed by us and the credits go to the original creators.

## Work in progress...

## How to run the code

### Fake Images Generation
To generate the images we used Stable Diffusion v1.5 available at HuggingFace. In order to use it, you have to send a request and accept the terms of service. Done this you will receive a token that, in the code "main.py", must be inserted in the variable "YOUR_TOKEN", row 22, in the place of 'your_Stable_Diffusion_token'.

### Classifiers codes and Datasets
The fake dataset must be downloaded from the link stated below, extracted and put in the `datasets/png_images/train`, `datasets/png_images/eval` and `datasets/png_images/test` folder naming the subfolders `fake`.

The real dataset must be split, running the `split_real_dataset.py` file, in order to run the classifiers codes (The FFHQ dataset must be downloaded first from the link stated [below](##Dataset), and then moved in the `datasets` folder, naming the subfolder containing all the images `archive`).

To run the 5_classes_classifier one has to download the StyleGAN datasets from the link stated below, then move them in the path `datasets/gan`, `datasets/gan2` and `datasets/gan3` and run the `split_5classes_dataset.py` file.

## Dataset
%insert images

The fake generated dataset that we propose is available at the drive folder: [Stable Diffusion fakes](https://drive.google.com/drive/folders/10-n9jY3USb5O_2bh4yUpo1IRPWxe1RIA); however new and different images can be generated using the code "main.py".

The other datasets used in this project for detetction and classification purpose were taken from external resources. They are:
* [FFHQ dataset](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) composed by real faces images;
* [StyleGAN dataset](https://iplab.dmi.unict.it/deepfakechallenge/training/1-STYLEGAN.zip) made available for the [Deepfake challenge](https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]);
* [StyleGAN2 dataset](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist) composed by the images generated from the famous website [This Person Does Not Exist](https://thispersondoesnotexist.com/);
* [StyleGAN3 dataset](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/) made available directly from NVIDIA.

## Data preprocessing
In order to balance the real and fake datasets, we removed the children in the real dataset (because there are no children in the fake one) using the code available [here](https://www.thepythoncode.com/article/predict-age-using-opencv/).
The `weights` folder is empty. It have to be filled by the model downloadable from the websites cited in the file `remove_children.py`.
