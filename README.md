# On the use of Stable Diffusion for creating realistic faces: from generation to detection

This Repo contains the code used to generate the fake Dataset proposed in the paper and the code used for the overall analysis. The other datasets cited and linked are **NOT** proposed by us and the credits go to the original creators.

## Abstract
The mass adoption of diffusion models has shown that artificial intelligence (AI) systems can be used to easily
generate realistic images. The spread of these technologies paves
the way to previously unimaginable creative uses while also
raising the possibility of malicious applications. In this work,
we propose a critical analysis of the overall pipeline, i.e., from
creating realistic human faces with Stable Diffusion v1.5
to recognizing fake ones. We first propose an analysis of the
prompts that allow the generation of extremely realistic faces
with a human-in-the-loop approach. Our objective is to identify
the text prompts that drive the image generation process to
obtain realistic photos that resemble everyday portraits captured
with any camera. Next, we study how complex it is to recognize
these fake contents for both AI-based models and non-expert
humans. We conclude that similar to other deepfake creation
techniques, despite some limitations in generalization across
different datasets, it is possible to use AI to recognize these
contents more accurately than non-expert humans would.

[[Paper]](https://ieeexplore.ieee.org/abstract/document/10156981)

## Dataset
| Real (FFHQ)   | Stable Diffusion (ours) | GAN    | GAN2   | GAN3  |
| ------------- |:-----------------------:|:------:|:------:|:-----:|
|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/real.png)|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/fake.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan2.png)| ![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/gan3.png)|

The fake generated dataset that we propose is available at the drive folder: [Stable Diffusion fakes](https://drive.google.com/drive/folders/10-n9jY3USb5O_2bh4yUpo1IRPWxe1RIA). The dataset was created using the prompts inside the [`prompts.txt`](prompts.txt) file. Each image's name is structured in this way:
```
<#prompt>-<#process>
```
where `<#prompt>` is the associated number of the prompt in the [`prompts.txt`](prompts.txt) file, numbered from 0, and `<#process>` is the number representing the order in which the image was generated starting from 0 (each of these images is generated using a different seed). For example, the file named `0-0` is the first generated image from the first prompt in the prompts file, while the one named `248-70` is the 71-st generated image from the 249-th prompt in the file:
| Generated Image | Prompt |
| --------------- |:-------:|
| ...| ... |
|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/4-54.png)| headshot portrait of a nigerian man, real life, realistic background, 50mm, Facebook, Instagram, shot on iPhone, HD, HDR color, 4k, natural lighting, photography |
| ...| ... |
|![alt text](https://github.com/LucaCorvitto/RealFaces_w_StableDiffusion/blob/main/readme_images/248-70.png)| headshot portrait of an old woman with braids blonde hair, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook, Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting|
| ...| ... |

However, new images can be generated using the code [`main.py`](main.py). In order to generate images from different prompts then the [`prompts.txt`](prompts.txt) file must be updated.

The other datasets used in this project for detection and classification purpose were taken from external resources. They are:
* [FFHQ dataset](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) composed by real faces images;
* [StyleGAN dataset](https://iplab.dmi.unict.it/deepfakechallenge/training/1-STYLEGAN.zip) made available for the [Deepfake challenge](https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]);
* [StyleGAN2 dataset](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist) composed by the images generated from the famous website [This Person Does Not Exist](https://thispersondoesnotexist.com/);
* [StyleGAN3 dataset](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/) made available directly from NVIDIA.


## How to run the code
### Fake Images Generation
To generate the images we used [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) from [HuggingFace](https://huggingface.co/). The code ([`main.py`](main.py)) is ready to run, since the license of the model does not need to be explicitly accepted through the UI anymore.

### Binary Classification
#### Fake Dataset
The fake dataset must be downloaded from the link stated [above](#dataset). It is split yet, but each zip must be extracted and put in the folders:
* [`train`](./datasets/png_images/train)
* [`eval`](./datasets/png_images/eval)
* [`test`](./datasets/png_images/test)

each one inside a subfolder named `fake`.

#### Real Dataset
The real dataset must be split, running the [`split_real_dataset.py`](split_real_dataset.py) file, in order to run the classifiers codes (The FFHQ dataset must be downloaded first from the link stated [above](#dataset), and then moved in the [`datasets`](datasets) folder, naming the subfolder containing all the images `archive`).

#### Data preprocessing
In order to balance the two datasets, we removed the children in the real one (because we avoided to generate children) using the code available [here](https://www.thepythoncode.com/article/predict-age-using-opencv/).
The [`weights`](weights) folder is empty. It have to be filled by the model downloadable from the websites cited in the file [`remove_children.py`](remove_children.py). The algorithm used it is not ours, so the credits go to the original authors.

### Multi-class classification
To run the 5_classes_classifier one has to download the StyleGAN datasets from the link stated [above](#dataset), then move them in the different folders:
* [`gan`](datasets/gan)
* [`gan2`](datasets/gan2)
* [`gan3`](datasets/gan3)
 
and run the [`split_5classes_dataset.py`](split_5classes_dataset.py) file.

## Authors
[Lorenzo Papa](https://github.com/lorenzopapa5), [Lorenzo Faiella](https://github.com/FiscalTax), [Luca Corvitto](https://github.com/LucaCorvitto), [Luca Maiano](https://github.com/lucamaiano), [Irene Amerini](https://github.com/ireneamerini).

## Cite us
L. Papa, L. Faiella, L. Corvitto, L. Maiano and I. Amerini, "On the use of Stable Diffusion for creating realistic faces: from generation to detection," 2023 11th International Workshop on Biometrics and Forensics (IWBF), Barcelona, Spain, 2023, pp. 1-6, doi: 10.1109/IWBF57495.2023.10156981.
