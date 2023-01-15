# RealFaces_w_StableDiffusion
Git Repo with code and Dataset of the paper "On the limitations of Stable Diffusion: from generation to detection of realistic images"

## Work in progress...

## Dataset
%insert images
The real dataset used in this project is the FFHQ dataset, available on kaggle at https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq.
The fake generated dataset is available at the drive folder: https://drive.google.com/drive/folders/10-n9jY3USb5O_2bh4yUpo1IRPWxe1RIA; however new and different images can be generated using the code "main.py"..
The stylegan dataset is available at:
The stylegan2 dataset is available on Kaggle at: https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist
The stylegan3 dataset is available at: https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/

## Data preprocessing
In order to balance the real and fake datasets, we tried to remove all the children in the real dataset (because there are no children in the fake one) using the code available at https://www.thepythoncode.com/article/predict-age-using-opencv/.
The "weights" folder is empty. It have to be filled by the model downloadable from the sites cited in the file "remove_children.py".

## Fake Images Generation
To generate the images we used Stable Diffusion v1.5 available at HuggingFace. In order to use it, you have to send a request and accept the terms of service. Done this you will receive a token that, in the code "main.py", must be inserted in the variable "YOUR_TOKEN", row 22, in the place of 'your_Stable_Diffusion_token'.
