# RealFaces_w_StableDiffusion
Git Repo with code and Dataset of the paper "On the limitations of Stable Diffusion: from generation to detection of realistic images"

## Work in progress...

## Dataset
The real dataset used in this project is the FFHQ dataset, available on kaggle at https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq

## Data preprocessing
In order to balance the real and fake datasets, we tried to remove all the children in the real dataset (because there are no children in the fake one) using the code available at https://www.thepythoncode.com/article/predict-age-using-opencv/.
The "weights" folder is empty. It have to be filled by the model downloadable from the sites cited in the file "remove_children".
