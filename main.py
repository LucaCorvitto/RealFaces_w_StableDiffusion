import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" #change number of gpu

import torch
from diffusers import DiffusionPipeline
from torch import autocast


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(f"{path}")

def roundplus(x,y):
    if x%y==0:
        return x//y
    else:
        return (x//y)+1

def main():
    #YOUR_TOKEN = 'your_Stable_Diffusion_token' # insert the Stable Diffusion token, available on HuggingFace
    num_inference_steps=50
    height=512
    width=512
    guidance_scale = 7
    num_train = 80 
    num_test = 10
    num_eval = 10
    high_end_gpu = True
    batch_size = 1
    
    #print(torch.cuda.is_available())
    #print(torch.cuda.device_count())
    #print(torch.cuda.current_device())

    with open('prompts.txt') as f:
        lines = f.readlines()

    prompts = []
    for i in lines:
        prompts.append(i.strip())
    
    if high_end_gpu:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")#, use_auth_token=YOUR_TOKEN)
    else:
        pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)#, use_auth_token=YOUR_TOKEN)

    pipe = pipe.to("cuda:0")

    num_train = roundplus(num_train,batch_size)
    num_test = roundplus(num_test,batch_size)
    num_eval = roundplus(num_eval,batch_size)

    makedir('train')
    makedir('test')
    makedir('eval')

#########################################################
##############   Saving images in train   ###############
#########################################################
    inc=0
    for i, prompt in enumerate(prompts):
        prompt = [prompt] * batch_size
        inc=0
        for j in range(num_train):
            if high_end_gpu:
                images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            else:
                with autocast("cuda"):
                    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            for image in images:
                image.save("train/{}-{}.png".format(i,inc))
                inc+=1

#########################################################
##############   Saving images in test   ################
#########################################################

    for i, prompt in enumerate(prompts):
        prompt = [prompt] * batch_size
        inc=0
        for j in range(num_test):
            if high_end_gpu:
                images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            else:
                with autocast("cuda"):
                    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            for image in images:
                image.save("test/{}-{}.png".format(i,inc))
                inc+=1

#########################################################
##############   Saving images in eval   ################
#########################################################
    
    for i, prompt in enumerate(prompts):
        prompt = [prompt] * batch_size
        inc=0
        for j in range(num_eval):
            if high_end_gpu:
                images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            else:
                with autocast("cuda"):
                    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images
            for image in images:
                image.save("eval/{}-{}.png".format(i,inc))
                inc+=1
            
if __name__ == '__main__':
    main()
