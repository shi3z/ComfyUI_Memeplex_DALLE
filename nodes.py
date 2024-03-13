from PIL import Image ,ImageOps
import numpy as np
import torch

import requests
import time
from PIL import Image
import io
import numpy as np
import json
import os

#OpenAIのAPIキーを入力
os.environ["OPENAI_API_KEY"] = "sk-"
from openai import OpenAI

# MemeplexのUIDとAPIKEYを入力
uid=""
apikey=""
def generate_image(prompt,negative,model="custom_sdxl_anime1",qty=1,width=1024,height=1024):
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return(response.data[0].url)
    """
    print(prompt,negative,model)
    url=f"https://memeplex.app/api/?uid={uid}&apikey={apikey}&steps=30&width={width}&height={height}&prompt={prompt}&qty={qty}&negative={negative}&model={model}"
    res = requests.get(url)
    result=json.loads(res.text)
    print(result)
    return result["result"]


class TextInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Memeplex"

    def run(self, text, seed = None):
        return (text,)

class MemeplexCustomSDXLRender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "prompt": ("STRING", {"forceInput": True}),
                             }
                             ,
                "optional":{
                            "negative": ("STRING", {"forceInput": True}),
                            }

                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)

    FUNCTION  = "run"
    CATEGORY = "Memeplex"

    def run(self, prompt,negative):
        print(prompt)
        url=generate_image(prompt,negative)[0]
        print("request queing,wait 60sec")

        time.sleep(10)
        output_images=[]
        urls=[url]
        urls.append(url.replace("0_","1_0_"))
        urls.append(url.replace("0_","2_1_0_"))
        urls.append(url.replace("0_","3_2_1_0_"))
        for url in urls:
            print(url)
            while True:
                try:
                    i=Image.open(io.BytesIO(requests.get(url).content))
                    if i.width>0:
                        break
                    print("retry")
                except Exception as e:
                    print(e)
                    print("retry in 30sec")
                time.sleep(30)

            #i = Image.open("input/example.png")        
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            print(image.shape)
            output_images.append(image)
        output_image = torch.cat(output_images, dim=0)
        #print(output_images.shape)
        return (output_image,)

class MemeplexRender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "prompt": ("STRING", {"forceInput": True}),
                             "width": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 64}),
                             "height": ("INT", {"default": 512, "min": 512, "max": 1024, "step": 64}),
                             "qty": ("INT", {"default": 9, "min": 1, "max": 9, "step": 1}),
                             "model": (["trinart","StableDiffusion-v1-5","StableDiffusion-v2-0"],)
                             }
                             ,
                "optional":{
                            "negative": ("STRING", {"forceInput": True}),
                            }

                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)

    FUNCTION  = "run"
    CATEGORY = "Memeplex"

    def run(self, prompt,negative,width,height,qty,model):
        print(prompt)
        urls=generate_image(prompt,negative,model=model,qty=qty,width=width,height=height)
        print("request queing,wait 10sec")

        time.sleep(3)
        output_images=[]
        for url in urls:
            while True:
                try:
                    i=Image.open(io.BytesIO(requests.get(url).content))
                    if i.width>0:
                        break
                    print("retry")
                except Exception as e:
                    print(e)
                    print("retry in 30sec")
                time.sleep(30)

            #i = Image.open("input/example.png")        
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            print(image.shape)
            output_images.append(image)
        output_image = torch.cat(output_images, dim=0)
        #print(output_images.shape)
        return (output_image,)


client = OpenAI()
def generate_image_DALLE(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return(response.data[0].url)

class DallERender:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "prompt": ("STRING", {"forceInput": True}),
                             }

                }
                
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)

    FUNCTION  = "run"
    CATEGORY = "Memeplex"

    def run(self, prompt):
        print(prompt)
        urls=[generate_image_DALLE(prompt)]
        print("request queing,wait 10sec")

        time.sleep(3)
        output_images=[]
        for url in urls:
            while True:
                try:
                    i=Image.open(io.BytesIO(requests.get(url).content))
                    if i.width>0:
                        break
                    print("retry")
                except Exception as e:
                    print(e)
                    print("retry in 30sec")
                time.sleep(30)
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            print(image.shape)
            output_images.append(image)
        output_image = torch.cat(output_images, dim=0)
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "TextInput": TextInput,
    "MemeplexCustomSDXLRender": MemeplexCustomSDXLRender,
    "MemeplexRender": MemeplexRender,
    "DallERender":DallERender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextInput": "TextInput",
    "MemeplexCustomSDXLRender": "MemeplexCustomSDXLRender",
    "MemeplexRender":"MemeplexRender",
    "DallERender":"DallERender"
}
