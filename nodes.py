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
import server

#OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-"
from openai import OpenAI

# Memeplex UID/APIKey (Optional)
uid=""
apikey=""

def gpt(utterance,model="gpt-3.5-turbo"):
    messages=[
        {"role": "system", "content": "You are specialist of LLM and Stable Diffusion,and artist."},
    ]
    messages.append({"role": "user", "content": utterance})
    #response = openai.chat(
    print("call gpt")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        #model="gpt-3.5-turbo-1106",
        #model="gpt-3.5-turbo",
        response_format={"type":"json_object"},
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

import requests
 
class GPT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING",{"forceInput": True}),
                 "prompt": ("STRING", {"multiline": True}),
                 "model": (["gpt-3.5-turbo","gpt-4-1106-preview"],)},
                 "optional": {"result": ("STRING", {"multiline": True})}
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Memeplex"

    def run(self, text, model,prompt,result=None):
        for i in range(5):
            prompt=("以下の条件を満たすStableDiffusion用のプロンプトをできるだけ詳細にかつ、masterpieceなどのいい感じのワードを加え、高画質にできるだけ近づけるようにして最低30語は使え。全て英語で考えろ。JSON形式で、回答はpromptというプロパティに格納しろ。日本語は一切使わず、プロンプト以外の余計なことも言うな\n"+
                prompt+"\n"+text)
            response=gpt(prompt,model=model)
            response=json.loads(response)
            if "prompt" in response:
                response=response["prompt"]
                break
            time.sleep(5)
        if i >= 4:
            print("gpt response error")
            throw("gpt response error")
        url = "http://localhost:8188/memeplex/update_text"
        headers = {'Content-Type': 'application/json'}
        data = {
            "text": response,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("データの送信に成功しました。")
        else:
            print(f"データの送信に失敗しました。ステータスコード: {response.status_code}")
        return (text,)

class TextSend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"forceInput": True})}}
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "MyCustomClient"

    def run(self, text):
        # テキストをクライアントに送信する（コンソールアプリの方で受け取る）
        text = text + " (from TextSend node)" 
        server.PromptServer.instance.send_sync("send_text", {"text": text})
        return ()


NODE_CLASS_MAPPINGS = {
    "TextSend": TextSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextSend": "TextSend",
}


def generate_image(prompt,negative,model="custom_sdxl_anime2",qty=1,width=1024,height=1024):
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
                             "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                             "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                             "qty": ("INT", {"default": 9, "min": 1, "max": 9, "step": 1}),
                             "model": (["SDXL1.0","trinart","StableDiffusion-v1-5","StableDiffusion-v2-0"],)
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
    "DallERender":DallERender,
    "GPT":GPT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextInput": "TextInput",
    "MemeplexCustomSDXLRender": "MemeplexCustomSDXLRender",
    "MemeplexRender":"MemeplexRender",
    "DallERender":"DallERender",
    "GPT":"GPT"
}
