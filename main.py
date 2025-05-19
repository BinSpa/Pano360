import os
import re
import json
import random
import base64
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from utils import *

# GPT-4o 模型配置
MODEL_NAME = "gpt-4.1"
api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量中读取
client = OpenAI(api_key=api_key)

def process_image(image_path, annot_path, dataset, task='basic', output_path='./'):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    if dataset == 'DensePass':
        # generate data directly
        reasoning_text, prompt = gpt_generate_imageonly(image_path, task, MODEL_NAME, client)
    elif dataset == 'Indoor360':
        reasoning_text, prompt = gpt_generate_imageboxes(image_path, annot_path, task, MODEL_NAME, client)
    # extract reasoning data from reasoning_text
    pattern = re.compile(
        r"Question:\s*\n(.*?)\n+"
        r"Reasoning Chain:\s*\n(.*?)\n+"
        r"Final Answer:\s*\n(.*?)(?=\n---|\Z)",
        re.DOTALL
    )
    
    reasoning_data = []
    for match in pattern.finditer(reasoning_text):
        question = match.group(1).strip()
        reasoning_chain = match.group(2).strip()
        final_answer = match.group(3).strip()

        reasoning_data.append({
            "question": question,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer
        })
    
    output = {
        "dataset": dataset,
        "task": task,
        "image_id": image_id,
        "prompt": prompt,
        "reasoning": reasoning_data,
    }
    with open(os.path.join(output_path, f"{dataset}_{image_id}.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    return len(reasoning_data)

def run_pipeline(dataset):
    # create dataset saved dir
    output_path = os.path.join("./output", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # define the path of datasets
    if dataset == "DensePass": # 100 image; 434 data
        image_dir = "../../PanoData/DensePASS/leftImg8bit/val"
        annot_dir = "../../PanoData/DensePASS/gtFine/val"
    elif dataset == "Indoor360":
        image_dir = "../../PanoData/360indoor/images"
        annot_dir = "../../PanoData/360indoor/annotations"
    # count the amount of the generated data
    data_nums = 0
    image_list = os.listdir(image_dir)[:5]
    print(f"{len(image_list)} images will be processed.")
    for fname in tqdm(image_list, desc="Processing"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, fname)
            annot_path = os.path.join(annot_dir, os.path.splitext(fname)[0] + ".json")
            if os.path.exists(image_path):
                data_num = process_image(image_path, annot_path, dataset=dataset, task='basic', output_path=output_path)
                data_nums += data_num
    print(f"{data_nums} generated data has been saved!")

if __name__ == "__main__":
    dataset = "Indoor360" 
    run_pipeline(dataset)
