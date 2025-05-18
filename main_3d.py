import os
import re
import json
import random
import base64
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from utils_3d import *
import psutil

# GPT-4o 模型配置
MODEL_NAME = "gpt-4.1"
client = OpenAI(api_key=None)

def run_pipeline(dataset):
    # create dataset saved dir
    output_path = os.path.join("./output", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    if dataset == "area3":
        image_dir = "../../PanoData/area_3/area_3/pano/rgb"
        annot_dir = "../../PanoData/area_3/area_3/pano/semantic"
        xyz_dir = "../../PanoData/area_3/area_3/pano/global_xyz"
        semantic_path = "../../PanoCode/2D-3D-Semantics/assets/semantic_labels.json"
    # count the amount of the generated data
    data_nums = 0
    image_list = os.listdir(image_dir)[:5]
    print(f"{len(image_list)} images will be processed.")
    for fname in tqdm(image_list, desc="Processing"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, fname)
            if dataset == "area3":
                annot_path = os.path.join(annot_dir, fname.replace('rgb', 'semantic'))
                xyz_path = os.path.join(xyz_dir, fname.replace('rgb.png', 'global_xyz.exr'))
                semantic_labels = load_labels(semantic_path)
                if os.path.exists(image_path):
                    data_num = process_area3(image_path, annot_path, xyz_path, semantic_labels, MODEL_NAME, client, output_path)
                    data_nums += data_num
    print(f"{data_nums} generated data has been saved!")

if __name__ == "__main__":
    dataset = "area3" 
    run_pipeline(dataset)