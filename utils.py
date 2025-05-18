from sklearn.cluster import KMeans
import numpy as np
import random
import math
import json
import cv2
import base64
from openai import OpenAI
from PIL import Image
from prompts import *

def deg2coord(lon, lat, W, H):
    """
    Convert (lon: longitude, lat: latitude) in radians
    to pixel coordinates (x, y) in equirectangular image.
    """
    x = (lon + np.pi) / (2 * np.pi) * W
    y = (np.pi / 2 - lat) / np.pi * H
    return x, y

def ang2coord(w_rad, h_rad, W, H):
    bbox_w = w_rad / 360 * W
    bbox_h = h_rad / 180 * H
    return bbox_w, bbox_h

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def select_object_combination(boxes, W, H, strategy="cluster"):
    """
    输入：
        boxes: list of (c_x, c_y, w, h)
        strategy: 'cluster' | 'central'
    输出：
        selected_boxes: list of selected (c_x, c_y, w, h)
    """
    if len(boxes) < 3:
        return []
    # bfov 转换为 bbox
    trans_boxes = []
    for box in boxes:
        lon, lat, w_rad, h_rad, item = box[2:]
        x, y = deg2coord(lon, lat, W, H)
        w, h = ang2coord(w_rad, h_rad, W, H)
        trans_boxes.append([x, y, w, h, item])
    
    if strategy == "cluster":
        centers = np.array([[c_x, c_y] for c_x, c_y, w, h, _ in trans_boxes])
        n_clusters = max(1, math.ceil(len(trans_boxes) / 4))

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(centers)

        # 按 cluster_id 分组
        clusters = {}
        for idx, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(idx)

        # 过滤掉小于3个的 cluster
        valid_clusters = [idxs for idxs in clusters.values() if len(idxs) >= 3]
        if not valid_clusters:
            return random.sample(trans_boxes, min(len(trans_boxes), 3))

        # 随机选择一个 cluster
        chosen_cluster = random.choice(valid_clusters)
        selected_idxs = chosen_cluster

        # 可限制最多返回 5 个目标
        selected_boxes = [trans_boxes[i] for i in selected_idxs]
        return selected_boxes

    elif strategy == "central":
        # 选择面积最大的 box 作为中心点
        areas = [w * h for c_x, c_y, w, h, _ in trans_boxes]
        central_idx = int(np.argmax(areas))
        c_x0, c_y0, _, _, _ = trans_boxes[central_idx]

        # 计算与其他目标的欧氏距离
        distances = [
            (i, np.linalg.norm([c_x - c_x0, c_y - c_y0]))
            for i, (c_x, c_y, _, _, _) in enumerate(trans_boxes) if i != central_idx
        ]

        # 按距离升序选取最近的 2~4 个邻居
        distances.sort(key=lambda x: x[1])
        neighbor_idxs = [i for i, d in distances[:4]]  # 至多 4 个

        selected_idxs = [central_idx] + neighbor_idxs[:4]
        selected_boxes = [trans_boxes[i] for i in selected_idxs]
        return selected_boxes

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
def gpt_generate_imageonly(image_path, task, model_name, client):
    image = Image.open(image_path)
    image_size = image.size
    base64_image = encode_image_to_base64(image_path)

    if task == 'basic':
        prompt = spatial_position_relationship_prompt.format(image_size)
    else:
        raise ValueError(f"Unknown task: {task}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.8,
        max_tokens=2048,
    )
    return response.choices[0].message.content, prompt

def gpt_generate_imageboxes(image_path, annot_path, task, model_name, client):
    # 1. Select Objects.
    # 2. Plot selected objects boxes on original pano-image.
    # 3. Use different prompt to generate data.
    
    # load image and annot
    image = Image.open(image_path)
    W, H = image.size
    with open(annot_path, 'r') as f:
        annot = json.load(f)
    boxes = annot["boxes"]
    # select object boxes and strategy. 
    box_groups = []
    for _ in range(4):
        bboxes = select_object_combination(boxes, W, H, strategy="cluster")
        box_groups.append(bboxes)
    bboxes = select_object_combination(boxes, W, H, strategy="central")
    box_groups.append(bboxes)
    # plot different color for different cluster
    colors = [
        (0, 0, 255),     # red
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (0, 165, 255),   # orange
        (255, 0, 255)    # magenta
        ]
    np_image = np.array(image)
    for group_idx, group in enumerate(box_groups):
        color = colors[group_idx % len(colors)]
        for (c_x, c_y, w, h, item) in group:
            x1 = int(c_x - w / 2)
            y1 = int(c_y - h / 2)
            x2 = int(c_x + w / 2)
            y2 = int(c_y + h / 2)

            # 绘制矩形框
            cv2.rectangle(np_image, (x1, y1), (x2, y2), color, thickness=2)

            # 在框左上角标注类别名称
            cv2.putText(np_image, item, (x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=color, thickness=2)
    gpt_image = Image.fromarray(np_image)
    image_size = image.size
    base64_image = encode_image_to_base64(image_path)
    # gpt_image.save('/hpc2hdd/home/yhuang489/ylguo/PanoCode/SpatialR360_DG/Check_Code/indoor360.png')
    prompt = object_spatial_relationship_prompt.format(image_size)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.8,
        max_tokens=2048,
    )
    return response.choices[0].message.content, prompt